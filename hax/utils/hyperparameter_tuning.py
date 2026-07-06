import jax
import jax.numpy as jnp
from hax.utils.loggers import bcolors


def _device_memory_stats(device):
    """``(bytes_limit, bytes_in_use)`` for the device, or ``None`` if unavailable.

    Reads them from the device's ``memory_stats`` (available on the GPU/TPU
    backends when ``XLA_PYTHON_CLIENT_PREALLOCATE=false``, which the hax CLI
    always sets). ``bytes_limit`` is the ceiling XLA enforces before raising
    ``Resource exhausted`` (the real OOM boundary); ``bytes_in_use`` is what is
    already resident (model params, optimizer state, metadata, ...). Their
    difference is the memory actually still available for a training step, which
    is what the batch must be sized against. Returns ``None`` on backends that do
    not expose the stats (e.g. CPU) so the caller can fall back to a fixed default.
    """
    try:
        stats = device.memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    limit = stats.get("bytes_limit") or stats.get("bytes_reservable_limit")
    if not limit:
        return None
    return int(limit), int(stats.get("bytes_in_use", 0))


def _pytree_nbytes(tree):
    """Total bytes of the array leaves of ``tree`` (nnx State, dict, ...)."""
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        nbytes = getattr(leaf, "nbytes", None)
        if nbytes is None:
            nbytes = getattr(getattr(leaf, "value", None), "nbytes", None)
        if nbytes is not None:
            total += int(nbytes)
    return total


def _peak_bytes(step, graphdef, state, md, rng, batch_size, input_shape_per_sample, step_kwargs):
    """Peak device memory (bytes) one ``step`` call would need at ``batch_size``.

    Ahead-of-time compiles ``step`` for a batch of that size **without running
    it** and reads XLA's own memory accounting. Only the two batch-dependent
    arguments (images and their integer labels) are passed as abstract
    ``ShapeDtypeStruct`` placeholders, so no image data is allocated; the real
    ``state`` (params + optimizer) and ``md`` columns are reused as-is. Compiling
    -- unlike executing -- does not allocate the working buffers, so this probes
    the memory cost without consuming it or provoking an OOM.

    Returns ``None`` if the backend does not provide a memory analysis.
    """
    x_abstract = jax.ShapeDtypeStruct((batch_size,) + tuple(input_shape_per_sample), jnp.float32)
    labels_abstract = jax.ShapeDtypeStruct((batch_size,), jnp.int32)

    compiled = step.lower(graphdef, state, x_abstract, labels_abstract, md, rng, **step_kwargs).compile()
    analysis = compiled.memory_analysis()
    if analysis is None:
        return None

    # argument (inputs incl. params/optimizer/md) + temp (activations, autodiff
    # scratch) + output (updated state). Aliasing between input and output state
    # is ignored on purpose so the estimate errs on the conservative/high side.
    return int(analysis.argument_size_in_bytes
               + analysis.temp_size_in_bytes
               + analysis.output_size_in_bytes)


def estimate_batch_size(graphdef, state, step, md, rng, input_shape_per_sample,
                        *, probe_sizes=(32, 128), safety=0.7, reserved_bytes=0,
                        multiple_of=8, min_batch=1, max_batch=1024,
                        device=None, step_kwargs=None, verbose=True):
    """Estimate a memory-safe batch size analytically, without running the step.

    The device memory one training ``step`` needs is affine in the batch size
    ``B``. This ahead-of-time compiles ``step`` at two batch sizes and reads XLA's
    ``memory_analysis`` (see :func:`_peak_bytes` -- compilation only, no execution,
    no image data allocated, no OOM), fits the peak-vs-``B`` line, and solves for
    the largest ``B`` whose predicted peak stays under ``safety`` x total device
    memory (minus ``reserved_bytes``). The result is rounded down to a multiple of
    ``multiple_of`` and clamped to ``[min_batch, max_batch]``.

    Why the conservative ``safety`` default and larger ``probe_sizes``: XLA's
    ``memory_analysis`` reports an *idealized* buffer-assignment total that on the
    GPU backend runs ~10-20% below the real allocator peak (fragmentation), and it
    is mildly *super-linear* in ``B`` -- so probing at tiny batches underestimates
    the slope. Probing in the tens-to-low-hundreds captures a representative slope,
    and budgeting to ~70% of memory absorbs the fragmentation gap plus the extra
    GPU memory the *surrounding* training loop uses but this isolated probe cannot
    see (an EMA copy, a second optimizer, cached decode/clustering kernels, input
    pipeline buffers). ``reserved_bytes`` lets the caller subtract known extras
    (e.g. the EMA buffer). The goal is "a big batch that reliably fits", not the
    theoretical maximum -- a slightly small batch is vastly cheaper than an OOM.

    This sizes for *throughput / GPU utilization*; it does not claim to be the
    accuracy-optimal batch size (larger batches can generalize worse and give
    fewer updates per epoch), so keep ``max_batch`` as a sane cap.

    Returns the chosen batch size (int), or ``None`` if the estimate could not be
    made (unsupported backend, tracing failure, ...) so the caller can fall back
    to a fixed default.
    """
    step_kwargs = dict(step_kwargs or {})
    device = device or jax.devices()[0]
    gib = 1024 ** 3

    if verbose:
        print(f"{bcolors.OKCYAN}\n###### Automatic batch size estimation (analytical)... ######{bcolors.ENDC}")

    mem = _device_memory_stats(device)
    if mem is None:
        if verbose:
            print(f"{bcolors.WARNING}  Could not read device memory stats; skipping automatic estimation.{bcolors.ENDC}")
        return None
    bytes_limit, bytes_in_use = mem

    probes = sorted(set(int(b) for b in probe_sizes if int(b) >= 1))
    if len(probes) < 2:
        raise ValueError(f"estimate_batch_size needs at least two distinct probe_sizes, got {probe_sizes!r}")

    def _try_probe(b):
        """memory_analysis peak at batch ``b``, or None if it could not be taken.

        Compilation never allocates the batch's working buffers, so this cannot
        OOM on batch size. The only failure mode is XLA's GEMM autotuner
        benchmarking kernels at a very large batch on a small GPU; we treat any
        such failure as "probe unavailable" and let the caller back off.
        """
        try:
            return _peak_bytes(step, graphdef, state, md, rng, b, input_shape_per_sample, step_kwargs)
        except Exception:
            return None

    # Low probe first: if even this cannot be analyzed, we cannot estimate at all.
    lo = probes[0]
    p_lo = _try_probe(lo)
    if p_lo is None:
        if verbose:
            print(f"{bcolors.WARNING}  Could not analyze the step at batch {lo}; "
                  f"falling back to a fixed batch size.{bcolors.ENDC}")
        return None

    # High probe: back it off (halving toward the low probe) until one is
    # analyzable, so a small GPU that trips the autotuner at the largest probe
    # still yields a valid second point instead of dropping the estimate.
    b_hi = p_hi = None
    hi = probes[-1]
    while hi > lo:
        p = _try_probe(hi)
        if p is not None:
            b_hi, p_hi = hi, p
            break
        hi //= 2
    if b_hi is None:
        if verbose:
            print(f"{bcolors.WARNING}  Could not analyze the step above batch {lo}; "
                  f"falling back to a fixed batch size.{bcolors.ENDC}")
        return None

    (b1, p1), (b2, p2) = (lo, p_lo), (b_hi, p_hi)
    slope = (p2 - p1) / (b2 - b1)
    intercept = p1 - slope * b1

    usable = safety * bytes_limit - reserved_bytes - intercept
    if slope <= 0:
        # Peak barely grows with the batch: the cap is the only sensible limit.
        estimate = max_batch
    elif usable <= 0:
        # Even the fixed cost already exceeds the safe budget.
        estimate = min_batch
    else:
        estimate = int(usable / slope)

    rounded = (estimate // multiple_of) * multiple_of
    rounded = max(min_batch, min(rounded, max_batch))

    if verbose:
        predicted = (intercept + slope * rounded) / gib
        budget = (safety * bytes_limit - reserved_bytes) / gib
        print(f"  Device memory: {bytes_limit / gib:.2f} GiB total, {bytes_in_use / gib:.2f} GiB already in use")
        print(f"  Budget: {safety:.0%} of total minus {reserved_bytes / gib:.2f} GiB reserved = {budget:.2f} GiB")
        print(f"  Analyzed peak at batch {b1}: {p1 / gib:.2f} GiB | at batch {b2}: {p2 / gib:.2f} GiB "
              f"(per-sample ~{slope / (1024 ** 2):.1f} MiB)")
        print(f"  Selected batch size: {bcolors.OKGREEN}{rounded}{bcolors.ENDC} "
              f"(predicted peak ~{predicted:.2f} GiB, capped at {max_batch})")

    return rounded
