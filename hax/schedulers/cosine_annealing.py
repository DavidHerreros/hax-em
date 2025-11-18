import numpy as np
import optax


class CosineAnnealingScheduler:

    @staticmethod
    def getScheduler(peak_value, total_steps, warmup_frac=None, **kwargs):
        if warmup_frac is None:
            return optax.cosine_decay_schedule(
                init_value=peak_value,
                decay_steps=total_steps,
                alpha=kwargs.pop("alpha", 0.0),   # min relative LR; >0 for a non-zero floor
                )
        else:
            return optax.warmup_cosine_decay_schedule(
                init_value=kwargs.pop("init_value", 0.0),          # LR at step 0
                peak_value=peak_value,                             # LR at end of warmup
                warmup_steps=int(warmup_frac * total_steps),
                decay_steps=total_steps,                           # total length (warmup + cosine)
                end_value=kwargs.pop("end_value", 0.0),            # LR at the very end
                )

    @staticmethod
    def getCyclicScheduler(peak_value, total_steps, num_cycles, cycle_fraction, warmup_frac=None, **kwargs):
        cycle_steps = [int(cycle_fraction * total_steps / (num_cycles - 1)) for _ in range(num_cycles - 1)]
        cycle_steps.append(total_steps - sum(cycle_steps))
        decay_factor = np.linspace(1., 0.5, num_cycles)
        if warmup_frac is None:
            kwargs = [dict(init_value=peak_value * decay_factor[idx],
                           decay_steps=cycle_steps[idx],
                           alpha=kwargs.pop("alpha", 0.0)) for idx in range(num_cycles)]
        else:
            init_value = [kwargs.pop("init_value", 0.0), ] + [0.0 for _ in range(num_cycles - 1)]
            kwargs = [dict(init_value=init_value[idx],                         # LR at step 0
                           peak_value=peak_value * decay_factor[idx],          # LR at end of warmup
                           warmup_steps=int(warmup_frac * cycle_steps[idx]),
                           decay_steps=cycle_steps[idx],                       # total length of one cycle (warmup + cosine)
                           end_value=kwargs.pop("end_value", 0.0),             # LR at the very end
                           ) for idx in range(num_cycles)]
        return optax.sgdr_schedule(cosine_kwargs=kwargs)