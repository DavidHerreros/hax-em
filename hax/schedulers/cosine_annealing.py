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
                peak_value=peak_value,      # LR at end of warmup
                warmup_steps=int(warmup_frac * total_steps),
                decay_steps=total_steps, # total length (warmup + cosine)
                end_value=kwargs.pop("end_value", 0.0),           # LR at the very end
                )
