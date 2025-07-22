import math


def cosine_annealing_lr_schedule(
    t: int,
    lr_min: float,
    lr_max: float,
    warmup_iterations: int,
    annealing_iterations: int,
) -> float:
    if t < warmup_iterations:
        return lr_max * t / warmup_iterations
    elif t > annealing_iterations:
        return lr_min
    else:
        return lr_min + 0.5 * (
            1
            + math.cos(
                (t - warmup_iterations)
                / (annealing_iterations - warmup_iterations)
                * math.pi
            )
        ) * (lr_max - lr_min)
