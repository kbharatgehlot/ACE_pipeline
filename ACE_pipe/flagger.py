import numpy as np
from tqdm import tqdm
from copy import copy


def flag_bad_baselines(bad_baselines, flags, ant1, ant2):
    updated_flags = copy(flags)
    for i, bl_pair in enumerate(tqdm(bad_baselines)):
        bl_idx = (ant1 == bl_pair[0]) * (ant2 == bl_pair[1])
        updated_flags[:, bl_idx, :, :] = 1
    return updated_flags


def flag_bad_times(bad_times_idx, flags):
    updated_flags = copy(flags)

    for i, t_idx in enumerate(tqdm(bad_times_idx)):
        updated_flags[t_idx, :, :, :] = 1
    return updated_flags

def flag_bad_freqs(bad_freqs_idx, flags):
    updated_flags = copy(flags)

    for i, f_idx in enumerate(tqdm(bad_freqs_idx)):
        updated_flags[:, :, f_idx, :] = 1
    return updated_flags

