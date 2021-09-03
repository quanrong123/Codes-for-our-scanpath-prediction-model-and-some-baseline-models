import numpy as np
from distributions import dirichlet, vi

def global_expected_stats(global_params):
    dir_params, vi_params = global_params
    dir_stats = dirichlet.expected_stats(dir_params)
    vi_stats = vi.expected_stats(vi_params)
    return dir_stats, vi_stats















