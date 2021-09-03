#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from distributions import dirichlet
# vi_stats = vi.expected_stats(vi_params)

def expected_stats(nat_param):
    # nat_param: [numworker, numclass(K), numgroups, numclasses(K)]
    # exp_stats: [numworker, numclass(K), numgroups, numclasses(K)]   
    exp_stats = dirichlet.expected_stats(nat_param)
    return exp_stats     

def log_partition(nat_param):
    # nat_param: [numworker, numclass(K), numgroups, numclasses(K)]
    # log_parti: [numworker, numclass(K), numgroups]
    log_parti = dirichlet.log_partition(nat_param)
    return log_parti
