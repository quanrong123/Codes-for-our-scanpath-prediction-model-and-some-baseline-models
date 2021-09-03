#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from distributions import dirichlet

def expected_stats(nat_param):
    # nat_param: [numgroups, numclass]
    #  exp_stats: [numgroups, numclass]
    exp_stats =  dirichlet.expected_stats(nat_param)
    return exp_stats 

     
def log_partition(nat_param):
    # nat_param: [numgroups, numclass]
    # log_parti: [numgroups]
    log_parti = dirichlet.log_partition(nat_param)
    return log_parti
     