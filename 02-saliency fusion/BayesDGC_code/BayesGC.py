#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BayesGC model for crowd classifcation.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time

from six.moves import range, zip
import tensorflow as tf

from distributions import catgorical, dirichlet, vi
from distributions import exp_family_kl

from shared import global_expected_stats

def z_mean_field(global_stats, z_ik):
    dir_stats, vi_stats = global_stats
    z_nat_param = dir_stats
    z_nat_param += z_ik
    z_stats = catgorical.expected_stats(z_nat_param)
    return z_nat_param, z_stats


def annotation_log_likelihood(z_stats, K, z_ik):
    M = tf.shape(z_stats)[0]
    anno_log = tf.multiply(z_ik, z_stats)
    return tf.reduce_sum(anno_log)

def local_kl_z(z_nat_param, dir_stats, z_stats):
    z_nat_param = z_nat_param - tf.reduce_logsumexp(z_nat_param, axis=-1, keepdims=True)
    nat_param_diff = z_nat_param - dir_stats
    return exp_family_kl(nat_param_diff, z_stats)

def global_kl(prior_global_params, global_params, global_stats, N, n_annotations):
    prior_dir_param, prior_vi_param = prior_global_params

    def _kl_helper(log_partition, param, prior_param, stats):
        nat_diff = param - prior_param
        log_z_diff = log_partition(param) - log_partition(prior_param)
        return exp_family_kl(nat_diff, stats, log_z_diff=log_z_diff)

    dir_param, vi_param = global_params
    dir_stats, vi_stats = global_stats
    dir_kl = _kl_helper(dirichlet.log_partition, dir_param, prior_dir_param, dir_stats)
    vi_kl = _kl_helper(vi.log_partition, vi_param, prior_vi_param, vi_stats)
    return (dir_kl) / N + tf.reduce_mean(vi_kl) / n_annotations

def elbo(local_kl_z, global_kl, ann_ll):
    obj = - local_kl_z - global_kl + ann_ll
    return obj


def get_zik(K, L, vi_stats):
    z_ki_list = []
    for k in range(K):
        z_tmp_list = []
        for l in range(K):
            tmp = tf.reduce_sum(tf.multiply(L[:, l, :], vi_stats[:, k, l]), axis=1)
            z_tmp_list.append(tmp)
        z_tmp = tf.stack(z_tmp_list)
        z_tmp_sum = tf.reduce_sum(z_tmp, axis=0)
        z_ki_list.append(z_tmp_sum)
    z_ki = tf.stack(z_ki_list)
    z_ik = tf.transpose(z_ki, [1, 0])
    return z_ik


def variational_message_passing(
        prior_global_params, global_params, K, W, N,
        L, n_annotations, ann_size_perW):
    global_stats = global_expected_stats(global_params)
    dir_stats, vi_stats = global_stats
    M = tf.shape(L)[0]

    z_ik = get_zik(K, L, vi_stats)

    z_nat_param, z_stats = z_mean_field(global_stats, z_ik)

    tf.summary.histogram('vi_stats_anOly', vi_stats)
    tf.summary.histogram('dir_stats_anOly', dir_stats)
    tf.summary.histogram('z_ik_anOly', z_ik)
    tf.summary.histogram('z_nat_param_anOly', z_nat_param)
    tf.summary.histogram('z_stats_anOly', z_stats)

    # Compute ELBO
    # log_p_ann_term: []
    log_p_ann_term = annotation_log_likelihood(z_stats, K, z_ik)
    ann_batch_size = tf.reduce_sum(L)
    log_p_ann_term = log_p_ann_term / ann_batch_size

    # log_kl_z_term: [M]
    local_kl_z_term = local_kl_z(z_nat_param, dir_stats, z_stats)
    local_kl_z_term = tf.reduce_sum(local_kl_z_term) / (tf.to_float(M))

    # global_kl_term: []
    global_kl_term = global_kl(
        prior_global_params, global_params, global_stats, N, n_annotations)

    log_p_ann_term = log_p_ann_term
    local_kl_z_term = local_kl_z_term
    global_kl_term = global_kl_term

    lower_bound = elbo(
        local_kl_z_term, global_kl_term,
        log_p_ann_term)

    tf.summary.scalar('LB_anOly', lower_bound)

    # Natural gradient for global variational parameters
    # z_stats: [M, K],
    # dir_updates: [K]
    dir_updates = tf.reduce_mean(z_stats, axis=0)

    vi_updates = tf.zeros(vi_stats.shape)
    vi_list_k2 = []
    for k2 in range(K):
        # Ltr: W, K4, M
        Ltr = tf.transpose(L)
        # tmp1:  elementwise product, W,K4,M
        tmp1 = Ltr[:, :, :]
        # tmp2tr: elementwise product, W, K4,M
        tmp2tr = tf.multiply(z_stats[:, k2], tmp1)
        # tmp2: M, K4, W
        tmp2 = tf.transpose(tmp2tr)
        # tmp_sum:  K4, W
        tmp_sum = tf.reduce_sum(tmp2, axis=0)
        # vi_list_gi: list of numgroups items, each item with size K_4,W

        # vi_list_k2: list of K2 items, each item with size numgroups, K_4, W
        vi_list_k2.append(tmp_sum)
        # vi_k2: K2, K4, W
    vi_k2 = tf.stack(vi_list_k2)

    ann_batch_size_perW = tf.reduce_sum(L, axis=(0, 1)) + 1e-8
    # elmentwise devision
    vi_k2 = vi_k2 / ann_batch_size_perW
    # vi_updates:  W, K2, K4
    vi_updates = tf.transpose(vi_k2, [2, 0, 1])

    updates = [dir_updates, vi_updates]
    tf.summary.histogram('dir_updates_anOly', dir_updates)
    tf.summary.histogram('vi_updates_anOly', vi_updates)

    # updates.append(vi_updates)
    nat_grads = [(prior_global_params[i] - global_params[i]) / N + updates[i]
                 for i in range(1)]
    nat_grads.append(
        tf.transpose(tf.transpose(prior_global_params[1] - global_params[1]) / (ann_size_perW + 1e-8)) + updates[1])

    # dir_stats, niw_stats, vi_stats, pi_stats = global_stats
    v_global_params = global_params
    v_global_stats = global_stats

    return lower_bound, nat_grads, log_p_ann_term, -local_kl_z_term, -global_kl_term, z_stats, v_global_params, v_global_stats


























