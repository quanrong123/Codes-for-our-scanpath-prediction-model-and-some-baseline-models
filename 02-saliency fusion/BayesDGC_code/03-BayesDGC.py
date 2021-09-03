#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BayesDGC model for crowd classifcation.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time
import random
from six.moves import range, zip
import tensorflow as tf
import numpy as np
import zhusuan as zs
from metrics import _evaluate_binary, _evaluate_binary_, _evaluate_multiclass, _evaluate_multilabel_cls, _evaluate_multilabel_rank

from utils import dataset, setup_logger
from distributions import niw, catgorical, mvn, dirichlet, vi, pi
from distributions import normalize, exp_family_kl

import scipy.io as scio

import BayesGC as BayesGC
from shared import global_expected_stats
import math

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("seed", 1234, """Random seed.""")


def get_global_params(scope, d, K, W, alpha, a_v=5., b_v=2., random_scale=None, trainable=True):
    # alpha: [K]

    def init_dir_param():
        # TODO: add randomness for variational
        #
        #  dir_pa: vactor of length
        # [K], row vector
        if random_scale:
            dir_nat_init = tf.random_uniform([K], minval=0, maxval=1.)
        else:
            dir_nat_init = alpha  # tf.ones([K]) * (alpha - 1.)
        return dir_nat_init

    def init_vi_param():
        # a_v=5, b_v=2,
        # beta_kl: dirichilete dimension, K*K
        # vi_nat_init: [numworker, numclass,  numclass]
        # [numworker, numclass, numclass]
        beta_init = tf.eye(K) * (a_v - b_v) + b_v

        vi_nat_init = tf.zeros([W, K, K]) + beta_init[None, :, :] - 1.
        return vi_nat_init

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        # dir_nat: K
        dir_nat = init_dir_param()
        # [K]: row vector
        dir_params = tf.get_variable(
            "dir_params", dtype=tf.float32, initializer=dir_nat,
            trainable=trainable)

        # [W, K, K]  confusion matrix vi [numworker, numclass, numclass]
        vi_nat = init_vi_param()
        vi_params = tf.get_variable(
            "vi_params", dtype=tf.float32, initializer=vi_nat,
            trainable=trainable)

    return dir_params, vi_params


@zs.reuse("classifier")
def classifier(scope, o, n_class):
    h = tf.layers.dense(tf.to_float(o), 100, activation=tf.nn.relu)
    z_logits = tf.layers.dense(h, n_class)

    z_dist = zs.distributions.OnehotCategorical(z_logits, group_ndims=1)
    z = tf.nn.softmax(z_logits) + 1e-8
    z /= tf.reduce_sum(z, 1, keepdims=True)
    log_z = tf.log(z)
    return z_dist, z_logits, z, log_z


def z_mean_field(global_stats, x_obs_param, z_ik=None):
    # dir_stats: [K]
    dir_stats, vi_stats = global_stats

    z_nat_param = dir_stats + x_obs_param

    ## update of annotation
    if z_ik is not None:
        z_nat_param += z_ik
    # z_stats: [M, K]
    z_stats = catgorical.expected_stats(z_nat_param)
    return z_nat_param, z_stats


def annotation_log_likelihood(z_stats, K, z_ik):
    # z_stats: [M, K],
    M = tf.shape(z_stats)[0]

    # element-wise product
    # anno_log: [M, K]
    anno_log = tf.multiply(z_ik, z_stats)

    return tf.reduce_sum(anno_log)

def local_kl_z(z_nat_param, dir_stats,  x_obs_param, z_stats):
    # z_nat_param: [M, K]
    # dir_stats: [K]
    # z_stats: [M, K]
    z_nat_param = z_nat_param - tf.reduce_logsumexp(z_nat_param, axis=-1,keepdims=True)
    nat_param_diff = z_nat_param - dir_stats - x_obs_param
    # ret: [M]
    return exp_family_kl(nat_param_diff, z_stats)


def global_kl(prior_global_params, global_params, global_stats, d, N, n_annotations=None):
    prior_dir_param, prior_vi_param = prior_global_params

    def _kl_helper(log_partition, param, prior_param, stats):
        nat_diff = param - prior_param
        log_z_diff = log_partition(param) - log_partition(prior_param)
        return exp_family_kl(nat_diff, stats, log_z_diff=log_z_diff)

    # dir_param: [K]
    # vi_params: [W,K,K]
    dir_param, vi_param = global_params
    # dir_stats: [K]
    # vi_stats: [W,K,K],
    dir_stats, vi_stats = global_stats
    # dir_kl: []
    dir_kl = _kl_helper(dirichlet.log_partition, dir_param, prior_dir_param,
                        dir_stats)

    # vi_kl: [W,K]
    vi_kl = _kl_helper(vi.log_partition, vi_param,
                       prior_vi_param, vi_stats)

    if n_annotations is not None:
        return (dir_kl) / N + (tf.reduce_mean(vi_kl)) / n_annotations
    else:
        return (dir_kl) / N


def elbo(local_kl_z, global_kl, ann_ll):
    obj = - local_kl_z - global_kl + ann_ll
    return obj


def get_zik(K, L, vi_stats):
    z_ki_list = []
    for k in range(K):
        z_tmp_list = []
        for l in range(K):
            # tmp: M
            tmp = tf.reduce_sum(tf.multiply(L[:, l, :], vi_stats[:, k, l]), axis=1)
            # tmp=tf.sparse_tensor_dense_matmul(L[:,l,:], vi_stats[:, k, :, l])
            z_tmp_list.append(tmp)
            # z_tmp : K,  M
        z_tmp = tf.stack(z_tmp_list)
        # z_tmp_sum :  M
        z_tmp_sum = tf.reduce_sum(z_tmp, axis=0)
        z_ki_list.append(z_tmp_sum)
        # z_kig : K, M,1
    z_ki = tf.stack(z_ki_list)
    # z_ikg : M, K,1
    z_ik = tf.transpose(z_ki, [1, 0])
    return z_ik


def variational_message_passing(scope,
                                prior_global_params, global_params, o, o_dim, d, K, W, N,
                                L=None, n_annotations=None, ann_size_perW=None):
    global_stats = global_expected_stats(global_params)
    dir_stats, vi_stats = global_stats
    M = tf.shape(o)[0]

    # classifier: o --> z
    # classifier(o, n_class):
    _, qy_logits, _, _ = classifier(scope, o, K)  # qy_x(x_unlabeled_ph, n_class)
    qy = tf.nn.softmax(qy_logits)
    qy /= tf.reduce_sum(qy, 1, keepdims=True)
    log_qy = tf.log(qy)

    x_obs_param = log_qy

    if L is not None:
        z_ik = get_zik(K, L, vi_stats)
    else:
        z_ik = None

    z_nat_param, z_stats = z_mean_field(global_stats, x_obs_param, z_ik=z_ik)

    # log_p_ann_term: []
    if L is not None:
        # z_stats: [M, K], z_inner_stats: [M, M]
        log_p_ann_term = annotation_log_likelihood(z_stats, K, z_ik)
        # ann_subsample_factor = n_ann / ann_batch_size
        ann_batch_size = tf.reduce_sum(L)
        # normalization, get the logprob of each annotation
        log_p_ann_term = log_p_ann_term / ann_batch_size
    else:
        log_p_ann_term = 0

    # log_kl_z_term: [M]
    # log_kl_z_term: [M]
    local_kl_z_term = local_kl_z(z_nat_param, dir_stats, x_obs_param, z_stats)
    local_kl_z_term = tf.reduce_sum(local_kl_z_term) / (tf.to_float(M))

    # global_kl_term: []
    global_kl_term = global_kl(prior_global_params, global_params, global_stats, d, N, n_annotations)

    log_p_ann_term = log_p_ann_term
    local_kl_z_term = 0.5 * local_kl_z_term
    global_kl_term = global_kl_term

    lower_bound = elbo(
        local_kl_z_term, global_kl_term,
        log_p_ann_term)

    # Natural gradient for global variational parameters
    # z_stats: [M, K]
    # dir_updates: [K]
    dir_updates = tf.reduce_mean(z_stats, axis=0)

    if L is not None:
        #  L:   [M, K, W] (numItem, numClass, numWorker) {0,1} , L(i,k,w) = 1 means worker w annotated label k to instance i
        # vi_updates:  [W, K, numgroup, K]  (numWorker, numClass, numgroup, numClass)
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

        #
        # [W]
        ann_batch_size_perW = tf.reduce_sum(L, axis=(0, 1)) + 1e-8
        # elmentwise devision
        vi_k2 = vi_k2 / ann_batch_size_perW
        # vi_updates:  W, K2, K4
        vi_updates = tf.transpose(vi_k2, [2, 0, 1])

        # updates.append(vi_updates)
        updates = [dir_updates, vi_updates]

        nat_grads = [(prior_global_params[i] - global_params[i]) / N + updates[i]
                     for i in range(1)]
        # in case of some worker didn't annotate any instance , i.e., ann_size_perW[widx] =0, should not update the parameters of widx worker
        # this should be done in the data preprocessing step, filter out nonvalid workers

        nat_grads.append(
            tf.transpose(tf.transpose(prior_global_params[1] - global_params[1]) / (ann_size_perW + 1e-8)) + updates[1])

    else:
        nat_grads = [(prior_global_params[i] - global_params[i]) / N + updates[i]
                     for i in range(len(updates))]

    v_global_params = global_params
    v_global_stats = global_stats

    return qy, log_qy, lower_bound, nat_grads, log_p_ann_term, -local_kl_z_term, -global_kl_term, z_stats, v_global_params, v_global_stats

def variational_message_passing_sup(scope,o, o_dim, d, K, t):
    # Build classifier
    _, qy_logits_l, _, _ = classifier(scope,o, K)
    qy_l = tf.nn.softmax(qy_logits_l)
    pred_y = tf.argmax(qy_l, 1)
    acc = tf.reduce_sum(
        tf.cast(tf.equal(pred_y, tf.argmax(t, 1)), tf.float32) /
        tf.cast(tf.shape(o)[0], tf.float32))
    onehot_cat = zs.distributions.OnehotCategorical(qy_logits_l)
    log_qy_x = onehot_cat.log_prob(t)
    lower_bound =tf.reduce_mean(log_qy_x)

    return lower_bound, qy_l, acc



feapath = '/disk4/quanrong/rebuttal_2021/MIT/sp_features_2000_1/'
labelpath = '/disk4/quanrong/rebuttal_2021/MIT/label_data_2000_1/'
feapathdir = os.listdir(feapath)
feapathdir.sort()
labelpathdir = os.listdir(labelpath)
labelpathdir.sort()
num_imgs = len(feapathdir)

epochs = 600
result_path = './results/BayesDGC1/'
if not os.path.exists(result_path):
    os.mkdir(result_path)
res_path = '/disk4/quanrong/rebuttal_2021/MIT/BayesDGC_result/'
if not os.path.exists(res_path):
    os.mkdir(res_path)
res_path1 = '/disk4/quanrong/rebuttal_2021/MIT/BayesDGC_result_GC/'
if not os.path.exists(res_path1):
    os.mkdir(res_path1)

logger = setup_logger('Epoch' + str(1), __file__, result_path)
image_sp = []
fvFeatOrg = []
GT = []
selAnnOrg = []
for index in range(900,950):
    image_name = labelpathdir[index]
    res_name = os.path.join(res_path, image_name)
    print(res_name)
    fea_path = os.path.join(feapath, image_name)
    label_path = os.path.join(labelpath, image_name)
    sal_fea = scio.loadmat(fea_path)
    sal_fea = sal_fea['sp_features']
    labels = scio.loadmat(label_path)
    candi_l = labels['L']
    #gt_labels = labels['true_labels']

    fvFeatOrg.append(sal_fea)
    selAnnOrg.append(candi_l)
    #GT.append(gt_labels)
    image_sp.append(sal_fea.shape[0])
#print(image_sp)
#print(np.sum(image_sp))
fvFeatOrg = np.concatenate(fvFeatOrg)  # num_Inst * 1475
selAnnOrg = np.concatenate(selAnnOrg)  # num_Inst * 4
#GT = np.concatenate(GT)  # num_Inst * 1

x_train = fvFeatOrg
x_test = x_train
x_train, x_test, mean_x_train, std_x_train = dataset.standardize(x_train, x_test)
fvFeat = x_train
#G_l = (GT == 1)  # indicator label matrix: 1 positive,  0 negative

nat_grad_scale = 1e4

#t_train = G_l  # num_sp * 1
o_train = fvFeat  # num_sp * 1475
n_train, o_dim = o_train.shape  # n_train: num_Inst,

#t_test = t_train

numInst, numWorker = selAnnOrg.shape
W = numWorker
K = 2
L_w = np.zeros([numInst, K, numWorker])
L_w[:, 0, :] = (selAnnOrg == 1)  # column index: 0 negative
L_w[:, 1, :] = (selAnnOrg == 2)  # column index: 1 positive
annotations = L_w  # num_sp * 2 * 4

# the number of instances
ann_inst_size = annotations.shape[0]  # num_sp
# the labels each Worker has given
ann_size_perW = annotations.sum(axis=(0, 1))

n_annotations = annotations.sum()  # num_sp*4
o_annotations = o_train  # num_sp * num_fea, 584*1475
#t_ann = t_train  # num_sp * 1, 1 for positive, 0 for negative

ann_test = annotations  # num_sp*2*4

seed = FLAGS.seed
np.random.seed(seed)
tf.set_random_seed(seed)

# prior parameters
d = 100
# prior_alpha = 1.1
prior_alpha = [0.1, 0.03]
prior_alpha = tf.cast(prior_alpha, tf.float32)

prior_a_v = 5.
prior_b_v = 2.

alpha = [1, 1]
alpha = tf.cast(alpha, tf.float32)

a_v = 5
b_v = 2
random_scale = 3.

iters = annotations.shape[0] // ann_inst_size
batch_size = o_train.shape[0] // iters

# Define training parameters
ann_batch_size = ann_inst_size  # num_sp

global_step = tf.Variable(0, trainable=False, name="global_step")
starter_learning_rate = 0.001
learning_rate = tf.train.cosine_decay_restarts(starter_learning_rate, global_step, 1, t_mul=2.0, m_mul=1.0, alpha=0.0, name=None)
prior_global_params = get_global_params('prior_epoch_' + str(1), d, K, W, prior_alpha, prior_a_v, prior_b_v, trainable=False)
global_params = get_global_params('variational_epoch_' + str(1), d, K, W, alpha, a_v, b_v, random_scale=random_scale, trainable=True)

ox = tf.placeholder(tf.float32, shape=[None, o_dim], name='ox' + 'epoch' + str(1))
ann_o = tf.placeholder(tf.float32, shape=[None, o_dim], name='ann_o' + 'epoch' + str(1))
L_ph = tf.placeholder(tf.float32, shape=[None, K, W], name='L_ph' + 'epoch' + str(1))
t_tr = tf.placeholder(tf.int32, shape=[None, K], name='t_tr' + 'epoch' + str(1))
# BayesGC_declariation
lower_bound_annOnly, nat_grads_annOnly, log_ann_annOnly, kl_z_annOnly, global_kl_annOnly, z_stats_annOnly, v_global_params_annOnly, v_global_stats_annOnly \
    = BayesGC.variational_message_passing(prior_global_params, global_params, K, W, ann_inst_size, L_ph, n_annotations, ann_size_perW)

z_pred_annOnly = tf.argmax(z_stats_annOnly, axis=-1)

optimizer_annOnly = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
nat_grads_annOnly_ = [-nat_grad_scale * 0.5 * (g_annOnly) for g_annOnly in (nat_grads_annOnly)]
global_grads_and_vars_annOnly = list(zip(nat_grads_annOnly_, global_params))
#if index == 0:
infer_op_preAnnOnly = optimizer_annOnly.apply_gradients(global_grads_and_vars_annOnly)

# BayesDGC dnn model pretrain declartion
scopename = 'd' + 'epoch' + str(1)
lower_bound_sup, z_stats_sup, z_acc_sup = variational_message_passing_sup(scopename, ox, o_dim, d, K, t_tr)
z_pred_sup = tf.argmax(z_stats_sup, axis=-1)
optimizer_sup = tf.train.AdamOptimizer(learning_rate)
net_vars_sup = (tf.trainable_variables(scope="classifier"))
net_grads_and_vars_sup = optimizer_sup.compute_gradients(0.5 * (-lower_bound_sup), var_list=net_vars_sup)

infer_op_sup = optimizer_sup.apply_gradients(net_grads_and_vars_sup)

# BayesDGC declaration:
qy, log_qy, lower_bound, nat_grads, log_p_ann, local_kl_z, global_kl, z_stats, v_global_params, v_global_stats \
     = variational_message_passing(scopename, prior_global_params, global_params, ann_o, o_dim, d, K, W,
                                      ann_inst_size, L_ph, n_annotations, ann_size_perW)
z_pred = tf.argmax(z_stats, axis=-1)  # colum index: 0 negative, 1 postive
optimizer = tf.train.AdamOptimizer(learning_rate)  # binary vae, AdamOptimizer best, no exception, fixed learnrate 1e-3,  -940
net_vars = (tf.trainable_variables(scope="classifier"))
net_grads_and_vars = optimizer.compute_gradients(-(0.5 * lower_bound), var_list=net_vars)
nat_grads_ = [-nat_grad_scale * (g) for g in nat_grads]
global_grads_and_vars = list(zip(nat_grads_, global_params))
infer_op = optimizer.apply_gradients(net_grads_and_vars + global_grads_and_vars)

## Initialization
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# BayesGC training
lblast = 0
for epoch in range(epochs):
    time_epoch = -time.time()
    #indices_ann = np.random.permutation(ann_inst_size)
    #annotations_raw = annotations[indices_ann]  # labels
    #t_train_raw = t_train[indices_ann]  # fea
    annotations_raw = annotations

    lbs = []
    t_probs = []
    t_preds = []

    log_anns = []
    kl_zs = []
    global_kls = []


    for t in range(1):
        # With annotation
        ann_batch = annotations_raw[t * ann_batch_size:(t + 1) * ann_batch_size, :, :]
        _, lb, t_prob, t_pred, t_log_ann, t_kl_z, t_global_kl = sess.run([infer_op_preAnnOnly, lower_bound_annOnly,
                                                                          z_stats_annOnly, z_pred_annOnly,
                                                                          log_ann_annOnly,
                                                                          kl_z_annOnly, global_kl_annOnly],
                                                                         feed_dict={L_ph: ann_batch})
        lbs.append(lb)
        t_probs.append(t_prob[:, 1])
        t_preds.append(t_pred)

        log_anns.append(t_log_ann)
        kl_zs.append(t_kl_z)

        global_kls.append(t_global_kl)

    time_epoch += time.time()
    #logger.info('>>> Train  Epoch {} epoch {} ({:.1f}s): ann LB = {}, log_ann={}, kl_z={},global_kl={} '
    #    .format(ip+1, epoch, time_epoch, np.mean(lbs), np.mean(log_anns), np.mean(kl_zs), np.mean(global_kls)))

    #train_acc, train_pre, train_rec, train_f, train_auc = _evaluate_binary_(t_train_raw, t_preds, t_probs)
    #logger.info('>>> Train  Epoch {} epoch {} ({:.1f}s): ann LB = {}, log_ann={}, kl_z={},  global_kl={} '
    #    'acc = {}, precision = {}, recall = {}, f1 = {}, auc={} '
    #    .format(1, epoch, time_epoch, np.mean(lbs), np.mean(log_anns), np.mean(kl_zs), np.mean(global_kls),
    #            train_acc, train_pre, train_rec, train_f, train_auc))

    #if np.abs(((lblast - np.mean(lbs)) / np.mean(lbs))) < 1e-4:
    #    break
    #lblast = np.mean(lbs)

test_lb_o, test_t_prob_o, test_t_pred_o, test_t_global_params, test_t_global_stats = sess.run(
    [lower_bound_annOnly, z_stats_annOnly, z_pred_annOnly, v_global_params_annOnly, v_global_stats_annOnly],
    feed_dict={L_ph: ann_test})


n0 = 0
for index1 in range(900,950):
    image_name = labelpathdir[index1]
    res_name1 = os.path.join(res_path1, image_name)
    print(res_name1)
    num_sp = image_sp[index1-900]
    prediction_binary = t_pred[n0:n0+num_sp]
    print(prediction_binary)
    prediction_s = t_prob[:,1][n0:n0+num_sp]
    print(prediction_s)
    scio.savemat(res_name1, {'prediction_b':prediction_binary, 'prediction_s':prediction_s})
    n0 = n0 + num_sp
    print(n0)








labEstpb_o = []
labEstpb_o.append(test_t_prob_o[:, 1])
labEst_o = []
labEst_o.append(test_t_pred_o)
#target = t_test
global_params_v = test_t_global_params
global_stats_v = test_t_global_stats

#perform_o = np.zeros([5])
#perform_o[0], perform_o[1], perform_o[2], perform_o[3], perform_o[4] = _evaluate_binary_(target, labEst_o, labEstpb_o)

#name = 'BayesGC_Epoch' + str(1) + 'perform.mat'

#name = os.path.join(result_path, name)
#scio.savemat(name, {'perform_o': perform_o, 'labEst_o': labEst_o, 'labEstpb_o': labEstpb_o, 'target': target,
#                    'global_params_v': global_params_v, 'global_stats_v': global_stats_v})

# pretrain BayesDGC's dnn model usgin resutls of BayesGC
est_lb, est_z_prob, est_t_pred = sess.run([lower_bound_annOnly, z_stats_annOnly, z_pred_annOnly], feed_dict={L_ph: annotations})

t_train_ev_est = est_t_pred  # {0,1}^ [numInst]
t_train_est = np.zeros([ann_inst_size, K])  # {0,1}^ [numInst, K]
t_train_est[:, 0] = (t_train_ev_est == 0)  # negtive
t_train_est[:, 1] = (t_train_ev_est == 1)  # postive

lblast = 0
for epoch in range(epochs):
    time_epoch = -time.time()
    #indices = np.random.permutation(n_train)

    o_annotations_raw = o_annotations#[indices]
    t_train_raw = t_train_est#[indices]
    t_train_ev_raw = t_train_ev_est#[indices]

    lbs = []
    t_probs = []
    t_preds = []
    t_accs = []
    for t in range(iters):
        # With annotation
        o_batch = o_annotations_raw[t * batch_size:(t + 1) * batch_size]
        t_batch = t_train_raw[t * batch_size:(t + 1) * batch_size]

        _, lb, t_prob, t_pred = sess.run([infer_op_sup, lower_bound_sup, z_stats_sup, z_pred_sup], feed_dict={ox: o_batch, t_tr: t_batch})

        lbs.append(lb)
        t_probs.append(t_prob[:, 1])
        t_preds.append(t_pred)

    time_epoch += time.time()
    train_acc, train_pre, train_rec, train_f = _evaluate_binary(t_train_ev_raw, t_preds)

    logger.info( 'Epoch {} epoch {} ({:.1f}s): LB = {}, acc = {}, precision = {}, recall = {}, f1 = {}'
            .format(1, epoch, time_epoch, np.mean(lbs), train_acc, train_pre, train_rec, train_f))

    #if np.abs(((lblast - np.mean(lbs)) / np.mean(lbs))) < 1e-3:
    #    break
    #lblast = np.mean(lbs)
n = 0
for index1 in range(900,950):
    image_name = labelpathdir[index1]
    res_name = os.path.join(res_path, image_name)
    print(res_name)
    num_sp = image_sp[index1-900]
    prediction_binary = t_pred[n:n+num_sp]
    print(prediction_binary)
    prediction_s = t_prob[:,1][n:n+num_sp]
    print(prediction_s)
    scio.savemat(res_name, {'prediction_b':prediction_binary, 'prediction_s':prediction_s})
    n = n + num_sp
    print(n)

