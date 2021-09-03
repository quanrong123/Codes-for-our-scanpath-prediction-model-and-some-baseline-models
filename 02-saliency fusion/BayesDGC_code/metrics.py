import numpy as np
from sklearn.metrics import hamming_loss, accuracy_score, f1_score,roc_auc_score,average_precision_score,coverage_error,label_ranking_average_precision_score,label_ranking_loss
from sklearn.metrics import precision_score, recall_score, roc_auc_score, adjusted_mutual_info_score


# binary performance measure:
# labels: 1-d array, true label for whole data {0,1}^[NumInst]
# pred_batches: 1-d array, predicted labels for unitl now instances   {0,1} ^[NumInstBatchNow]
def _evaluate_binary(labels, pred_batches):
    preds = np.hstack(pred_batches)
    truths = labels[:preds.size]
    # accuracy
    acc = accuracy_score(truths, preds)
    # precision
    prec = precision_score(truths, preds)
    # recall
    rec = recall_score(truths, preds)
    # f1_score
    f1 = f1_score(truths, preds)
    return acc, prec, rec, f1

def _evaluate_binary_(labels, pred_batches, pred_batches_pb):
    preds = np.hstack(pred_batches)
    preds_pb = np.hstack(pred_batches_pb)
    truths = labels[:preds.size]
    # accuracy
    acc = accuracy_score(truths, preds)
    # precision
    prec = precision_score(truths, preds)
    # recall
    rec = recall_score(truths, preds)
    # f1_score
    f1 = f1_score(truths, preds)
    auc = roc_auc_score(truths, preds_pb)

    return acc, prec, rec, f1, auc

def _evaluate_binary_auc(labels_pb, pred_batches_pb):
    preds_pb = np.hstack(pred_batches_pb)
    truths_pb = labels_pb[:preds_pb.size]
    auc = roc_auc_score(truths_pb, preds_pb)
    return auc

# multiclass performance measure:
# labels: 1-d array, true label for whole data  {label name} ^[NumInst]
# pred_batches: 1-d array, predicted labels for unitl now instances   {label name} ^[NumInstBatchNow]
def _evaluate_multiclass(labels, pred_batches):
    preds = np.hstack(pred_batches)
    truths = labels[:preds.size]
    # accuracy
    acc = accuracy_score(truths, preds)
    # precision_mic
    prec_mic = precision_score(truths, preds, average='micro')
    # recall_mic
    rec_mic = recall_score(truths, preds, average='micro')
    # f1_score _mic
    f1_mic = f1_score(truths, preds, average='macro')
    # precision_mac
    prec_mac = precision_score(truths, preds, average='macro')
    # recall_mac
    rec_mac = recall_score(truths, preds, average='macro')
    # f1_score _mac
    f1_mac = f1_score(truths, preds, average='macro')

    return acc, prec_mic, rec_mic, f1_mic, prec_mac, rec_mac, f1_mac

# multilabel classification performance measure:
# labels: 2-d label indicator matrix, true label for whole data {0,1}^{NumInst, numLabel}
# pred_batches: 2-d label indicator matrix,  predicted labels for unitl now instances    {0, 1} ^[NumInstBatchNow, numLabel]
def _evaluate_multilabel_cls(labels, pred_batches):
    preds = np.hstack(pred_batches)
    truths = labels[:preds.size]
    # hamming loss
    hml = hamming_loss(labels, pred_batches)
    # precision_mic
    prec_mic = precision_score(truths, preds, average='micro')
    # recall_mic
    rec_mic = recall_score(truths, preds, average='micro')
    # f1_score _mic
    f1_mic = f1_score(truths, preds, average='macro')
    # precision_mac
    prec_mac = precision_score(truths, preds, average='macro')
    # recall_mac
    rec_mac = recall_score(truths, preds, average='macro')
    # f1_score _mac
    f1_mac = f1_score(truths, preds, average='macro')

    return hml, prec_mic, rec_mic, f1_mic, prec_mac, rec_mac, f1_mac

# multilabel ranking performance measure:
#labels: 2-d label indicator matrix, sparse matrix, true label for whole data  {0, 1} ^[NumInst, numLabel]
# pred_batches: 2-d label score matrix,  predicted labels probabilities for unitl now instances    [0, 1] ^[NumInstBatchNow, numLabel]
def _evaluate_multilabel_rank(labels, pred_batches):
    preds = np.hstack(pred_batches)
    truths = labels[:preds.size]
    # coverage_error
    covr = coverage_error(truths, preds)
    # label_ranking_loss
    rkl = label_ranking_loss(truths, preds)
    # average_precision_score mic
    avp_mic = average_precision_score(truths, preds, average='micro')
    # roc_auc_score mic
    auc_mic = roc_auc_score(truths, preds, average='micro')
    # average_precision_score mac
    avp_mac = average_precision_score(truths, preds, average='macro')
    # roc_auc_score mac
    auc_mac = roc_auc_score(truths, preds, average='macro')

    return covr, rkl, avp_mic, auc_mic, avp_mac, auc_mac

def _evaluate(pred_batches, labels):
    preds = np.hstack(pred_batches)
    truths = labels[:preds.size]
    acc, _ = cluster_acc(preds, truths)
    nmi = adjusted_mutual_info_score(truths, labels_pred=preds)
    return acc, nmi















