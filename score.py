from sklearn import metrics
import numpy as np
from tqdm import tqdm

def predict(scores, threshold):
    # Predict classes from scores and a decision threshold 
    return scores >= threshold


def fpr(true_classes, scores, threshold):
    # False Positive Rate
    impostor_scores = scores[true_classes == 0]
    return sum(predict(impostor_scores, threshold))/len(impostor_scores)


def tpr(true_classes, scores, threshold):
    # True Positive Rate
    genuine_scores = scores[true_classes == 1]
    return sum(predict(genuine_scores, threshold))/len(genuine_scores)


def f1_score(true_classes, scores, threshold):
    predicted_classes = predict(scores, threshold)
    return metrics.f1_score(true_classes, predicted_classes)


def accuracy_score(true_classes, scores, threshold):
    predicted_classes = predict(scores, threshold)
    return metrics.accuracy_score(true_classes, predicted_classes)


def eer(true_classes, scores, thresholds, return_threshold=False):
    # Equal Error Rate
    far_list = [fpr(true_classes, scores, threshold) for threshold in thresholds]
    frr_list = [1-tpr(true_classes, scores, threshold) for threshold in thresholds]
    min_abs_diff_far_frr = float('inf')
    eer_threshold = None
    eer_far = None
    eer_frr = None
    for threshold, far, frr in zip(thresholds, far_list, frr_list):
        abs_diff_far_frr = abs(far - frr)
        if abs_diff_far_frr < min_abs_diff_far_frr:
            min_abs_diff_far_frr = abs_diff_far_frr
            eer_threshold = threshold
            eer_far = far
            eer_frr = frr
            
    if return_threshold:
        return (eer_far, eer_frr), eer_threshold
    return (eer_far, eer_frr)


def cmc_aux(true_classes, similarity_matrix, rank, recognition=None):
    ranked_matching_scores_indices = np.argsort(-similarity_matrix.to_numpy(), axis=0) # Sort by decreasing similarity in each column
    if recognition is None:
        recognition = np.zeros(true_classes.shape)
    recognition_rate = None
    for i in range(len(true_classes)):
        recognition[i] = recognition[i] or (true_classes[i] in similarity_matrix.iloc[ranked_matching_scores_indices[rank-1, :]].index.to_list())
    recognition_rate = recognition.mean()
    return recognition_rate, recognition


def cmc(true_classes, similarity_matrix, rank):
    # Cumulative Matching Characteristic curve
    assert(rank >= 1)
    cmc_list = []
    recognition = np.zeros(true_classes.shape)
    ranks = np.arange(1, rank+1, 1)
    for rank in tqdm(ranks):
        cmc_element, recognition = cmc_aux(true_classes, similarity_matrix, rank, recognition)
        cmc_list.append(cmc_element)
    return cmc_list