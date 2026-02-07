"""
Evaluation metrics for deepfake detection.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def compute_auc(y_true, y_scores):
    """
    Compute Area Under the ROC Curve (AUC).
    
    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_scores: Predicted scores or probabilities
    
    Returns:
        AUC score between 0 and 1
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    if len(y_true) == 0 or len(y_scores) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length")
    
    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain at least two classes")
    
    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception as e:
        raise ValueError(f"Error computing AUC: {str(e)}")
    
    return auc


def compute_eer(y_true, y_scores):
    """
    Compute Equal Error Rate (EER).
    
    The EER is the point where False Positive Rate equals False Negative Rate.
    
    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_scores: Predicted scores or probabilities
    
    Returns:
        EER value between 0 and 1 (lower is better)
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    if len(y_true) == 0 or len(y_scores) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length")
    
    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain at least two classes")
    
    try:
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Compute False Negative Rate
        fnr = 1 - tpr
        
        # Find the threshold where FPR and FNR are closest
        eer_threshold_idx = np.argmin(np.abs(fpr - fnr))
        
        # EER is the average of FPR and FNR at that point
        eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
        
    except Exception as e:
        raise ValueError(f"Error computing EER: {str(e)}")
    
    return eer


def compute_ap(y_true, y_scores):
    """
    Compute Average Precision (AP).
    
    AP summarizes the precision-recall curve as the weighted mean of precisions
    achieved at each threshold.
    
    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_scores: Predicted scores or probabilities
    
    Returns:
        Average Precision score between 0 and 1
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    if len(y_true) == 0 or len(y_scores) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length")
    
    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain at least two classes")
    
    try:
        ap = average_precision_score(y_true, y_scores)
    except Exception as e:
        raise ValueError(f"Error computing AP: {str(e)}")
    
    return ap
