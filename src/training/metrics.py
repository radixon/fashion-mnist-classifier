import torch
from sklearn.metrics import f1_score
from typing import List, Union


def calculate_f1_score(y_true: Union[torch.Tensor, List[int]],
                       y_pred: Union[torch.Tensor, List[int]],
                       average: str = 'weighted') -> float:
    """
    Calculate F1-score

    Args:
        y_true: True Labels
        y_pred: Predicted Labels
        average (str): Type of averaging for F1-score

    Returns:
        float: The calculated F1-score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    return f1_score(y_true, y_pred, average=average)
