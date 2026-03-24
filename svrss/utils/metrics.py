"""Class to computes metrics for Carrada"""
import numpy as np
from scipy.stats import hmean
# from sklearn.metrics import confusion_matrix
import torch
from svrss.utils.distributed_utils import reduce_value

# from https://stackoverflow.com/questions/59080843/faster-method-of-computing-confusion-matrix
def confusion_matrix(y_true, y_pred, num_class):
    N = num_class
    y = N * y_true + y_pred
    y = torch.bincount(y)
    if len(y) < N * N:
        y = torch.cat((y, torch.zeros(N * N - len(y), dtype=y.dtype, device=y.device)))
    y = y.reshape(N, N)
    return y

class Evaluator:
    """Class to evaluate a model with quantitative metrics
    using a ground truth mask and a predicted mask.

    PARAMETERS
    ----------
    num_class: int
    """

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def get_pixel_prec_class(self, harmonic_mean=False):
        """Pixel Precision"""
        diag = np.diag(self.confusion_matrix)
        denom = np.nansum(self.confusion_matrix, axis=0)
        prec_by_class = np.divide(diag, denom, out=np.zeros_like(diag, dtype=float), where=denom != 0)
        if harmonic_mean:
            prec = hmean(prec_by_class)
        else:
            prec = np.mean(prec_by_class)
        return prec, prec_by_class

    def get_pixel_recall_class(self, harmonic_mean=False):
        """Pixel Recall"""
        diag = np.diag(self.confusion_matrix)
        denom = np.nansum(self.confusion_matrix, axis=1)
        recall_by_class = np.divide(diag, denom, out=np.zeros_like(diag, dtype=float), where=denom != 0)
        if harmonic_mean:
            recall = hmean(recall_by_class)
        else:
            recall = np.mean(recall_by_class)
        return recall, recall_by_class

    def get_pixel_acc_class(self, harmonic_mean=False):
        """Pixel Accuracy"""
        diag = np.diag(self.confusion_matrix)
        row_sum = np.nansum(self.confusion_matrix, axis=1)
        # Per-class pixel accuracy is TP / GT pixels for each class.
        acc_by_class = np.divide(diag, row_sum, out=np.zeros_like(diag, dtype=float), where=row_sum != 0)
        total = np.nansum(self.confusion_matrix)
        global_acc = float(diag.sum() / total) if total != 0 else 0.0
        if harmonic_mean:
            acc = hmean(acc_by_class)
        else:
            acc = global_acc
        return acc, acc_by_class

    def get_miou_class(self, harmonic_mean=False):
        """Mean Intersection over Union"""
        diag = np.diag(self.confusion_matrix)
        denom = (np.nansum(self.confusion_matrix, axis=1)
                 + np.nansum(self.confusion_matrix, axis=0)
                 - diag)
        miou_by_class = np.divide(diag, denom, out=np.zeros_like(diag, dtype=float), where=denom != 0)
        if harmonic_mean:
            miou = hmean(miou_by_class)
        else:
            miou = np.mean(miou_by_class)
        return miou, miou_by_class

    def get_dice_class(self, harmonic_mean=False):
        """Dice"""
        _, prec_by_class = self.get_pixel_prec_class()
        _, recall_by_class = self.get_pixel_recall_class()
        # Add epsilon term to avoid /0
        dice_by_class = 2*prec_by_class*recall_by_class/(prec_by_class + recall_by_class + 1e-8)
        if harmonic_mean:
            dice = hmean(dice_by_class)
        else:
            dice = np.mean(dice_by_class)
        return dice, dice_by_class
    
    # def _generate_matrix(self, labels, predictions):
    #     matrix = confusion_matrix(labels.flatten(), predictions.flatten(),
    #                               labels=list(range(self.num_class)))
    #     return matrix
    
    def _generate_matrix(self, labels, predictions):
        matrix = confusion_matrix(labels.flatten(), predictions.flatten(),
                                  num_class=self.num_class)
        return matrix.cpu().numpy()

    def add_batch(self, labels, predictions):
        """Method to add ground truth and predicted masks by batch
        and update the global confusion matrix (entire dataset)

        PARAMETERS
        ----------
        labels: torch tensor or numpy array
            Ground truth masks
        predictions: torch tensor or numpy array
            Predicted masks
        """
        assert labels.shape == predictions.shape
        self.confusion_matrix += self._generate_matrix(labels, predictions)

    def reset(self):
        """Method to reset the confusion matrix"""
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def update(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix.numpy()
    
    def get(self):
        return torch.from_numpy(self.confusion_matrix)
    
    # def add_batch_dist(self, labels, predictions):
    #     assert labels.shape == predictions.shape
    #     conf_matrix = torch.from_numpy(self._generate_matrix(labels.cpu(), predictions.cpu())).to('cuda')
    #     conf_matrix = reduce_value(conf_matrix, average=False)
    #     self.confusion_matrix += conf_matrix.cpu().numpy()
    def add_batch_dist(self, labels, predictions):
        assert labels.shape == predictions.shape
        conf_matrix = confusion_matrix(labels.flatten(), predictions.flatten(),
                                  num_class=self.num_class)
        conf_matrix = reduce_value(conf_matrix, average=False)
        self.confusion_matrix += conf_matrix.cpu().numpy()
