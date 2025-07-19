"""
Metrics.
"""

import inspect
from replay import metrics as base_class
from replay.metrics import OfflineMetrics

import warnings
warnings.filterwarnings("ignore", message=".*contains queries that are not presented in recommendations.*")


DEFAULT_METRICS = ['NDCG', 'HitRate', 'Recall', 'MRR', 'Coverage']


class Evaluator:
    """Class for computing recommendation metrics.
    """

    def __init__(self, metrics=DEFAULT_METRICS, top_k=[10, 100],
                 modes=['Mean'], user_col='user_id',
                 item_col='item_id', prediction_col='prediction'):
        """Args:
            metrics (list): List with metrics name. The names are taken from the Replay library.
            topk (list): Consider the highest k scores in the ranking. Defaults to [10, 100].
            modes (list): Classes for calculating aggregation metrics. Defaults to Mean.
                Available modes: Median, ConfidenceInterval, PerUser.
            user_col (str): Defaults to 'user_id'.
            item_col (str): Defaults to 'item_id'.
            prediction_col (str): Defaults to 'prediction'."""

        self.metrics = metrics
        self.top_k = top_k
        self.modes = modes
        self.user_col = user_col
        self.item_col = item_col
        self.prediction_col = prediction_col

        class_method = [x[0] for x in inspect.getmembers(base_class)[:19]]

        if not isinstance(metrics, list):
            raise ValueError("Use the list data type for metrics.")

        if not isinstance(top_k, list):
            raise ValueError("Use the list data type for top_k.")

        if not isinstance(modes, list):
            raise ValueError("Use the list data type for modes.")

        if len(modes) > 1 and 'PerUser' in modes:
            raise ValueError("Mode 'PerUser' can use only alone.")

        for mode in modes:
            if mode not in class_method:
                raise ValueError(f"{mode} is not available in Replay. Look at the documentation. https://sb-ai-lab.github.io/RePlay/pages/modules/metrics.html#replay.metrics")

        for metric in metrics:
            if metric not in class_method:
                raise ValueError(f"{metric} is not available in Replay. Look at the documentation. https://sb-ai-lab.github.io/RePlay/pages/modules/metrics.html#replay.metrics")

    def compute_metrics(self, test, recs, train=None):
        """Compute all metrics.

        Args:
            test (pd.DataFrame): Dataframe with test data.
            recs (pd.DataFrame): Dataframe with recommendations.
            train (pd.DataFrame): Dataframe with train data.

        Returns:
            metrics
        """

        metrics_list = []
        for metric in self.metrics:
            for k in self.top_k:
                if metric == 'Coverage':
                    metrics_list.append(getattr(base_class, metric)(topk=k))
                else:
                    for mode in self.modes:
                        mode = getattr(base_class, mode)()
                        metrics_list.append(getattr(base_class, metric)(topk=k, mode=mode))

        metrics = OfflineMetrics(
            metrics_list, query_column=self.user_col, 
            item_column=self.item_col, rating_column=self.prediction_col)(recs, test, train)

        return metrics