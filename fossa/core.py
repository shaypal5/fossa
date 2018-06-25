"""scikit-learn classifier wrapper for fasttext."""

import abc

from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, ClassifierMixin


class FossaPredictorABC(BaseEstimator, ClassifierMixin, metaclass=abc.ABCMeta):
    """An abstact base class for sklearn-compliant time-series predictors."""

    @staticmethod
    def _validate_x(X):
        try:
            if len(X.index.levels) != 2:
                raise ValueError(
                    "PreviousWindowAnomalyPredictor requires a DataFrame "
                    "with a two-leveled multi-index, where the first indexes "
                    "time windows and the second level indexes class/topic "
                    "frequency per-window."
                )
            if len(X.columns) != 1:
                raise ValueError(
                    "PreviousWindowAnomalyPredictor requires a DataFrame with "
                    "a single column.")
            col_lbl = X.columns[0]
            if not is_numeric_dtype(X[col_lbl]):
                raise ValueError(
                    "PreviousWindowAnomalyPredictor requires a DataFrame with "
                    "a single column of a numeric dtype.")
            return X
        except AttributeError:
            raise ValueError("PreviousWindowAnomalyPredictor requires "
                             "pandas.DataFrame objects with multi-level index "
                             "as input.")
