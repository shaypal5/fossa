"""scikit-learn classifier wrapper for fasttext."""

from abc import ABC

from scipy.stats import power_divergence
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, ClassifierMixin


class FossaAnomalyDetectorABC(BaseEstimator, ClassifierMixin, ABC):
    """An abstact base class for sklearn-y time-series anomaly detectors."""

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


class PowerDivergenceAnomalyDetectorABC(FossaAnomalyDetectorABC, ABC):
    """An abstact class for power-divergence time-series anomaly detectors.

    Parameters
    ----------
    power : float or str, optional
        `power` gives the power in the Cressie-Read power divergence
        statistic.  The default is 1.  For convenience, `power` may be
        assigned one of the following strings, in which case the
        corresponding numerical value is used::
            String              Value   Description
            "pearson"             1     Pearson's chi-squared statistic.
                                        In this case, the function is
                                        equivalent to `stats.chisquare`.
            "log-likelihood"      0     Log-likelihood ratio. Also known as
                                        the G-test.
            "freeman-tukey"      -1/2   Freeman-Tukey statistic.
            "mod-log-likelihood" -1     Modified log-likelihood ratio.
            "neyman"             -2     Neyman's statistic.
            "cressie-read"        2/3   Recommended by Cressie & Read.
        This parameter is forwarded to the `lambda_` parameter of the
        `scipy.stats.power_divergence` method. See details there.
    ddof : int, optional
        "Delta degrees of freedom": adjustment to the degrees of freedom
        for the p-value.  The p-value is computed using a chi-squared
        distribution with ``k - 1 - ddof`` degrees of freedom, where `k`
        is the number of observed frequencies.  The default value of `ddof`
        is 0. This parameter is forwarded to the `ddof` parameter of the
        `scipy.stats.power_divergence` method. See details there.
    """

    _param_str = 'Parameters\n    ' + ('-' * len('Parameters')) + '\n'
    _loc = __doc__.find(_param_str)
    # +4 so not to take the first four spaces
    _param_subdoc = __doc__[_loc + len(_param_str) + 4:]

    def __init__(self, power=None, ddof=None):
        if power is None:
            power = 1
        if ddof is None:
            ddof = 0
        self.power = power
        self.ddof = ddof

    def _power_divergence_test(self, f_obs, f_exp):
        """Tests the null hypothesis that the observed categorical data has the
        expected frequencies, using Cressie-Read power divergence statistic.

        Parameters
        ----------
        f_obs : array_like
            Observed frequencies in each category.
        f_exp : array_like, optional
            Expected frequencies in each category. By default the categories
            are assumed to be equally likely.
        """
        return power_divergence(
            f_obs=f_obs,
            f_exp=f_exp,
            ddof=self.ddof,
            axis=0,
            lambda_=self.power,
        )
