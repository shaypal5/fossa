import numpy as np
import pandas as pd

from fossa.base import FossaAnomalyDetectorABC


class MockModel(FossaAnomalyDetectorABC):

    def __init__(self):
        self.prediction = 1

    def __init__(self, prediction=1):
        self.prediction = prediction

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self.fit(X=X, y=None)

    def predict(self, X):
        res_df = pd.DataFrame(index=X.index)
        # res_df.sort_index(level=['date', 'category'], ascending=[1, 0], inplace=True)
        res_df['direction'] = np.full(len(X), self.prediction)
        return res_df[['direction']]
