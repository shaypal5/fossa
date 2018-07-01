import pandas as pd
import numpy as np

from fossa.core import FossaPredictorABC


class MockModel(FossaPredictorABC):

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self.fit(X=X, y=None)

    def predict(self, X):
        res_df = pd.DataFrame(index=X.index)
        #res_df.sort_index(level=['date', 'category'], ascending=[1, 0], inplace=True)
        res_df['direction'] = np.ones(len(X))
        return res_df[['direction']]
