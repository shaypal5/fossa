import math

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def read_data(path):
    """Reads data to be fed into the model.

    Parameters
    ----------
    path : String
        path to the CSV fie containing the data to be modeled. The file should
        contain a date/time column, a category column and the value for this
        category and this date/time.

    Returns
    -------
    df : pandas.DataFrame
        Returns a pandas DataFrame with a two-leveled multi-index, the first
        indexing time and the second indexing class/topic frequency
        per-window, and a single column of a numeric dtype, giving said
        frequency.
     """
    df = pd.read_csv(path, index_col=None)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date', 'category'])
    print('Read {0} rows'.format(len(df)))
    return df


def eval_models(X, y, models, n_splits=None, verbose=False):
    """Evalutes one or more models with the provided dataset

    Parameters
    ----------
    X : pandas.DataFrame
        A pandas DataFrame with a two-leveled multi-index, the first indexing
        time and the second indexing class/topic frequency per-window, and a
        single column of a numeric dtype, giving said frequency.
    y : pandas.DataFrame
        A pandas DataFrame with a two-leveled multi-index, the first indexing
        time and the second indexing class/topic frequency per-window, and a
        single column of a integer type (-1,0,1), with the ground truth labels
        for each time and class/topic
    models : list of FossaPredictorABC based models.
        The models to evaluate.
    n_splits : integer
        The number of splits for TimeSeriesSplit.
        See  http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

    Returns
    -------
    res : dict
        A dictionary of metrics per model
    """

    if X is None:
        raise TypeError
    if y is None:
        raise TypeError

    if n_splits is None:
        n_splits = int(len(X.index.levels[0]) / 2)
        print("Running {} splits".format(n_splits))

    if n_splits > len(X.index.levels[0]):
        n_splits = int(len(X.index.levels[0]) / 2)
        print(("Warning: n_splits cannot be larger than the number of dates. "
              "Reducing n_splits to {}").format(n_splits))

    per_model_res = {}

    # X = X.reset_index().set_index('date') # split over dates
    # y = y.reset_index().set_index('date')  # split over dates
    datetimeindex = X.index.levels[0]
    categories = X.index.levels[1]
    for model in models:
        counter = 0
        metrics = {}
        for category in categories:
            metrics[category] = {
                'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0,
                'num_samples': 0, 'num_values': 0,
            }

        tscv = TimeSeriesSplit(n_splits=n_splits)
        if verbose:
            print("Model: {}".format(str(model)))
        for train, test in tscv.split(datetimeindex):
            if verbose:
                print("Iteration: %s, Train size: %s, Test size: %s. " % (
                    counter, len(train), len(test)))

            train_samples = X.loc[datetimeindex[train]]
            train_samples.index = train_samples.index.remove_unused_levels()

            train_labels = y.loc[datetimeindex[train]]
            train_labels.index = train_labels.index.remove_unused_levels()

            test_samples = X.loc[datetimeindex[test]]
            test_samples.index = test_samples.index.remove_unused_levels()

            test_labels = y.loc[datetimeindex[test]]
            test_labels.index = test_labels.index.remove_unused_levels()

            model.fit(train_samples, train_labels)
            prediction = model.predict(test_samples)

            results = pd.merge(
                prediction, test_labels, left_index=True, right_index=True)
            results = pd.merge(
                results, test_samples, left_index=True, right_index=True)
            results.sort_index(
                level=['date', 'category'], ascending=True, inplace=True)
            # print(results)
            for category in results.index.levels[1]:
                category_results = results.loc[pd.IndexSlice[:, category], :]
                category_results.index = category_results.index.remove_unused_levels()
                if verbose:
                    print(category)
                    print(category_results)
                metrics[category]['TP'] = metrics[category]['TP'] + np.sum(
                    (category_results['trend'].values != 0) & (
                            category_results['trend'].values == category_results['is_anomaly'].values))
                metrics[category]['FP'] = metrics[category]['FP'] + np.sum(
                    (category_results['trend'].values != 0) & (
                        category_results['is_anomaly'].values == 0))
                metrics[category]['TN'] += np.sum(
                    (category_results['trend'].values == 0) & (
                        category_results['is_anomaly'].values == 0))
                metrics[category]['FN'] += np.sum(
                    (category_results['trend'] == 0).values & (
                        category_results['is_anomaly'].values != 0))
                metrics[category]['num_samples'] += len(category_results)
                metrics[category]['num_values'] += np.sum(
                    category_results['value'])

            counter += 1
            if verbose:
                print(metrics)
        final_metrics = get_final_metrics(metrics)
        per_model_res[str(model)] = final_metrics
    return per_model_res


def get_final_metrics(metrics):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for category in metrics:
        TP += np.sum(metrics[category]['TP'])
        FP += np.sum(metrics[category]['FP'])
        TN += np.sum(metrics[category]['TN'])
        FN += np.sum(metrics[category]['FN'])

    final_metrics = dict()
    if (TP+FP) > 0:
        final_metrics['precision'] = TP / (TP + FP)
    else:
        final_metrics['precision'] = math.nan
    if (TP+FN) > 0:
        final_metrics['recall'] = TP / (TP + FN)
    else:
        final_metrics['recall'] = math.nan
    final_metrics['f1'] = f_beta(
        final_metrics['precision'], final_metrics['recall'], 1)
    final_metrics['f0.5'] = f_beta(
        final_metrics['precision'], final_metrics['recall'], 0.5)
    final_metrics['raw'] = metrics
    return final_metrics


def f_beta(precision, recall, beta):
    if math.isnan(precision) or math.isnan(recall):
        return math.nan
    return ((1 + beta ** 2) * precision * recall) / (
        ((beta ** 2) * precision) + recall)
