import datetime
from typing import List

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder


def feature_select_chi2(stcs, labels, task_name, k_best='all'):
    """
    select top k features through chi-square test
    :param stcs: list of sentences
    :param labels: list of label you want to score chi2 for
    :param k_best:  k words with the best score
    :param label_name: name task that chi2 test for
    :return:
    """
    bin_cv = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, binary=True)
    le = LabelEncoder()
    X = bin_cv.fit_transform(stcs, np.nan)
    y = le.fit_transform(labels).reshape(-1, 1)

    skb = SelectKBest(chi2, k=k_best)
    skb.fit(X, y)

    feature_ids = skb.get_support(indices=True)
    feature_names = bin_cv.get_feature_names()
    result = {}
    vocab = []

    for new_fid, old_fid in enumerate(feature_ids):
        feature_name = feature_names[old_fid]
        vocab.append(feature_name)

    result['word'] = vocab
    result['_score'] = list(skb.scores_)
    result['_pvalue'] = list(skb.pvalues_)
    result_df = pd.DataFrame.from_dict(result)
    result_df = result_df.sort_values('_score', ascending=False).reset_index()
    result_df.to_csv('./dataset/output/chi2_score_dict/{}.csv'.format(task_name))
    # we only care about the final extracted feature vocabulary
    return result


def run_feature_selection_chi2(input_list: List, label_list: List, train_path: str, task_name: str):
    for label in label_list:
        col_label_name = label
        for input in input_list:
            TASK_NAME = task_name + "_" + input + "_" + label
            col_input_name = input

            train_df = pd.read_csv(train_path)
            feature_select_chi2(train_df[col_input_name], train_df[col_label_name], task_name = TASK_NAME)