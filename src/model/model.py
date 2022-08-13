import pickle
import datetime

from abc import ABC
from typing import List

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class Model:

    def __init__(self, task_name, vocab_path, base_model) -> None:
        self.task_name = task_name
        self.vocab_path = vocab_path
        self.vocab = []
        self.model = base_model
        self.k_best = 0
        self.thresh_hold = 0

    def set_vocab_chi2(self, k_best):
        vocab_df = pd.read_csv(self.vocab_path).head(k_best)
        self.vocab = list(vocab_df.word)
        self.k_best = k_best


    def represent_feature_chi2(self, list_input_objects):
        features = []
        for obj in list_input_objects:
            _feature = [1 if word in obj.stc.split(' ') else 0 for word in self.vocab]
            features.append(_feature)
        
        return features


    def train(self, inputs, outputs):
        features = inputs
        label = outputs
        self.model.fit(features, label)

    
    def save(self, label):
        pickle.dump(self.model,
            open('./trained_save_model/{a}_model/model_{b}_{c}_{d}.pkl'.format(
                a = self.task_name,
                b = self.model,
                c = label,
                d = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), 'wb'))

    
    def load(self, path):
        model = pickle.load(open(path, 'rb'))
        self.model = model


    def predict(self, encode_inputs: List):
        features = encode_inputs
        
        if self.thresh_hold != 0:
            predict_prob = self.model.predict_proba(features)[:, 1]  #predict_proba : probability for each label
            predict = np.where(predict_prob > self.thresh_hold, 1, 0)
        else:
            predict = self.model.predict(features)
    
        return predict


    def get_evaluate(self, y_test, y_predicted):
        p = precision_score(y_true = y_test, y_pred = y_predicted)
        r = recall_score(y_true = y_test, y_pred = y_predicted)
        f1 = f1_score(y_true = y_test, y_pred = y_predicted)

        return p, r, f1


    def set_optimal_threshold(self, encode_inputs: List, y_test):
        
        print("* Tuning threshold {} model ...".format(self.task_name))
        X = encode_inputs
        y_predict_proba = self.model.predict_proba(X)[:, 1]
        
        thresholds = np.arange(0, 1, 0.001)
        max_f1 = 0
        opt_t = 0

        for t in thresholds:
            predict_t = np.where(y_predict_proba > t, 1, 0)
            f1 = f1_score(y_test, predict_t)
            if f1 > max_f1:
                max_f1 = f1
                opt_t = t
        
        self.threshold = opt_t
        print("  Tuning threshold {} model DONE!".format(self.task_name))