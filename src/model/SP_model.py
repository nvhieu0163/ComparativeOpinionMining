from abc import ABC


from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from model.model import Model


class SentimentPolarizationModel(Model, ABC):

    def get_evaluate(self, y_test, y_predicts, pos_label):
        p = precision_score(y_test, y_predicts, labels=[pos_label], average = 'macro')
        r = recall_score(y_test, y_predicts, labels=[pos_label], average = 'macro')
        f1 = f1_score(y_test, y_predicts, labels=[pos_label], average = 'macro')
        return p, r, f1
