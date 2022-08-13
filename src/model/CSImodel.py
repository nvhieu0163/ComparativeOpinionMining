import datetime
import pickle
from abc import ABC
from typing import List


import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from model.model import Model


class ComparativeSentenceModel(Model, ABC):
    pass