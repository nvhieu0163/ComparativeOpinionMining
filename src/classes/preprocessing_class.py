from pyvi import ViTokenizer, ViPosTagger
from classes.task_class import InputObject

#This is input class define for task IBO
class IBO_InputObject(InputObject):
    def __init__(self, stc_idx, stc, subject: str, object:str):
        super().__init__(stc_idx, stc)
        self.subject = subject
        self.object = object


#This is output class define for task IBO
class IBO_OutputObject():
    def __init__(self, stc_idx, pos, tag):
        self.stc_idx = stc_idx
        self.pos = pos
        self.tag = tag


class NER_SentenceGetter(object):    
    def __init__(self, dataframe):
        self.n_sent = 1
        self.data = dataframe
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(), 
                                                           s['POS'].values.tolist(), 
                                                           s['Tag'].values.tolist())]
        self.grouped = self.data.groupby('sentence#').apply(agg_func)
        self.sentences = [s for s in self.grouped]


#This is class for pyvi module
class Pyvi:
    def __init__(self):
        pass





