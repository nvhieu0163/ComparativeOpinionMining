from pyvi import ViTokenizer, ViPosTagger
from classes.task_class import InputObject

#This is input class define for task IBO
class IBOInputObject(InputObject):
    def __init__(self, stc_idx, stc, subject: str, object:str):
        super().__init__(stc_idx, stc)
        self.subject = subject
        self.object = object


#This is output class define for task IBO
class IBOOutputObject():
    def __init__(self, stc_idx, pos, tag) -> None:
        self.stc_idx = stc_idx
        self.pos = pos
        self.tag = tag


#This is class for pyvi module
class Pyvi:
    def __init__(self):
        pass





