from typing import Any

# THIS FILE INCLUDE CLASSES FOR INPUT AND OUTPUT OF TASKS

# This is input class define for task CSI, AC, SP 
class InputObject:
    def __init__(self, stc_idx: int, stc: str):
        self.stc_idx = stc_idx
        self.stc = stc


# This is output class define for task CSI, AC, SP 
class OutputObject:
    def __init__(self, score: int):
        self.score = score

