from typing import Any

# This is classes define for task CSI, AC, SP 

class InputObject:
    def __init__(self, stc_idx: int, stc: str) -> None:
        self.stc_idx = stc_idx
        self.stc = stc


class OutputObject:
    def __init__(self, score: int) -> None:
        self.score = score

