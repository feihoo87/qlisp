from typing import Callable

import numpy as np


def regesterGateMatrix(gate: str,
                       mat: Callable | np.ndarray,
                       N: int | None = ...,
                       docs: str = ...):
    ...


def applySeq(seq: list[tuple], psi0=None, ignores:list[str]=...) -> np.ndarray:
    ...


def seq2mat(seq: list[tuple], U=None, ignores:list[str]=...) -> np.ndarray:
    ...
