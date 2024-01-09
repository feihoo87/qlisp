from .matricies import (CR, CX, CZ, SWAP, H, S, Sdag, SQiSWAP, T, Tdag, U,
                        Unitary2Angles, fSim, iSWAP, make_immutable, rfUnitary,
                        sigmaI, sigmaM, sigmaP, sigmaX, sigmaY, sigmaZ,
                        synchronize_global_phase)
from .simple import applySeq, regesterGateMatrix, seq2mat

try:
    from qlispc import *
except ImportError:
    pass
