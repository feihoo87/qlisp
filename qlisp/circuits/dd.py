from itertools import repeat

import numpy as np


def DD(qubit, t, gates, pos):
    seq = []
    i = 0
    for gate in gates:
        gap = t * (pos[i] - pos[i - 1]) if i > 0 else t * pos[0]
        seq.append((('Delay', gap), qubit))
        seq.append((gate, qubit))
        i += 1
    gap = t * (1 - pos[-1]) if len(pos) > 0 else t
    seq.append((('Delay', gap), qubit))
    return seq


def XY4(qubit, t):
    pos = np.arange(1, 5) / 5
    return DD(qubit, t, ['X', 'Y', 'X', 'Y'], pos)


def XY8(qubit, t):
    pos = np.arange(1, 9) / 9
    return DD(qubit, t, ['X', 'Y', 'X', 'Y', 'Y', 'X', 'Y', 'X'], pos)


def XY16(qubit, t):
    pos = np.arange(1, 17) / 17
    return DD(qubit, t, [
        'X', 'Y', 'X', 'Y', 'Y', 'X', 'Y', 'X', 'X', 'Y', 'X', 'Y', 'Y', 'X',
        'Y', 'X'
    ], pos)


def UDD(qubit, n, t):
    j = np.arange(n) + 1
    return DD(qubit, t, repeat('Y', times=n),
              np.sin(np.pi * j / (2 * n + 2))**2)


def CPMG(qubit, n, t):
    j = np.arange(n) + 1
    return DD(qubit, t, repeat('Y', times=n), (j - 0.5) / n)


def CP(qubit, n, t):
    j = np.arange(n) + 1
    return DD(qubit, t, repeat('X', times=n), (j - 0.5) / n)


def Ramsey(qubit, t, f=0):
    return [('X/2', qubit), (('Delay', t), qubit),
            (('rfUnitary', np.pi / 2, 2 * np.pi * f * t), qubit)]


def SpinEcho(qubit, t, f=0):
    return [('X/2', qubit), (('Delay', t / 2), qubit),
            (('rfUnitary', np.pi, np.pi * f * t), qubit),
            (('Delay', t / 2), qubit), ('X/2', qubit)]
