import numpy as np

from qlisp import applySeq


def test_PorterThomas_distribution():

    def randomSeq(depth, N):
        seq = []
        for i in range(depth):
            for j in range(N):
                seq.append((str(np.random.choice(['X/2', 'Y/2', 'W/2'])), j))
            for j in range(i % 2, N, 2):
                seq.append(('SQiSWAP', j, (j + 1) % N))
        return seq

    p = []
    # run 1000 random circuit on 6 qubits
    for i in range(1000):
        seq = randomSeq(10, 6)
        psi = applySeq(seq)
        p.extend(list(np.abs(psi)**2))
    p = np.asarray(p)

    # check distribution of probabilities
    N = 2**6
    y, x = np.histogram(N * p, bins=50, density=True)
    x = (x[:-1] + x[1:]) / 2

    assert np.sum((np.exp(-x) - y)**2) < 1e-3
