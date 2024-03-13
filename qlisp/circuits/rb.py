import random

import numpy as np
from cycles.clifford import cliffordOrder

from ..clifford.utils import twoQubitCliffordSequence
from ..simple import seq2mat
from ._rb.clifford import inv, mat2index, mul
from .utils import mapping_qubits

_index2seq = [twoQubitCliffordSequence(i) for i in range(cliffordOrder(2))]


def twoQubitGate(gates):
    return {
        ('CZ', 'CZ'): ('CZ', (0, 1)),
        ('C', 'Z'): ('CZ', (0, 1)),
        ('Z', 'C'): ('CZ', (0, 1)),
        ('CX', 'CX'): ('Cnot', (0, 1)),
        ('XC', 'XC'): ('Cnot', (1, 0)),
        ('CR', 'CR'): ('CR', (0, 1)),
        ('RC', 'RC'): ('CR', (1, 0)),
        ('C', 'X'): ('Cnot', (0, 1)),
        ('X', 'C'): ('Cnot', (1, 0)),
        ('C', 'R'): ('CR', (0, 1)),
        ('R', 'C'): ('CR', (1, 0)),
        ('iSWAP', 'iSWAP'): ('iSWAP', (0, 1)),
        ('SWAP', 'SWAP'): ('SWAP', (0, 1)),
        ('SQiSWAP', 'SQiSWAP'): ('SQiSWAP', (0, 1)),
    }[gates]


def seq2qlisp(seq, qubits):
    if len(seq) > 2:
        raise ValueError("Only support 1 or 2 bits.")
    if len(seq) != len(qubits):
        raise ValueError("seq size and qubit num mismatched.")

    qlisp = []
    for gates in zip(*seq):
        try:
            qlisp.append(twoQubitGate(gates))
        except:
            for gate, i in zip(gates, qubits):
                qlisp.append((gate, i))
    return qlisp


def circuit_to_index(circuit: list) -> int:
    if not circuit:
        return 0
    mat = seq2mat(circuit)
    if mat.shape[0] == 2:
        mat = np.kron(np.eye(2), mat)
    return mat2index(mat)


def index_to_circuit(index: int, qubits=(0, ), base=None, rng=None) -> list:
    if len(qubits) > 2:
        raise ValueError('Only support 1 or 2 qubits')
    if rng is None:
        rng = random.Random()
    if base is None:
        base = _index2seq
    seq = rng.choice(base[index])
    if len(qubits) == 1:
        seq = (seq[1], )
    return seq2qlisp(seq, range(len(qubits)))


def generateRBCircuit(qubits, cycle, seed=None, interleaves=[], base=None):
    """Generate a random Clifford RB circuit.

    Args:
        qubits (list): The qubits to use.
        cycle (int): The cycles of clifford sequence.
        seed (int): The seed for the random number generator.
        interleaves (list): The interleaves to use.
        base (list): The basic two-qubit Clifford sequence.

    Returns:
        list: The RB circuit.
    """
    if isinstance(qubits, (str, int)):
        qubits = {0: qubits}
    else:
        qubits = {i: q for i, q in enumerate(qubits)}

    MAX = cliffordOrder(len(qubits))

    interleaves_index = circuit_to_index(interleaves)

    ret = []
    index = 0
    rng = random.Random(seed)

    for _ in range(cycle):
        i = rng.randrange(MAX)
        index = mul(i, index)
        ret.extend(index_to_circuit(i, qubits, base, rng))
        index = mul(interleaves_index, index)
        ret.extend(interleaves)

    ret.extend(index_to_circuit(inv(index), qubits, base, rng))

    return mapping_qubits(ret, qubits)
