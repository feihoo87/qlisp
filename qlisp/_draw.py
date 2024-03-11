import numpy as np

from .simple import gate2mat, gate_name


def to_old(qlisp):
    ret = []
    for gate, *qubits in qlisp:
        if len(qubits) == 1 and isinstance(qubits[0], tuple):
            qubits = qubits[0]
        ret.append((gate, tuple(qubits)))
    return ret


def to_new(qlisp):
    ret = []
    for gate, *qubits in qlisp:
        if len(qubits) == 1 and isinstance(qubits[0], tuple):
            qubits = qubits[0]
        ret.append((gate, *tuple(qubits)))
    return ret


def draw(qlisp):
    from qiskit import QuantumCircuit
    qlisp = to_new(qlisp)

    all_qubits = set()
    all_cbits = set()

    def key(x):
        if isinstance(x, int):
            return ('Q', x)
        else:
            try:
                return (x[0], int(x[1:]))
            except:
                return (x, 0)

    for gate, *qubits in qlisp:
        all_qubits.update(qubits)
        if gate_name(gate) == 'Measure':
            cbit = gate[1]
            all_cbits.add(cbit)
    all_qubits = sorted(all_qubits, key=key)
    all_cbits = sorted(all_cbits)

    qubit_map = {q: i for i, q in enumerate(all_qubits)}
    cbit_map = {c: i for i, c in enumerate(all_cbits)}

    circuit = QuantumCircuit(len(all_qubits), len(all_cbits))

    for gate, *qubits in qlisp:
        if gate_name(gate) == 'Measure':
            cbit = gate[1]
            circuit.measure([qubit_map[q] for q in qubits], [cbit_map[cbit]])
        elif gate_name(gate) == 'Barrier':
            circuit.barrier([qubit_map[q] for q in qubits])
        elif gate_name(gate) in [
                'CX', 'Cnot', 'CZ', 'CY', 'CH', 'iSWAP', 'SWAP'
        ]:
            {
                'CX': circuit.cx,
                'Cnot': circuit.cx,
                'CZ': circuit.cz,
                'CY': circuit.cy,
                'CH': circuit.ch,
                'iSWAP': circuit.iswap,
                'SWAP': circuit.swap,
            }[gate_name(gate)](*[qubit_map[q] for q in qubits])
        elif gate_name(gate) in [
                'I', 'X', 'Y', 'Z', 'H', 'T', 'S', '-T', '-S', 'Reset'
        ]:
            {
                'I': circuit.id,
                'X': circuit.x,
                'Y': circuit.y,
                'Z': circuit.z,
                'H': circuit.h,
                'T': circuit.t,
                'S': circuit.s,
                '-T': circuit.tdg,
                '-S': circuit.sdg,
                'Reset': circuit.reset
            }[gate_name(gate)](qubit_map[qubits[0]])
        else:
            try:
                mat, n = gate2mat(gate)
            except:
                mat = np.eye(2**len(qubits))
            if isinstance(gate, tuple):
                label = f'{gate_name(gate)}{tuple(gate[1:])}'
            else:
                label = f'{gate_name(gate)}'
            circuit.unitary(mat, [qubit_map[q] for q in qubits], label=label)
    return circuit.draw(output='mpl')
