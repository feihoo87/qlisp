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

    parametric_gates = {
        'Init': circuit.prepare_state,
        'Rx': circuit.rx,
        'Ry': circuit.ry,
        'Rz': circuit.rz,
        'P': circuit.p,
        'U': circuit.u,
        'u1': circuit.p,
        'u2': lambda phi, lam: circuit.u(np.pi / 2, phi, lam),
        'u3': circuit.u,
        'rfUnirary': circuit.r,
        'R': lambda phi: circuit.r(np.pi / 2, phi),
        'CP': circuit.cp,
        'CRx': circuit.crx,
        'CRy': circuit.cry,
        'CRz': circuit.crz,
        'CU': circuit.cu
    }

    simple_gates = {
        'I': circuit.id,
        'X': circuit.x,
        'Y': circuit.y,
        'Z': circuit.z,
        'H': circuit.h,
        'T': circuit.t,
        'S': circuit.s,
        'X/2': circuit.sx,
        'Z/2': circuit.s,
        '-T': circuit.tdg,
        '-S': circuit.sdg,
        '-Z/2': circuit.sdg,
        '-X/2': circuit.sxdg,
        'Reset': circuit.reset,
        'CX': circuit.cx,
        'Cnot': circuit.cx,
        'CZ': circuit.cz,
        'CY': circuit.cy,
        'CH': circuit.ch,
        'iSWAP': circuit.iswap,
        'SWAP': circuit.swap,
        'CCX': circuit.ccx,
        'CCZ': circuit.ccz,
        'CSWAP': circuit.cswap,
    }

    for gate, *qubits in qlisp:
        if gate_name(gate) == 'Measure':
            cbit = gate[1]
            circuit.measure([qubit_map[q] for q in qubits], [cbit_map[cbit]])
        elif gate_name(gate) == 'Delay':
            duration = gate[1]
            if np.abs(duration) < 1e-9:
                duration *= 1e12
                unit = 'ps'
            elif np.abs(duration) < 1e-6:
                duration *= 1e9
                unit = 'ns'
            elif np.abs(duration) < 1e-3:
                duration *= 1e6
                unit = 'us'
            elif np.abs(duration) < 1:
                duration *= 1e3
                unit = 'ms'
            else:
                unit = 's'
            if duration >= 0:
                circuit.delay(duration, qubit_map[qubits[0]], unit)
            else:
                label = f'Delay({duration:.1f} [{unit}])'
                circuit.unitary(np.eye(2), [qubit_map[q] for q in qubits],
                                label=label)
        elif gate_name(gate) == 'Barrier':
            circuit.barrier([qubit_map[q] for q in qubits])
        elif gate_name(gate) in simple_gates:
            simple_gates[gate_name(gate)](*[qubit_map[q] for q in qubits])
        elif gate_name(gate) in parametric_gates:
            parametric_gates[gate_name(gate)](*gate[1:],
                                              *[qubit_map[q] for q in qubits])
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
