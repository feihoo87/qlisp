import numpy as np
from numpy import pi

# one qubit clifford gates
# each gate is represented by a tuple (theta, phi, lam) where
# u3(theta, phi, lam) = Rz(phi) Ry(theta) Rz(lam)
# the angles are in units of pi/2
one_qubit_clifford_u3_angles = [
    # Paulis
    (0, 0, 0),             #0 :  I
    (2, 2, 0),             #1 :  X
    (2, 0, 0),             #2 :  Y
    (0, 1, 1),             #3 :  Z

    # 2 pi / 3 rotations
    (1, 3, 0),             #4 :
    (1, 3, 2),             #5 :
    (1, 1, 0),             #6 :
    (1, 1, 2),             #7 :
    (1, 0, 1),             #8 :
    (1, 0, 3),             #9 :
    (1, 2, 1),             #10:
    (1, 2, 3),             #11:

    # pi / 2 rotations
    (1, 3, 1),             #12:  X/2
    (1, 1, 3),             #13: -X/2
    (1, 0, 0),             #14:  Y/2
    (1, 2, 2),             #15: -Y/2
    (0, 0, 1),             #16:  Z/2
    (0, 0, 3),             #17: -Z/2

    # Hadamard-like
    (1, 2, 0),             #18:
    (1, 0, 2),             #19: Hadamard
    (1, 1, 1),             #20:
    (1, 3, 3),             #21:
    (2, 3, 0),             #22:
    (2, 1, 0)              #23:
] #yapf: disable

# one quibit clifford gates represented by the u3 gate
one_qubit_clifford_seq = [('u3', theta * pi / 2, phi * pi / 2, lam * pi / 2)
                          for theta, phi, lam in one_qubit_clifford_u3_angles]

# one qubit clifford gates represented by the H and S gates
one_qubit_clifford_seq2 = [
    # Paulis
    [],                              # I
    ['H', 'S', 'S', 'H'],            # X
    ['H', 'S', 'S', 'H', 'S', 'S'],  # Y
    ['S', 'S'],                      # Z

    # 2 pi / 3 rotations
    ['S', 'S', 'H', 'S', 'S', 'S'],
    ['H', 'S', 'S', 'S'],
    ['S', 'S', 'H', 'S'],
    ['H', 'S'],
    ['S', 'S', 'S', 'H'],
    ['S', 'H'],
    ['S', 'S', 'S', 'H', 'S', 'S'],
    ['S', 'H', 'S', 'S'],

    # pi / 2 rotations
    ['S', 'S', 'S', 'H', 'S', 'S', 'S'], #  X/2
    ['S', 'H', 'S'],                     # -X/2
    ['S', 'S', 'H'],                     #  Y/2
    ['H', 'S', 'S'],                     # -Y/2
    ['S'],                               #  Z/2
    ['S', 'S', 'S'],                     # -Z/2

    # Hadamard-like
    ['S', 'S', 'H', 'S', 'S'],
    ['H'],
    ['S', 'S', 'S', 'H', 'S'],
    ['S', 'H', 'S', 'S', 'S'],
    ['H', 'S', 'S', 'H', 'S'],
    ['H', 'S', 'S', 'H', 'S', 'S', 'S']
] #yapf: disable

# one qubit clifford gates represented by the pi/2 pulse
one_qubit_clifford_seq3 = [
    # Paulis
    ('I', ),
    ('X', ),
    ('Y', ),
    ('Y', 'X'),

    # 2 pi / 3 rotations
    ('X/2', 'Y/2'),
    ('X/2', '-Y/2'),
    ('-X/2', 'Y/2'),
    ('-X/2', '-Y/2'),
    ('Y/2', 'X/2'),
    ('Y/2', '-X/2'),
    ('-Y/2', 'X/2'),
    ('-Y/2', '-X/2'),

    # pi / 2 rotations
    ('X/2', ),
    ('-X/2', ),
    ('Y/2', ),
    ('-Y/2', ),
    ('-X/2', 'Y/2', 'X/2'),
    ('-X/2', '-Y/2', 'X/2'),

    # Hadamard-like
    ('X', 'Y/2'),
    ('X', '-Y/2'), # Hadamard
    ('Y', 'X/2'),
    ('Y', '-X/2'),
    ('X/2', 'Y/2', 'X/2'),
    ('-X/2', 'Y/2', '-X/2')
] #yapf: disable

one_qubit_clifford_seq_inv = {
    g: i
    for i, g in enumerate(one_qubit_clifford_seq)
}
one_qubit_clifford_seq_inv['H'] = 19
one_qubit_clifford_seq_inv['S'] = 16
one_qubit_clifford_seq_inv['P'] = 16
one_qubit_clifford_seq_inv['Z/2'] = 16
one_qubit_clifford_seq_inv['I'] = 0
one_qubit_clifford_seq_inv['X'] = 1
one_qubit_clifford_seq_inv['Y'] = 2
one_qubit_clifford_seq_inv['Z'] = 3
one_qubit_clifford_seq_inv['X/2'] = 12
one_qubit_clifford_seq_inv['-X/2'] = 13
one_qubit_clifford_seq_inv['Y/2'] = 14
one_qubit_clifford_seq_inv['-Y/2'] = 15
one_qubit_clifford_seq_inv['-S'] = 17
one_qubit_clifford_seq_inv['-Z/2'] = 17
one_qubit_clifford_seq_inv.update({f'C1_{i}': i for i in range(24)})

# one qubit clifford multiplication table
# the table is defined by the following rules:
# the i-th one qubit clifford gate followed by the j-th one qubit clifford gate
# is equal to the k-th one qubit clifford gate
# where k = one_qubit_clifford_mul_table[i, j]
one_qubit_clifford_mul_table = np.array([
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
    [1,0,3,2,6,7,4,5,11,10,9,8,13,12,18,19,22,23,14,15,21,20,16,17],
    [2,3,0,1,7,6,5,4,10,11,8,9,20,21,15,14,23,22,19,18,12,13,17,16],
    [3,2,1,0,5,4,7,6,9,8,11,10,21,20,19,18,17,16,15,14,13,12,23,22],
    [4,7,5,6,11,8,9,10,2,3,1,0,22,17,21,12,14,18,13,20,23,16,15,19],
    [5,6,4,7,10,9,8,11,1,0,2,3,23,16,12,21,19,15,20,13,22,17,18,14],
    [6,5,7,4,8,11,10,9,3,2,0,1,16,23,20,13,18,14,12,21,17,22,19,15],
    [7,4,6,5,9,10,11,8,0,1,3,2,17,22,13,20,15,19,21,12,16,23,14,18],
    [8,9,11,10,1,3,2,0,7,4,5,6,19,14,22,16,20,12,23,17,15,18,13,21],
    [9,8,10,11,2,0,1,3,6,5,4,7,14,19,23,17,13,21,22,16,18,15,20,12],
    [10,11,9,8,3,1,0,2,4,7,6,5,18,15,17,23,12,20,16,22,14,19,21,13],
    [11,10,8,9,0,2,3,1,5,6,7,4,15,18,16,22,21,13,17,23,19,14,12,20],
    [12,13,21,20,18,19,14,15,22,17,23,16,1,0,4,5,8,10,6,7,2,3,11,9],
    [13,12,20,21,14,15,18,19,16,23,17,22,0,1,6,7,11,9,4,5,3,2,8,10],
    [14,19,15,18,22,16,23,17,20,21,12,13,8,9,2,0,6,4,1,3,10,11,7,5],
    [15,18,14,19,17,23,16,22,12,13,20,21,10,11,0,2,5,7,3,1,8,9,4,6],
    [16,23,22,17,12,21,20,13,19,14,15,18,5,6,8,11,3,0,10,9,7,4,1,2],
    [17,22,23,16,21,12,13,20,14,19,18,15,4,7,9,10,0,3,11,8,6,5,2,1],
    [18,15,19,14,16,22,17,23,21,20,13,12,11,10,3,1,4,6,0,2,9,8,5,7],
    [19,14,18,15,23,17,22,16,13,12,21,20,9,8,1,3,7,5,2,0,11,10,6,4],
    [20,21,13,12,19,18,15,14,17,22,16,23,3,2,7,6,10,8,5,4,0,1,9,11],
    [21,20,12,13,15,14,19,18,23,16,22,17,2,3,5,4,9,11,7,6,1,0,10,8],
    [22,17,16,23,13,20,21,12,15,18,19,14,7,4,11,8,2,1,9,10,5,6,0,3],
    [23,16,17,22,20,13,12,21,18,15,14,19,6,5,10,9,1,2,8,11,4,7,3,0],
], dtype=np.int8) #yapf: disable


_one_qubit_clifford_index = {
    (0, 0, 0): 0,
    (0, 0, 1): 16,
    (0, 0, 2): 3,
    (0, 0, 3): 17,
    (0, 1, 0): 16,
    (0, 1, 1): 3,
    (0, 1, 2): 17,
    (0, 1, 3): 0,
    (0, 2, 0): 3,
    (0, 2, 1): 17,
    (0, 2, 2): 0,
    (0, 2, 3): 16,
    (0, 3, 0): 17,
    (0, 3, 1): 0,
    (0, 3, 2): 16,
    (0, 3, 3): 3,
    (1, 0, 0): 14,
    (1, 0, 1): 8,
    (1, 0, 2): 19,
    (1, 0, 3): 9,
    (1, 1, 0): 6,
    (1, 1, 1): 20,
    (1, 1, 2): 7,
    (1, 1, 3): 13,
    (1, 2, 0): 18,
    (1, 2, 1): 10,
    (1, 2, 2): 15,
    (1, 2, 3): 11,
    (1, 3, 0): 4,
    (1, 3, 1): 12,
    (1, 3, 2): 5,
    (1, 3, 3): 21,
    (2, 0, 0): 2,
    (2, 0, 1): 22,
    (2, 0, 2): 1,
    (2, 0, 3): 23,
    (2, 1, 0): 23,
    (2, 1, 1): 2,
    (2, 1, 2): 22,
    (2, 1, 3): 1,
    (2, 2, 0): 1,
    (2, 2, 1): 23,
    (2, 2, 2): 2,
    (2, 2, 3): 22,
    (2, 3, 0): 22,
    (2, 3, 1): 1,
    (2, 3, 2): 23,
    (2, 3, 3): 2,
}


def one_qubit_clifford_index(gate):
    if gate in one_qubit_clifford_seq_inv:
        return one_qubit_clifford_seq_inv[gate]

    match gate:
        case ('R', phi):
            return one_qubit_clifford_index(
                ('U', pi / 2, phi - pi / 2, pi / 2 - phi))
        case ('rfUnitary', theta, phi) | ('rf', theta, phi):
            return one_qubit_clifford_index(
                ('U', theta, phi - pi / 2, pi / 2 - phi))
        case ('u3', theta, phi, lam) | ('U', theta, phi, lam):
            theta_i = round(4 * np.mod(theta / (2 * pi), 1), 12)
            if np.mod(theta_i, 1) > 1e-9:
                return -1
            theta_i = round(theta_i)
            if theta_i == 1:
                pass
            elif theta_i == 0:
                phi, lam = phi + lam, 0
            elif theta_i == 2:
                phi, lam = phi - lam, 0
            elif theta_i == 3:
                theta_i = 1
                phi, lam = phi + pi, lam + pi
            else:
                return -1
            phi_i = round(4 * np.mod(phi / (2 * pi), 1), 12)
            lam_i = round(4 * np.mod(lam / (2 * pi), 1), 12)
            if np.mod(phi_i, 1) < 1e-9 and np.mod(lam_i, 1) < 1e-9:
                return _one_qubit_clifford_index[theta_i,
                                                 round(phi_i),
                                                 round(lam_i)]
            else:
                return -1
        case ('u2', phi, lam):
            return one_qubit_clifford_index(('U', pi / 2, phi, lam))
        case ('u1', lam) | ('Rz', lam) | ('P', lam):
            return one_qubit_clifford_index(('U', 0, 0, lam))
        case ('Rx', theta):
            return one_qubit_clifford_index(('U', theta, -pi / 2, pi / 2))
        case ('Ry', theta):
            return one_qubit_clifford_index(('U', theta, 0, 0))
        case _:
            return -1


def _two_qubit_clifford_circuit(n,
                                allowed_two_qubit_gates=('CZ', 'CX', 'Cnot',
                                                         'iSWAP', 'SWAP')):
    S1 = [0, 8, 7]  #  I, Rot(2/3 pi, (1,1,1)), Rot(2/3 pi, (1,-1,-1))
    if n < 576:
        i, j = np.unravel_index(n, (24, 24))
        yield (i, 0)
        yield (j, 1)
    elif n < 5760:
        n -= 576
        i, j, k, l = np.unravel_index(n, (24, 24, 3, 3))
        if 'CX' in allowed_two_qubit_gates or 'Cnot' in allowed_two_qubit_gates:
            yield (i, 0)
            yield (j, 1)
            yield ('CX', 0, 1)
            yield (S1[k], 0)
            yield (S1[l], 1)
        elif 'CZ' in allowed_two_qubit_gates:
            yield (i, 0)
            yield (one_qubit_clifford_mul_table[j, 19], 1)
            yield ('CZ', 0, 1)
            yield (S1[k], 0)
            yield (one_qubit_clifford_mul_table[19, S1[l]], 1)
        elif 'iSWAP' in allowed_two_qubit_gates:
            yield (one_qubit_clifford_mul_table[i, 8], 0)
            yield (one_qubit_clifford_mul_table[j, 4], 1)
            yield ('iSWAP', 0, 1)
            yield (4, 1)
            yield ('iSWAP', 0, 1)
            yield (one_qubit_clifford_mul_table[10, S1[k]], 0)
            yield (one_qubit_clifford_mul_table[18, S1[l]], 1)
        else:
            raise ValueError('Imperfect two qubit gate set.')
    elif n < 10944:
        n -= 5760
        i, j, k, l = np.unravel_index(n, (24, 24, 3, 3))
        if 'iSWAP' in allowed_two_qubit_gates:
            yield (i, 0)
            yield (j, 1)
            yield ('iSWAP', 0, 1)
            yield (S1[k], 0)
            yield (S1[l], 1)
        elif 'CZ' in allowed_two_qubit_gates:
            yield (one_qubit_clifford_mul_table[i, 8], 0)
            yield (one_qubit_clifford_mul_table[j, 8], 1)
            yield ('CZ', 0, 1)
            yield (4, 0)
            yield (4, 1)
            yield ('CZ', 0, 1)
            yield (one_qubit_clifford_mul_table[8, S1[k]], 0)
            yield (one_qubit_clifford_mul_table[8, S1[l]], 1)
        elif 'CX' in allowed_two_qubit_gates or 'Cnot' in allowed_two_qubit_gates:
            yield (one_qubit_clifford_mul_table[i, 8], 0)
            yield (one_qubit_clifford_mul_table[j, 4], 1)
            yield ('CX', 0, 1)
            yield (4, 0)
            yield (4, 1)
            yield ('CX', 0, 1)
            yield (one_qubit_clifford_mul_table[8, S1[k]], 0)
            yield (one_qubit_clifford_mul_table[0, S1[l]], 1)
        else:
            raise ValueError('Imperfect two qubit gate set.')
    else:
        n -= 10944
        i, j = np.unravel_index(n, (24, 24))
        if 'SWAP' in allowed_two_qubit_gates:
            yield (i, 0)
            yield (j, 1)
            yield ('SWAP', 0, 1)
        elif 'iSWAP' in allowed_two_qubit_gates:
            yield (i, 0)
            yield (j, 1)
            yield ('iSWAP', 0, 1)
            yield (4, 1)
            yield ('iSWAP', 0, 1)
            yield (8, 0)
            yield ('iSWAP', 0, 1)
            yield (14, 1)
        elif 'CZ' in allowed_two_qubit_gates:
            yield (i, 0)
            yield (one_qubit_clifford_mul_table[j, 19], 1)
            yield ('CZ', 0, 1)
            yield (19, 1)
            yield (19, 0)
            yield ('CZ', 1, 0)
            yield (19, 0)
            yield (19, 1)
            yield ('CZ', 0, 1)
            yield (19, 1)
        elif 'CX' in allowed_two_qubit_gates or 'Cnot' in allowed_two_qubit_gates:
            yield (i, 0)
            yield (j, 1)
            yield ('CX', 0, 1)
            yield ('CX', 1, 0)
            yield ('CX', 0, 1)
        else:
            raise ValueError('Imperfect two qubit gate set.')


def two_qubit_clifford_circuit(n,
                               allowed_two_qubit_gates=('CZ', 'CX', 'Cnot',
                                                        'iSWAP', 'SWAP')):
    """
    生成第 n 个群元对应的操作序列
    """
    circ = []
    for gate, *qubits in _two_qubit_clifford_circuit(n,
                                                     allowed_two_qubit_gates):
        if not isinstance(gate, str):
            gate = one_qubit_clifford_seq[gate]
        circ.append((gate, *qubits))
    return circ


if __name__ == "__main__":
    import itertools

    from qlisp import seq2mat, synchronize_global_phase

    # test the one qubit clifford multiplication table
    for i, j in itertools.product(range(24), repeat=2):
        U1 = seq2mat([(one_qubit_clifford_seq[i], 0),
                      (one_qubit_clifford_seq[j], 0)])
        k = one_qubit_clifford_mul_table[i, j]
        U2 = seq2mat([(one_qubit_clifford_seq[k], 0)])

        U1 = synchronize_global_phase(U1)
        U2 = synchronize_global_phase(U2)

        assert np.allclose(U1, U2)

    # test the one qubit clifford index function
    for i, g in enumerate(one_qubit_clifford_seq):
        assert one_qubit_clifford_index(g) == i

    for g in [
            'H', 'S', 'P', 'Z/2', 'I', 'X', 'Y', 'Z', 'X/2', '-X/2', 'Y/2',
            '-Y/2', '-S', '-Z/2'
    ]:
        U1 = synchronize_global_phase(seq2mat([(g, 0)]))
        U2 = synchronize_global_phase(
            seq2mat([(one_qubit_clifford_seq[one_qubit_clifford_seq_inv[g]], 0)
                     ]))
        assert np.allclose(U1, U2)

    # test the two qubit clifford circuit function

    # 相位规范化后，2 比特 Clifford 群元的矩阵元仅可能取以下值
    elms = [
        0, 0, 0, 0, 1, 1j, -1, -1j, 1 / np.sqrt(2), 1j / np.sqrt(2),
        -1 / np.sqrt(2), -1j / np.sqrt(2), 0.5, 0.5j, -0.5, -0.5j
    ]

    # 速查表，key 为模与相位组成的元组
    # 对于模，0,1,2,3 分别对应 0, 1, 1/sqrt(2), 1/2
    # 对于相位， 0,1,2,3 分别对应 0, pi/2, pi, 3pi/2
    elms_map = {
        k: v
        for k, v in zip(itertools.product(range(4), repeat=2), elms)
    }

    def encode_matrix(mat, norm=True):
        """
        将一个 2 比特 Clifford 群元对应的矩阵转换为 64 位的整数
        
        由于规范化后矩阵元只有 13 种可能取值，故每个矩阵元可用 4 位二进制整数表示，
        4 x 4 的矩阵可以用不超过 64 位的整数表示
        """

        # 仅相隔一个全局相位的操作等价，故令第一个非零的矩阵元相位为 0，保证操作与矩阵一一对应
        if norm:
            mat = synchronize_global_phase(mat)

        absData, phaseData = 0, 0
        for index, (i, j) in zip(itertools.count(start=0, step=2),
                                 itertools.product(range(4), repeat=2)):
            for k, v in elms_map.items():
                if abs(v - mat[i, j]) < 1e-3:
                    a, phase = k
                    break
            else:
                raise ValueError(f"Element {mat[i, j]} not allowed.")
            absData |= a << index
            phaseData |= phase << index
        return absData | (phaseData << 32)

    elements = []

    for n in range(11520):
        seq = two_qubit_clifford_circuit(n)
        U0 = synchronize_global_phase(seq2mat(seq))
        elements.append(encode_matrix(U0))

        seq = two_qubit_clifford_circuit(n, allowed_two_qubit_gates=('CZ', ))
        U = synchronize_global_phase(seq2mat(seq))
        assert np.allclose(U, U0)

        seq = two_qubit_clifford_circuit(n, allowed_two_qubit_gates=('CX', ))
        U = synchronize_global_phase(seq2mat(seq))
        assert np.allclose(U, U0)

        seq = two_qubit_clifford_circuit(n,
                                         allowed_two_qubit_gates=('iSWAP', ))
        U = synchronize_global_phase(seq2mat(seq))
        assert np.allclose(U, U0)

    assert len(set(elements)) == 11520
