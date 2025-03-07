#ifndef __PAULI_H__
#define __PAULI_H__

#include <stdint.h>

#include "bit_count.h"

// 将一个 64 位整数 n 按奇数位和偶数位拆分为两个 32 位整数 x 和 z
#define split_index_uint64(n, x, z)                 \
    {                                               \
        (x) = 0;                                    \
        (z) = 0;                                    \
        for (uint64_t i = 0; i < 32; i++)           \
        {                                           \
            (x) |= (((n) >> (2 * i)) & 1) << i;     \
            (z) |= (((n) >> (2 * i + 1)) & 1) << i; \
        }                                           \
    }

/*
 * 将一个长度为 N 的 64 位整数数组所表示的大整数 n 按奇数位和偶数位拆分为
 * 两个长度为 N/2 的 64 位整数数组 x 和 z
 * 整数按照 little-endian 排列
 * N 必须是偶数
 */
static inline void split_index(uint64_t N, uint64_t *n, uint64_t *x, uint64_t *z)
{
    uint64_t x1, z1, x2, z2;

    for (uint64_t i = 0; i < N; i += 2)
    {
        split_index_uint64(*n++, x1, z1);
        split_index_uint64(*n++, x2, z2);
        *x++ = x1 | (x2 << 32);
        *z++ = z1 | (z2 << 32);
    }
}

static const uint64_t X_mask = 0x5555555555555555ULL;
static const uint64_t Z_mask = 0xAAAAAAAAAAAAAAAAULL;

/*
 * 将一个复数原位逆时针旋转 phase 角度
 * phase = 0, 1, 2, 3 分别代表 0°, 90°, 180°, 270°
 * phase = 4 代表结果清零
 * real 和 imag 是输入和输出的实部和虚部
 */
static inline void complex_rot(double *real, double *imag, uint64_t phase)
{
    double tmp = *real;
    switch (phase)
    {
    case 0:
        break;
    case 1:
        *real = -*imag;
        *imag = tmp;
        break;
    case 2:
        *real = -*real;
        *imag = -*imag;
        break;
    case 3:
        *real = *imag;
        *imag = -tmp;
        break;
    default:
        *real = 0.0;
        *imag = 0.0;
        break;
    }
}

/*
 * 计算两个 Pauli 矩阵的乘积
 * a 和 b 是两个 Pauli 矩阵的序号，res 是输出的 Pauli 矩阵的序号
 * 返回值是结果的附加系数。即：
 * Paulis[a] * Paulis[b] = sign(ret) * Paulis[res]
 *
 * 如果基底按照 IXYZ 的顺序排列
 * 即算符按 I...II, I...IX, I...IY, I...IZ, I...XI, I...XX, I...XY, I...XZ, ... 的顺序排列
 * 则返回值 0, 1, 2, 3 分别代表 1, -i, -1, i
 *
 * 如果基底按照 IZXY 的顺序排列
 * 即算符按 I...II, I...IX, I...IZ, I...IY, I...XI, I...XX, I...XZ, I...XY, ... 的顺序排列
 * 则返回值 0, 1, 2, 3 分别代表 1, i, -1, -i
 */
static inline uint64_t int_pauli_mul(uint64_t a, uint64_t b, uint64_t *res)
{
    uint64_t c = a ^ b;
    uint64_t az = a >> 1, bz = b >> 1, cz = c >> 1;

    uint64_t l = (a | az) & (b | bz) & (c | cz) & X_mask;
    uint64_t h = ((az & b) ^ (c & cz)) & l;
    *res = c;

    // if Pauli matirx is sorted as I, X, Y, Z
    // the sign is 1, -i, -1, i
    // if Pauli matirx is sorted as I, X, Z, Y
    // the sign is 1, i, -1, -i
    return ((bit_count(h) << 1) ^ bit_count(l));
}

/*
 * 计算两个 Pauli 矩阵的乘积
 * 与 int_pauli_mul 的区别是，参数和返回值用最低两位表示 Pauli 矩阵的附加系数
 */
static inline uint64_t int_pauli_mul_with_sign(uint64_t a, uint64_t b)
{
    uint64_t sign = a + b;
    a >>= 2;
    b >>= 2;
    uint64_t c = a ^ b;
    uint64_t az = a >> 1, bz = b >> 1, cz = c >> 1;

    uint64_t l = (a | az) & (b | bz) & (c | cz) & X_mask;
    uint64_t h = ((az & b) ^ (c & cz)) & l;

    // if Pauli matirx is sorted as I, X, Y, Z
    // the sign is 1, -i, -1, i
    // if Pauli matirx is sorted as I, X, Z, Y
    // the sign is 1, i, -1, -i
    sign += (bit_count(h) << 1) ^ bit_count(l);
    return (sign & 3) | (c << 2);
}

/*
 * 计算由 x，z 标识的 Pauli 矩阵的第 r 行的非零元素
 * c 是非零元素的列，sign 是输出的非零元素的值
 * sign % 4 的值 0, 1, 2, 3 分别代表 1, i, -1, -i
 */
static inline void pauli_tensor_nozero_element(uint64_t x, uint64_t z, uint64_t r, uint64_t *c, uint64_t *sign)
{
    *c = x ^ r;
    // 0: 1, 1: i, 2: -1, 3: -i, 4 : 0
    *sign += bit_count(x & z) + (bit_count(z & *c) << 1);
}

/*
 * 计算第 n 个 Pauli 矩阵的第 r 行，第 c 列的元素
 * 基底按照 IXZY 的顺序排列，即算符按
 * I...II, I...IX, I...IZ, I...IY, I...XI, I...XX, I...XZ, I...XY, ...
 * 的顺序排列
 * 返回值 0, 1, 2, 3 分别代表 1, i, -1, -i， 4 代表 0
 */
static inline uint64_t pauli_xzy_tensor_element_int(uint64_t n, uint64_t r, uint64_t c)
{
    uint64_t c1, sign = 0;
    uint64_t x = 0;
    uint64_t z = 0;

    split_index_uint64(n, x, z);

    pauli_tensor_nozero_element(x, z, r, &c1, &sign);

    if (c1 != c)
        return 4;

    // 0: 1, 1: i, 2: -1, 3: -i, 4 : 0
    return sign & 3;
}

/*
 * 计算第 n 个 Pauli 矩阵的第 r 行，第 c 列的元素
 * 基底按照 IXYZ 的顺序排列，即算符按
 * I...II, I...IX, I...IY, I...IZ, I...XI, I...XX, I...XY, I...XZ, ...
 * 的顺序排列
 * 返回值 0, 1, 2, 3 分别代表 1, i, -1, -i， 4 代表 0
 */
static inline uint64_t pauli_xyz_tensor_element_int(uint64_t n, uint64_t r, uint64_t c)
{
    uint64_t c1, sign = 0;
    uint64_t x = 0;
    uint64_t z = 0;

    split_index_uint64(n, x, z);
    x = x ^ z;

    pauli_tensor_nozero_element(x, z, r, &c1, &sign);

    if (c1 != c)
        return 4;

    // 0: 1, 1: i, 2: -1, 3: -i, 4 : 0
    return sign & 3;
}

#endif // __PAULI_H__