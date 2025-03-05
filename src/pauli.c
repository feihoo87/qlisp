#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

// 检查是否是 x86 平台并且支持 POPCNT 指令
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#if defined(__POPCNT__) || (defined(_MSC_VER) && defined(__AVX__))
#define HAS_POPCNT
#endif
#elif defined(__arm__) || defined(__aarch64__)
#define HAS_ARM
#endif

/*
 * 计算一个无符号整数 n 的二进制表示中 1 的个数
 */
static inline unsigned int bit_count(unsigned int n)
{
    unsigned int count = 0;

#ifdef HAS_POPCNT
#if defined(_MSC_VER) // 如果是 MSVC 编译器
    count = __popcnt(n);
#else // 其他支持 POPCNT 的编译器 (如 GCC)
    __asm__(
        "movl %1, %%eax;"      // 将输入值 n 移动到 eax 寄存器
        "popcnt %%eax, %%eax;" // 使用 popcnt 指令计算位计数
        "movl %%eax, %0;"      // 将结果存储到输出变量 count
        : "=r"(count)          // 输出操作数
        : "r"(n)               // 输入操作数
        : "%eax"               // 受影响的寄存器
    );
#endif
#elif defined(HAS_ARM)
    count = __builtin_popcount(n);
#else
    // 如果不支持 POPCNT 指令，使用一个手动计算的方法
    while (n)
    {
        count += n & 1;
        n >>= 1;
    }
#endif

    return count;
}

// 将一个 64 位整数 n 按奇数位和偶数位拆分为两个 32 位整数 x 和 z
#define split_index_uint64(n, x, z)             \
    {                                           \
        x = 0;                                  \
        z = 0;                                  \
        for (uint64_t i = 0; i < 32; i++)       \
        {                                       \
            x |= (((n) >> 2 * i) & 1) << i;     \
            z |= (((n) >> 2 * i + 1) & 1) << i; \
        }                                       \
    }

#define X_mask 0x5555555555555555ULL
#define Z_mask 0xAAAAAAAAAAAAAAAAULL

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
    return ((bit_count(h) << 1) ^ bit_count(l)) & 3;
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

void pauli_imul(uint64_t *left, uint64_t *right, size_t N)
{
    uint64_t sign = *left + *right;
    uint64_t *first = right;
    uint64_t a = *left >> 2, b = *right >> 2;
    uint64_t c;
    sign += int_pauli_mul(a, b, &c);
    *right = c << 2;

    left++;
    right++;
    N--;

    while (N)
    {
        a = *left, b = *right;
        sign += int_pauli_mul(a, b, &c);
        *right = c;
        left++;
        right++;
        N--;
    }
    *first |= sign & 3;
}

/*
 * 计算第 n 个 Pauli 矩阵的第 r 行，第 c 列的元素
 * 基底按照 IXZY 的顺序排列，即算符按
 * I...II, I...IX, I...IZ, I...IY, I...XI, I...XX, I...XZ, I...XY, ...
 * 的顺序排列
 * 返回值 0, 1, 2, 3 分别代表 1, i, -1, -i， 4 代表 0
 */
uint64_t pauli_xzy_tensor_element_int(uint64_t n, uint64_t r, uint64_t c)
{
    uint64_t x = 0;
    uint64_t z = 0;

    split_index_uint64(n, x, z);

    if (x ^ r != c)
        return 4;

    // 0: 1, 1: i, 2: -1, 3: -i, 4 : 0
    return (bit_count(x & z) + (bit_count(z & c) << 1)) & 3;
}

/*
 * 计算第 n 个 Pauli 矩阵的第 r 行，第 c 列的元素
 * 基底按照 IXYZ 的顺序排列，即算符按
 * I...II, I...IX, I...IY, I...IZ, I...XI, I...XX, I...XY, I...XZ, ...
 * 的顺序排列
 * 返回值 0, 1, 2, 3 分别代表 1, i, -1, -i， 4 代表 0
 */
uint64_t pauli_xyz_tensor_element_int(uint64_t n, uint64_t r, uint64_t c)
{
    uint64_t x = 0;
    uint64_t z = 0;

    split_index_uint64(n, x, z);
    x = x ^ z;

    if (x ^ r != c)
        return 4;

    // 0: 1, 1: i, 2: -1, 3: -i, 4 : 0
    return (bit_count(x & z) + (bit_count(z & c) << 1)) & 3;
}

typedef struct
{
    uint8_t sign;
    size_t number_of_qubit;
    size_t size;
    uint64_t *x;
    uint64_t *z;
} PauliOperator;

typedef struct
{
    size_t number_of_qubit;
    size_t size;
    uint8_t *sign;
    uint64_t *x;
    uint64_t *z;
} Stabilizers;

void imul_paulis(PauliOperator *a, PauliOperator *b)
{
    uint64_t x, z, h, l;
    uint64_t *x1 = a->x, *x2 = b->x, *z1 = a->z, *z2 = b->z;
    a->sign += b->sign;
    for (size_t i = 0; i < a->size && i < b->size; i++, x1++, x2++, z1++, z2++)
    {
        x = *x1 ^ *x2;
        z = *z1 ^ *z2;

        // find the qubits that any x1,z1,x2,z2 != 0 and x1,z1 != x2,z2
        l = (*x1 | *z1) & (*x2 | *z2) & (x | z);

        h = ((*z2 & *x1) ^ (z & x)) & l;

        a->sign += 2 * bit_count(h) + bit_count(l);
        *x1 = x;
        *z1 = z;
    }
    a->sign &= 3;
}

PauliOperator *create_pauli_operator(size_t number_of_qubit)
{
    PauliOperator *pauli = (PauliOperator *)malloc(sizeof(PauliOperator));
    pauli->sign = 0;
    pauli->number_of_qubit = number_of_qubit;
    pauli->size = (number_of_qubit + 63) / 64;
    pauli->x = (uint64_t *)calloc(pauli->size, sizeof(uint64_t));
    pauli->z = (uint64_t *)calloc(pauli->size, sizeof(uint64_t));
    return pauli;
}

void destroy_pauli_operator(PauliOperator *pauli)
{
    free(pauli->x);
    free(pauli->z);
    free(pauli);
}

PauliOperator *tenser_product(PauliOperator *a, PauliOperator *b)
{
    PauliOperator *c = create_pauli_operator(a->number_of_qubit + b->number_of_qubit);
    c->sign = a->sign + b->sign;
    memcpy(c->x, a->x, a->size * sizeof(uint64_t));
    memcpy(c->z, a->z, a->size * sizeof(uint64_t));

    if (64 * a->size == a->number_of_qubit)
    {
        memcpy(c->x + a->size, b->x, b->size * sizeof(uint64_t));
        memcpy(c->z + a->size, b->z, b->size * sizeof(uint64_t));
    }
    else
    {
        int rest_bit = 64 * a->size - a->number_of_qubit;
        uint64_t *x = c->x + a->size - 1;
        uint64_t *z = c->z + a->size - 1;

        *x |= b->x[0] << rest_bit;
        *z |= b->z[0] << rest_bit;

        for (size_t i = 0; i < b->size - 1; i++, x++, z++)
        {
            *x = b->x[i] >> (64 - rest_bit) | b->x[i + 1] << rest_bit;
            *z = b->z[i] >> (64 - rest_bit) | b->z[i + 1] << rest_bit;
        }

        *x = b->x[b->size - 1] >> (64 - rest_bit);
        *z = b->z[b->size - 1] >> (64 - rest_bit);
    }

    c->sign &= 3;
    return c;
}

PauliOperator *load_from_string(const char *str)
{
    size_t number_of_qubit = strlen(str);
    uint8_t sign = 0;

    if (*str == '+')
    {
        str++;
        number_of_qubit--;
    }
    if (*str == '-')
    {
        sign += 2;
        str++;
        number_of_qubit--;
    }
    if (*str == 'i')
    {
        sign += 1;
        str++;
        number_of_qubit--;
    }

    PauliOperator *pauli = create_pauli_operator(number_of_qubit);

    pauli->sign = sign & 3;

    for (size_t i = 0; i < number_of_qubit; i++)
    {
        int b = i % 64;
        switch (*str++)
        {
        case 'X':
            pauli->x[i / 64] |= 1ULL << b;
            break;
        case 'Y':
            pauli->x[i / 64] |= 1ULL << b;
            pauli->z[i / 64] |= 1ULL << b;
            break;
        case 'Z':
            pauli->z[i / 64] |= 1ULL << b;
            break;
        default:
            break;
        }
    }
    return pauli;
}

void print_pauli_operator(PauliOperator *pauli)
{
    switch (pauli->sign)
    {
    case 0:
        break;
    case 1:
        printf("i");
        break;
    case 2:
        printf("-");
        break;
    case 3:
        printf("-i");
        break;
    }
    for (size_t i = 0; i < pauli->number_of_qubit; i++)
    {
        int b = i % 64;
        switch (((pauli->x[i / 64] >> b) & 1) + 2 * ((pauli->z[i / 64] >> b) & 1))
        {
        case 0:
            printf("I");
            break;
        case 1:
            printf("X");
            break;
        case 2:
            printf("Z");
            break;
        case 3:
            printf("Y");
            break;
        }
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    PauliOperator *a = load_from_string(*(argv + 1));
    PauliOperator *b = load_from_string(*(argv + 2));
    print_pauli_operator(a);
    print_pauli_operator(b);
    imul_paulis(a, b);
    print_pauli_operator(a);
    destroy_pauli_operator(a);
    destroy_pauli_operator(b);
    return 0;
}
