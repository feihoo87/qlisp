#ifndef __BIT_COUNT_H__
#define __BIT_COUNT_H__

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

#endif // __BIT_COUNT_H__