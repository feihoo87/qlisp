#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <Python.h>

#include "pauli.h"

#if PY_VERSION_HEX >= 0x030D0000  /* Python 3.13.0 */
static inline int PyLong_AsByteArray(
    PyLongObject *v,
    unsigned char *bytes,
    size_t n)
{
    return PyLong_AsNativeBytes(
        (PyObject *)v,
        bytes,
        n,
        Py_ASNATIVEBYTES_LITTLE_ENDIAN | Py_ASNATIVEBYTES_UNSIGNED_BUFFER); // 小端序, 无符号
}
#else
static inline int PyLong_AsByteArray(
    PyLongObject *v,
    unsigned char *bytes,
    size_t n)
{
    int is_little_endian = 1; // 小端序
    int is_signed = 0;        // 无符号

    return _PyLong_AsByteArray(v, bytes, n, is_little_endian, is_signed);
}
#endif


// Python C API

static PyObject *pauli_mul(PyObject *self, PyObject *args)
{
    PyObject *a, *b;
    if (!PyArg_ParseTuple(args, "OO", &a, &b))
    {
        return NULL;
    }

    if (!(PyLong_Check(a) && PyLong_Check(b)))
    {
        PyErr_SetString(PyExc_TypeError, "Arguments must be integers");
        return NULL;
    }

    // 计算字节数
    Py_ssize_t bits_a = _PyLong_NumBits(a);
    Py_ssize_t bits_b = _PyLong_NumBits(b);
    Py_ssize_t num_bits = (bits_a > bits_b) ? bits_a : bits_b;
    Py_ssize_t length = (num_bits + 7) / 8;
    Py_ssize_t num_of_uint64 = (length + 7) / 8;
    length = num_of_uint64 * 8;

    // 处理 length 为 0 的情况
    if (length == 0)
    {
        return Py_BuildValue("iO", 0, PyLong_FromLong(0));
    }
    // 处理 length 为 1 的情况
    if (length == 1)
    {
        uint64_t a_int = PyLong_AsUnsignedLong(a);
        uint64_t b_int = PyLong_AsUnsignedLong(b);
        uint64_t sign = 0, result = 0;
        sign = int_pauli_mul(a_int, b_int, &result);
        return Py_BuildValue("iK", sign & 3, result);
    }

    unsigned char *buffer = (unsigned char *)malloc(length * 3);
    if (!buffer)
    {
        PyErr_NoMemory();
        return NULL;
    }

    int ret = PyLong_AsByteArray(
        (PyLongObject *)a,
        buffer,
        length);
    if (ret == -1)
    {
        free(buffer);
        return NULL;
    }
    ret = PyLong_AsByteArray(
        (PyLongObject *)b,
        buffer + length,
        length);
    if (ret == -1)
    {
        free(buffer);
        return NULL;
    }

    // 初始化指针
    const uint64_t *a_data = (const uint64_t *)buffer;
    const uint64_t *b_data = (const uint64_t *)buffer + num_of_uint64;
    uint64_t *result = (uint64_t *)buffer + 2 * num_of_uint64;

    uint64_t sign = 0;

    for (Py_ssize_t i = 0; i < num_of_uint64; i++)
    {
        sign += int_pauli_mul(a_data[i], b_data[i], &result[i]);
    }

#if PY_VERSION_HEX >= 0x030D0000  /* Python 3.13.0 */
    /* 使用新公共 API */
    PyObject *result_int = PyLong_FromNativeBytes(
        (const void *)result,
        (size_t)length,
        Py_ASNATIVEBYTES_LITTLE_ENDIAN | Py_ASNATIVEBYTES_UNSIGNED_BUFFER
    );
#else
    /* 使用老的私有 API（Python ≤3.12）*/
    PyObject *result_int = (PyObject *)_PyLong_FromByteArray(
        (const unsigned char *)result,
        length,
        1,
        0
    );
#endif

    free(buffer);
    return Py_BuildValue("iO", sign & 3, result_int);
}

static PyObject *pauli_element(PyObject *self, PyObject *args)
{
    PyObject *r, *c, *N;
    int *order;
    if (!PyArg_ParseTuple(args, "OOOi", &N, &r, &c, &order))
    {
        return NULL;
    }

    Py_ssize_t bits_r = 2 * _PyLong_NumBits(r);
    Py_ssize_t bits_c = 2 * _PyLong_NumBits(c);
    Py_ssize_t bits_N = _PyLong_NumBits(N);
    Py_ssize_t num_bits = (bits_r > bits_c) ? bits_r : bits_c;
    num_bits = (num_bits > bits_N) ? num_bits : bits_N;
    Py_ssize_t length = (num_bits + 7) / 8;
    Py_ssize_t num_of_uint64 = (length + 7) / 8;

    // 处理 length 为 0 的情况
    if (length == 0)
    {
        return Py_BuildValue("i", 0);
    }
    // 处理 length 为 1 的情况
    if (length == 1)
    {
        uint64_t r_int = PyLong_AsUnsignedLong(r);
        uint64_t c_int = PyLong_AsUnsignedLong(c);
        uint64_t N_int = PyLong_AsUnsignedLong(N);
        uint64_t result = 0;
        if (order == 0)
        {
            result = pauli_xzy_tensor_element_int(N_int, r_int, c_int);
        }
        else
        {
            result = pauli_xyz_tensor_element_int(N_int, r_int, c_int);
        }
        return Py_BuildValue("i", result);
    }

    unsigned char *buffer = (unsigned char *)malloc(length * 3);
    if (!buffer)
    {
        PyErr_NoMemory();
        return NULL;
    }

    int ret = PyLong_AsByteArray(
        (PyLongObject *)r,
        buffer,
        length);
    if (ret == -1)
    {
        free(buffer);
        return NULL;
    }
    ret = PyLong_AsByteArray(
        (PyLongObject *)c,
        buffer + length,
        length);
    if (ret == -1)
    {
        free(buffer);
        return NULL;
    }
    ret = PyLong_AsByteArray(
        (PyLongObject *)N,
        buffer + 2 * length,
        length);
    if (ret == -1)
    {
        free(buffer);
        return NULL;
    }

    // 初始化指针
    const uint32_t *r_data = (const uint32_t *)buffer;
    const uint32_t *c_data = (const uint32_t *)(buffer + length);
    const uint64_t *N_data = (const uint64_t *)(buffer + 2 * length);

    uint64_t res, result = 0;
    for (Py_ssize_t i = 0; i < num_of_uint64; i++)
    {
        if (order == 0)
        {
            res = pauli_xzy_tensor_element_int(N_data[i], r_data[i], c_data[i]);
        }
        else
        {
            res = pauli_xyz_tensor_element_int(N_data[i], r_data[i], c_data[i]);
        }
        if (res == 4)
        {
            return Py_BuildValue("i", 4);
        }
        result += res;
    }
    // double real = 1.0, imag = 0.0;
    // complex_rot(&real, &imag, result);
    // return PyComplex_FromDoubles(real, imag);

    return Py_BuildValue("i", result & 3);
}

static PyObject *pauli_nozero_element(PyObject *self, PyObject *args)
{
    PyObject *r, *N;

    if (!PyArg_ParseTuple(args, "OO", &N, &r))
    {
        return NULL;
    }

    Py_ssize_t bits_r = 2 * _PyLong_NumBits(r);
    Py_ssize_t bits_N = _PyLong_NumBits(N);
    Py_ssize_t num_bits = bits_r;
    num_bits = (num_bits > bits_N) ? num_bits : bits_N;
    Py_ssize_t length = (num_bits + 7) / 8;
    Py_ssize_t num_of_uint64 = (length + 7) / 8;

    // 处理 length 为 0 的情况
    if (length == 0)
    {
        return Py_BuildValue("ii", 0, 0);
    }
    // 处理 length 为 1 的情况
    if (length == 1)
    {
        uint64_t r_int = PyLong_AsUnsignedLong(r);
        uint64_t N_int = PyLong_AsUnsignedLong(N);
        uint64_t c_int = 0;
        uint64_t result = 0;
        result = pauli_xzy_tensor_element_int(N_int, r_int, c_int);
        return Py_BuildValue("ii", c_int, result);
    }

    unsigned char *buffer = (unsigned char *)malloc(length * 3);
    if (!buffer)
    {
        PyErr_NoMemory();
        return NULL;
    }

    int ret = PyLong_AsByteArray(
        (PyLongObject *)r,
        buffer,
        length);
    if (ret == -1)
    {
        free(buffer);
        return NULL;
    }
    ret = PyLong_AsByteArray(
        (PyLongObject *)N,
        buffer + length,
        length);
    if (ret == -1)
    {
        free(buffer);
        return NULL;
    }
    ret = PyLong_AsByteArray(
        (PyLongObject *)N,
        buffer + 2 * length,
        length);
    if (ret == -1)
    {
        free(buffer);
        return NULL;
    }

    // 初始化指针
    const uint32_t *r_data = (const uint32_t *)buffer;
    const uint32_t *c_data = (const uint32_t *)(buffer + length);
    const uint64_t *N_data = (const uint64_t *)(buffer + 2 * length);

    uint64_t res, result = 0;
    for (Py_ssize_t i = 0; i < num_of_uint64; i++)
    {
        res = pauli_xzy_tensor_element_int(N_data[i], r_data[i], c_data[i]);
        if (res == 4)
        {
            return Py_BuildValue("i", 4);
        }
        result += res;
    }
    // double real = 1.0, imag = 0.0;
    // complex_rot(&real, &imag, result);
    // return PyComplex_FromDoubles(real, imag);

    return Py_BuildValue("i", result & 3);
}

// 生成bytes的示例函数
static PyObject *bytes_generate(PyObject *self, PyObject *args)
{
    unsigned char buffer[5] = {0x48, 0x65, 0x6c, 0x6c, 0x6f}; // "Hello"的二进制表示
    return PyBytes_FromStringAndSize((const char *)buffer, 5);
}

typedef struct
{
    PyObject_HEAD size_t number_of_qubit;
} PauliMatrixElementGeneratorObject;

static PyObject *PauliMatrixElementGenerator_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PauliMatrixElementGeneratorObject *self;
    self = (PauliMatrixElementGeneratorObject *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        PyBytes_AsString(PyBytes_FromString("Hello, world!"));
        self->number_of_qubit = 0;
        // ...
    }
    return (PyObject *)self;
}

static int PauliMatrixElementGenerator_init(PauliMatrixElementGeneratorObject *self, PyObject *args, PyObject *kwds)
{
    return 0;
}

static PyObject *PauliMatrixElementGenerator_iter(PyObject *self)
{
    Py_INCREF(self);
    return self;
}

static PyObject *PauliMatrixElementGenerator_next(PauliMatrixElementGeneratorObject *self)
{
    while (1)
    {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
}

static void PauliMatrixElementGenerator_dealloc(PauliMatrixElementGeneratorObject *self)
{
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyTypeObject PauliMatrixElementGeneratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "PauliMatrixElementGenerator",
    .tp_doc = "Pauli matrix element generator",
    .tp_basicsize = sizeof(PauliMatrixElementGeneratorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PauliMatrixElementGenerator_new,
    .tp_init = (initproc)PauliMatrixElementGenerator_init,
    .tp_dealloc = (destructor)PauliMatrixElementGenerator_dealloc,
    .tp_iter = PauliMatrixElementGenerator_iter,
    .tp_iternext = (iternextfunc)PauliMatrixElementGenerator_next,
};

static PyMethodDef PauliMethods[] = {
    {"mul", pauli_mul, METH_VARARGS, "Multiply two Pauli matrices"},
    {"element", pauli_element, METH_VARARGS, "Get the element of a Pauli matrix"},
    {"_generate", bytes_generate, METH_NOARGS, "Generate a bytes object"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef Pauli_module = {
    PyModuleDef_HEAD_INIT,
    "_pauli",
    NULL,
    -1,
    PauliMethods};

PyMODINIT_FUNC PyInit__pauli(void)
{
    PyObject *m;
    m = PyModule_Create(&Pauli_module);
    return m;
}
