#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <string.h>

typedef struct
{
    size_t max_row;
    size_t max_col;
    size_t max_size;
    size_t length;
    npy_complex128 *data;
    npy_intp *dims;
} Operators;

npy_complex128 paulis_data[] = {
    {1.0, 0.0}, // 0
    {0.0, 0.0},
    {0.0, 0.0},
    {1.0, 0.0}, // 3

    {0.0, 0.0},
    {1.0, 0.0}, // 5
    {1.0, 0.0}, // 6
    {0.0, 0.0},

    {0.0, 0.0},
    {0.0, -1.0}, // 9
    {0.0, 1.0},  // 10
    {0.0, 0.0},

    {1.0, 0.0}, // 12
    {0.0, 0.0},
    {0.0, 0.0},
    {-1.0, 0.0}, // 15
};
npy_intp paulis_dims[] = {2, 2, 2, 2, 2, 2, 2, 2};

Operators paulis = {
    .data = paulis_data,
    .dims = paulis_dims,
    .max_row = 2,
    .max_col = 2,
    .max_size = 4,
    .length = 4,
};

static inline size_t power(size_t a, size_t N)
{
    size_t result = 1;
    size_t base = a;

    while (N > 0)
    {
        // If N is odd, multiply the result with the current base
        if (N % 2 == 1)
        {
            result *= base;
        }
        // Square the base
        base *= base;
        // Divide N by 2
        N /= 2;
    }

    return result;
}

void _pauli_tensor_element(
    size_t op_list_len,
    size_t n,
    size_t r, size_t c,
    double *real, double *imag)
{
    double re, im;

    *real = 1.0;
    *imag = 0.0;
    n <<= 2;

    for (int i = op_list_len - 1; i >= 0; i--)
    {
        unsigned int mask = 0x9669;

        size_t j = (n & 0xC) | ((r << 1) & 2) | (c & 1);
        n >>= 2;

        if ((mask & (1 << j)) == 0)
        {
            *real = 0.0;
            *imag = 0.0;
            return;
        }

        npy_complex128 value = paulis_data[j];

        re = *real * value.real - *imag * value.imag;
        im = *real * value.imag + *imag * value.real;

        *real = re;
        *imag = im;

        r >>= 1;
        c >>= 1;
    }
}

void _tensor_element(
    Operators *operators,
    size_t op_list_len,
    npy_intp *op_list,
    size_t r, size_t c,
    double *real, double *imag)
{
    double re, im;

    *real = 1.0;
    *imag = 0.0;

    for (int i = op_list_len - 1; i >= 0; i--)
    {
        size_t rows = operators->dims[op_list[i] * 2];
        size_t cols = operators->dims[op_list[i] * 2 + 1];
        size_t j = (r % rows) * operators->max_col + c % cols;

        npy_complex128 value = operators->data[op_list[i] * operators->max_size + j];

        if (value.real == 0.0 && value.imag == 0.0)
        {
            *real = 0.0;
            *imag = 0.0;
            return;
        }

        re = *real * value.real - *imag * value.imag;
        im = *real * value.imag + *imag * value.real;

        *real = re;
        *imag = im;

        r /= rows;
        c /= cols;
    }
}

double _qst_mat_element(
    Operators *gate_set,
    size_t op_list_len,
    npy_intp *op_list,
    size_t i, size_t j)
{
    // U[ik] * P[j, kl] * conj(U[il])
    double re, im, re2, im2, re3, im3, tmp;
    double result = 0.0;

    for (size_t k = 0; k < (1 << op_list_len); k++)
    {
        _tensor_element(gate_set, op_list_len, op_list, i, k, &re, &im);
        // printf("U[%d%d] ==> re: %f, im: %f\n", i, k, re, im);
        if (re == 0.0 && im == 0.0)
        {
            continue;
        }
        re3 = re;
        im3 = im;
        for (size_t l = 0; l < (1 << op_list_len); l++)
        {
            re2 = re3;
            im2 = im3;
            _pauli_tensor_element(op_list_len, j, k, l, &re, &im);
            // printf("P[%d,%d%d] ==> re: %f, im: %f\n", j, k, l, re, im);
            if (re == 0.0 && im == 0.0)
            {
                continue;
            }
            tmp = re2 * re - im2 * im;
            im2 = re2 * im + im2 * re;
            re2 = tmp;
            _tensor_element(gate_set, op_list_len, op_list, i, l, &re, &im);
            // printf("U[%d%d] ==> re: %f, im: %f\n", i, l, re, im);
            result += re2 * re + im2 * im;
        }
    }
    return result;
}

double _qpt_mat_element(Operators *gate_set,
                        size_t N, npy_intp *before_op_list, npy_intp *after_op_list,
                        size_t m, size_t n, size_t i)
{
    double re, im, real, imag, tmp, ret = 0.0;
    size_t dim = 1 << N;

    for (size_t j = 0; j < dim; j++)
    {
        for (size_t s = 0; s < dim; s++)
        {
            for (size_t k = 0; k < dim; k++)
            {
                for (size_t q = 0; q < dim; q++)
                {
                    real = 1.0;
                    imag = 0.0;
                    _tensor_element(gate_set, N, before_op_list, k, 0, &re, &im);
                    tmp = real * re - imag * im;
                    imag = real * im + imag * re;
                    real = tmp;
                    _tensor_element(gate_set, N, before_op_list, q, 0, &re, &im);
                    tmp = real * re + imag * im;
                    imag = -real * im + imag * re;
                    real = tmp;
                    _tensor_element(gate_set, N, after_op_list, i, j, &re, &im);
                    tmp = real * re - imag * im;
                    imag = real * im + imag * re;
                    real = tmp;
                    _tensor_element(gate_set, N, after_op_list, i, s, &re, &im);
                    tmp = real * re + imag * im;
                    imag = -real * im + imag * re;
                    real = tmp;
                    _pauli_tensor_element(N, m, s, j, &re, &im);
                    tmp = real * re - imag * im;
                    imag = real * im + imag * re;
                    real = tmp;
                    _pauli_tensor_element(N, n, q, k, &re, &im);
                    tmp = real * re - imag * im;
                    imag = real * im + imag * re;
                    real = tmp;

                    ret += real;
                }
            }
        }
    }

    return ret;
}

static inline void _load_operators(
    Operators *operators,
    PyArrayObject *gate_set_obj, PyArrayObject *dims_obj)
{
    operators->data = (npy_complex128 *)PyArray_DATA(gate_set_obj);
    operators->dims = (npy_intp *)PyArray_DATA(dims_obj);
    operators->max_row = (size_t)PyArray_DIM(gate_set_obj, 1);
    operators->max_col = (size_t)PyArray_DIM(gate_set_obj, 2);
    operators->max_size = operators->max_row * operators->max_col;
    operators->length = (size_t)PyArray_DIM(gate_set_obj, 0);
}

static PyObject *tensor_element(PyObject *self, PyObject *args)
{
    PyArrayObject *op_list_obj, *operators_obj, *dims_obj;
    Operators operators;
    size_t r, c;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "OOOnn", &operators_obj, &dims_obj, &op_list_obj, &r, &c))
    {
        return NULL;
    }

    // Ensure the inputs are numpy arrays
    if (!PyArray_Check(op_list_obj) || !PyArray_Check(operators_obj) || !PyArray_Check(dims_obj))
    {
        PyErr_SetString(PyExc_TypeError, "Inputs must be numpy arrays");
        return NULL;
    }

    // Get pointers to the data as C-types
    npy_intp *op_list = (npy_intp *)PyArray_DATA(op_list_obj);
    int op_list_len = (int)PyArray_DIM(op_list_obj, 0);

    _load_operators(&operators, operators_obj, dims_obj);

    double real, imag;

    _tensor_element(&operators, op_list_len, op_list, r, c, &real, &imag);

    return PyComplex_FromDoubles(real, imag);
}

static PyObject *qst_mat_element(PyObject *self, PyObject *args)
{
    PyArrayObject *gate_set_obj, *dims_obj, *op_list_obj;
    Operators gate_set;
    size_t r, c;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "OOOnn", &gate_set_obj, &dims_obj, &op_list_obj, &r, &c))
    {
        return NULL;
    }

    // Ensure the inputs are numpy arrays
    if (!PyArray_Check(op_list_obj) || !PyArray_Check(gate_set_obj) || !PyArray_Check(dims_obj))
    {
        PyErr_SetString(PyExc_TypeError, "Inputs must be numpy arrays");
        return NULL;
    }

    // Get pointers to the data as C-types
    npy_intp *op_list = (npy_intp *)PyArray_DATA(op_list_obj);
    int op_list_len = (int)PyArray_DIM(op_list_obj, 0);

    _load_operators(&gate_set, gate_set_obj, dims_obj);

    return PyFloat_FromDouble(_qst_mat_element(&gate_set, op_list_len, op_list, r, c) / (1 << op_list_len));
}

static PyObject *qpt_mat_element(PyObject *self, PyObject *args)
{
    PyArrayObject *gate_set_obj, *dims_obj, *before_list_obj, *after_list_obj;
    Operators gate_set;
    size_t m, n, i;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "OOOOnnn", &gate_set_obj, &dims_obj, &before_list_obj, &after_list_obj, &m, &n, &i))
    {
        return NULL;
    }

    // Ensure the inputs are numpy arrays
    if (!PyArray_Check(before_list_obj) || !PyArray_Check(after_list_obj) || !PyArray_Check(gate_set_obj) || !PyArray_Check(dims_obj))
    {
        PyErr_SetString(PyExc_TypeError, "Inputs must be numpy arrays");
        return NULL;
    }

    // Get pointers to the data as C-types
    npy_intp *before_op_list = (npy_intp *)PyArray_DATA(before_list_obj);
    npy_intp *after_op_list = (npy_intp *)PyArray_DATA(after_list_obj);

    int N = (int)PyArray_DIM(before_list_obj, 0);

    _load_operators(&gate_set, gate_set_obj, dims_obj);

    return PyFloat_FromDouble(_qpt_mat_element(&gate_set, N, before_op_list, before_op_list, m, n, i));
}

static PyObject *pauli_element(PyObject *self, PyObject *args)
{
    PyArrayObject *gate_set_obj, *dims_obj, *op_list_obj;
    Operators gate_set;
    size_t N, r, c, op_list_len;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "nnnn", &op_list_len, &N, &r, &c))
    {
        return NULL;
    }

    double real, imag;

    _pauli_tensor_element(op_list_len, N, r, c, &real, &imag);

    return PyComplex_FromDoubles(real, imag);
}

typedef struct
{
    PyObject_HEAD size_t number_of_qubit;
    size_t r, c, count;
    size_t max_row, max_col;
    npy_intp *current_op_list;
    Operators gate_set;
} QSTMatrixGeneratorObject;

static PyObject *QSTMatrixGenerator_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    QSTMatrixGeneratorObject *self;
    self = (QSTMatrixGeneratorObject *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->number_of_qubit = 0;
        self->current_op_list = NULL;
        self->r = 0;
        self->c = 0;
        self->count = 0;
        self->max_row = 0;
        self->max_col = 0;
    }
    return (PyObject *)self;
}

static int QSTMatrixGenerator_init(QSTMatrixGeneratorObject *self, PyObject *args, PyObject *kwds)
{
    PyArrayObject *gate_set_obj, *dims_obj;
    Operators gate_set;
    size_t number_of_qubits;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "OOn", &gate_set_obj, &dims_obj, &number_of_qubits))
    {
        return -1;
    }

    // Ensure the inputs are numpy arrays
    if (!PyArray_Check(dims_obj) || !PyArray_Check(gate_set_obj))
    {
        PyErr_SetString(PyExc_TypeError, "Inputs must be numpy arrays");
        return -1;
    }

    // Get pointers to the data as C-types
    self->gate_set.data = (npy_complex128 *)PyArray_DATA(gate_set_obj);
    self->gate_set.dims = (npy_intp *)PyArray_DATA(dims_obj);
    self->gate_set.max_row = (size_t)PyArray_DIM(gate_set_obj, 1);
    self->gate_set.max_col = (size_t)PyArray_DIM(gate_set_obj, 2);
    self->gate_set.max_size = self->gate_set.max_row * self->gate_set.max_col;
    self->gate_set.length = (size_t)PyArray_DIM(gate_set_obj, 0);

    // self->current = 0;
    // self->stop = power(gate_set.length, number_of_qubits);
    self->number_of_qubit = number_of_qubits;
    self->current_op_list = (npy_intp *)calloc(number_of_qubits, sizeof(npy_intp));
    // if (next_product(self->current_op_list, self->number_of_qubit, self->gate_set.length))
    // {
    //     self->number_of_qubit = 0;
    // }
    self->r = 1;
    self->c = 1;
    self->count = 0;
    self->max_row = power(2, number_of_qubits);
    self->max_col = power(4, number_of_qubits);

    return 0;
}

static PyObject *QSTMatrixGenerator_iter(PyObject *self)
{
    Py_INCREF(self);
    return self;
}

static inline PyObject *_QSTMatrixGenerator_next_make_result(long r, long c, double v)
{
    PyObject *tuple = PyTuple_New(3);
    if (!tuple)
    {
        return NULL;
    }
    PyObject *i = PyLong_FromLong(r);
    if (!i)
    {
        Py_DECREF(tuple);
        return NULL;
    }
    PyTuple_SetItem(tuple, 0, i);
    PyObject *j = PyLong_FromLong(c);
    if (!j)
    {
        Py_DECREF(tuple);
        return NULL;
    }
    PyTuple_SetItem(tuple, 1, j);
    PyObject *value = PyFloat_FromDouble(v);
    if (!value)
    {
        Py_DECREF(tuple);
        return NULL;
    }
    PyTuple_SetItem(tuple, 2, value);

    return tuple;
}

static inline int next_product(npy_intp *ret, size_t repeat, size_t max)
{
    for (size_t i = repeat; i > 0; i--)
    {
        if (ret[i - 1] < max)
        {
            ret[i - 1]++;
            for (size_t j = i; j < repeat; j++)
            {
                ret[j] = 0;
            }
            return 0;
        }
    }
    return 1; // Indicate that we are done
}

static PyObject *QSTMatrixGenerator_next(QSTMatrixGeneratorObject *self)
{
    while (1)
    {
        if (self->number_of_qubit == 0)
        {
            PyErr_SetNone(PyExc_StopIteration);
            return NULL;
        }

        size_t N = (1 << self->number_of_qubit);
        size_t row = self->count / (self->max_col - 1);
        size_t col = self->count % (self->max_col - 1);

        double ret = _qst_mat_element(
            &self->gate_set,
            self->number_of_qubit,
            self->current_op_list,
            self->r, self->c);

        self->c++;
        if (self->c >= self->max_col)
        {
            self->c = 1;
            self->r++;
        }
        if (self->r >= self->max_row)
        {
            self->r = 1;
            if (next_product(self->current_op_list, self->number_of_qubit, self->gate_set.length - 1))
            {
                self->number_of_qubit = 0;
            }
        }
        self->count++;

        if (ret >= 1e-18 || ret <= -1e-18)
        {
            return _QSTMatrixGenerator_next_make_result(row, col, ret / N);
        }
    }
}

static void QSTMatrixGenerator_dealloc(QSTMatrixGeneratorObject *self)
{
    free(self->current_op_list);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyTypeObject QSTMatrixGeneratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "QSTMatrixGenerator",
    .tp_doc = "QST matrix element generator",
    .tp_basicsize = sizeof(QSTMatrixGeneratorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = QSTMatrixGenerator_new,
    .tp_init = (initproc)QSTMatrixGenerator_init,
    .tp_dealloc = (destructor)QSTMatrixGenerator_dealloc,
    .tp_iter = QSTMatrixGenerator_iter,
    .tp_iternext = (iternextfunc)QSTMatrixGenerator_next,
};

static PyMethodDef TensorMethods[] = {
    {"tensor_element", tensor_element, METH_VARARGS, "Compute the tensor element"},
    {"pauli_element", pauli_element, METH_VARARGS, "Compute the pauli element"},
    {"qst_mat_element", qst_mat_element, METH_VARARGS, "Compute the qst matrix element"},
    {"qpt_mat_element", qpt_mat_element, METH_VARARGS, "Compute the qpt matrix element"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef tensormodule = {
    PyModuleDef_HEAD_INIT,
    "_tensor",
    NULL,
    -1,
    TensorMethods};

PyMODINIT_FUNC PyInit__tensor(void)
{
    PyObject *m;

    import_array();

    if (PyType_Ready(&QSTMatrixGeneratorType) < 0)
        return NULL;

    m = PyModule_Create(&tensormodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&QSTMatrixGeneratorType);
    if (PyModule_AddObject(m, "QSTMatrixGenerator", (PyObject *)&QSTMatrixGeneratorType) < 0)
    {
        Py_DECREF(&QSTMatrixGeneratorType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
