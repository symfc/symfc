/* Copyright (C) 2023 Atsushi Togo */
/* All rights reserved. */

/* This file is part of phonopy. */

/* Redistribution and use in source and binary forms, with or without */
/* modification, are permitted provided that the following conditions */
/* are met: */

/* * Redistributions of source code must retain the above copyright */
/*   notice, this list of conditions and the following disclaimer. */

/* * Redistributions in binary form must reproduce the above copyright */
/*   notice, this list of conditions and the following disclaimer in */
/*   the documentation and/or other materials provided with the */
/*   distribution. */

/* * Neither the name of the phonopy project nor the names of its */
/*   contributors may be used to endorse or promote products derived */
/*   from this software without specific prior written permission. */

/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS */
/* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT */
/* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS */
/* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE */
/* COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, */
/* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, */
/* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; */
/* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER */
/* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT */
/* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN */
/* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE */
/* POSSIBILITY OF SUCH DAMAGE. */

#include <Python.h>
#include <float.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <stddef.h>
#include <stdio.h>

/* Build dynamical matrix */
static PyObject* py_kron_nn33_long(PyObject* self, PyObject* args);
static PyObject* py_kron_nn33_int(PyObject* self, PyObject* args);
static PyObject* py_projector_sum_rule_long(PyObject* self, PyObject* args);
static PyObject* py_projector_sum_rule_int(PyObject* self, PyObject* args);
struct module_state {
    PyObject* error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static PyObject* error_out(PyObject* m) {
    struct module_state* st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}

static PyMethodDef _symfc_methods[] = {
    {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
    {"kron_nn33_long", py_kron_nn33_long, METH_VARARGS,
     "Compute kron and transform n3n3 indices to nn33 indices."},
    {"kron_nn33_int", py_kron_nn33_int, METH_VARARGS,
     "Compute kron and transform n3n3 indices to nn33 indices."},
    {"projector_sum_rule_long", py_projector_sum_rule_long, METH_VARARGS,
     "Set row and column indices for projector_sum_rule."},
    {"projector_sum_rule_int", py_projector_sum_rule_int, METH_VARARGS,
     "Set row and column indices for projector_sum_rule."},
    {NULL, NULL, 0, NULL}};

static int _symfc_traverse(PyObject* m, visitproc visit, void* arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int _symfc_clear(PyObject* m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT,
                                       "_symfc",
                                       NULL,
                                       sizeof(struct module_state),
                                       _symfc_methods,
                                       NULL,
                                       _symfc_traverse,
                                       _symfc_clear,
                                       NULL};

#define INITERROR return NULL

PyObject* PyInit__symfc(void) {
    PyObject* module = PyModule_Create(&moduledef);
    struct module_state* st;
    if (module == NULL) INITERROR;
    st = GETSTATE(module);

    st->error = PyErr_NewException("_symfc.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }
    return module;
}

static PyObject* py_kron_nn33_int(PyObject* self, PyObject* args) {
    PyArrayObject* py_row;
    PyArrayObject* py_col;
    PyArrayObject* py_data;
    PyArrayObject* py_row_R;
    PyArrayObject* py_col_R;
    PyArrayObject* py_data_R;
    int* row_R;
    int* col_R;
    double* data_R;
    int* row;
    int* col;
    double* data;
    long size_3n;

    long size_R, count, i, j;
    long i_row_R, j_row_R, k_row_R, l_row_R, i_col_R, j_col_R, k_col_R, l_col_R;

    if (!PyArg_ParseTuple(args, "OOOOOOl", &py_row, &py_col, &py_data,
                          &py_row_R, &py_col_R, &py_data_R, &size_3n)) {
        return NULL;
    }

    size_R = PyArray_DIMS(py_row_R)[0];
    row_R = (int*)PyArray_DATA(py_row_R);
    col_R = (int*)PyArray_DATA(py_col_R);
    data_R = (double*)PyArray_DATA(py_data_R);
    row = (int*)PyArray_DATA(py_row);
    col = (int*)PyArray_DATA(py_col);
    data = (double*)PyArray_DATA(py_data);

    count = 0;
    for (i = 0; i < size_R; i++) {
        i_row_R = row_R[i] / 3;
        j_row_R = row_R[i] % 3;
        k_row_R = col_R[i] / 3;
        l_row_R = col_R[i] % 3;
        for (j = 0; j < size_R; j++) {
            i_col_R = row_R[j] / 3;
            j_col_R = row_R[j] % 3;
            k_col_R = col_R[j] / 3;
            l_col_R = col_R[j] % 3;
            row[count] =
                i_row_R * 3 * size_3n + i_col_R * 9 + j_row_R * 3 + j_col_R;
            col[count] =
                k_row_R * 3 * size_3n + k_col_R * 9 + l_row_R * 3 + l_col_R;
            data[count] = data_R[i] * data_R[j];
            count++;
        }
    }
    Py_RETURN_NONE;
}

static PyObject* py_kron_nn33_long(PyObject* self, PyObject* args) {
    PyArrayObject* py_row;
    PyArrayObject* py_col;
    PyArrayObject* py_data;
    PyArrayObject* py_row_R;
    PyArrayObject* py_col_R;
    PyArrayObject* py_data_R;
    long* row_R;
    long* col_R;
    double* data_R;
    long* row;
    long* col;
    double* data;
    long size_3n;

    long size_R, count, i, j;
    long i_row_R, j_row_R, k_row_R, l_row_R, i_col_R, j_col_R, k_col_R, l_col_R;

    if (!PyArg_ParseTuple(args, "OOOOOOl", &py_row, &py_col, &py_data,
                          &py_row_R, &py_col_R, &py_data_R, &size_3n)) {
        return NULL;
    }

    size_R = PyArray_DIMS(py_row_R)[0];
    row_R = (long*)PyArray_DATA(py_row_R);
    col_R = (long*)PyArray_DATA(py_col_R);
    data_R = (double*)PyArray_DATA(py_data_R);
    row = (long*)PyArray_DATA(py_row);
    col = (long*)PyArray_DATA(py_col);
    data = (double*)PyArray_DATA(py_data);

    count = 0;
    for (i = 0; i < size_R; i++) {
        i_row_R = row_R[i] / 3;
        j_row_R = row_R[i] % 3;
        k_row_R = col_R[i] / 3;
        l_row_R = col_R[i] % 3;
        for (j = 0; j < size_R; j++) {
            i_col_R = row_R[j] / 3;
            j_col_R = row_R[j] % 3;
            k_col_R = col_R[j] / 3;
            l_col_R = col_R[j] % 3;
            row[count] =
                i_row_R * 3 * size_3n + i_col_R * 9 + j_row_R * 3 + j_col_R;
            col[count] =
                k_row_R * 3 * size_3n + k_col_R * 9 + l_row_R * 3 + l_col_R;
            data[count] = data_R[i] * data_R[j];
            count++;
        }
    }
    Py_RETURN_NONE;
}

static PyObject* py_projector_sum_rule_long(PyObject* self, PyObject* args) {
    PyArrayObject* py_row;
    PyArrayObject* py_col;
    PyArrayObject* py_row_in;
    PyArrayObject* py_col_in;
    long natom;

    if (!PyArg_ParseTuple(args, "OOOOl", &py_row, &py_col, &py_row_in,
                          &py_col_in, &natom)) {
        return NULL;
    }

    long size = PyArray_DIMS(py_row_in)[0];
    long* row = (long*)PyArray_DATA(py_row);
    long* col = (long*)PyArray_DATA(py_col);
    long* row_in = (long*)PyArray_DATA(py_row_in);
    long* col_in = (long*)PyArray_DATA(py_col_in);

    long count = 0;
    for (int i = 0; i < natom; i++) {
        long shift = 9 * natom * i;
        for (int j = 0; j < size; j++) {
            row[count] = row_in[j] + shift;
            col[count] = col_in[j] + shift;
            count++;
        }
    }
    Py_RETURN_NONE;
}

static PyObject* py_projector_sum_rule_int(PyObject* self, PyObject* args) {
    PyArrayObject* py_row;
    PyArrayObject* py_col;
    PyArrayObject* py_row_in;
    PyArrayObject* py_col_in;
    long natom;

    if (!PyArg_ParseTuple(args, "OOOOl", &py_row, &py_col, &py_row_in,
                          &py_col_in, &natom)) {
        return NULL;
    }

    long size = PyArray_DIMS(py_row_in)[0];
    int* row = (int*)PyArray_DATA(py_row);
    int* col = (int*)PyArray_DATA(py_col);
    int* row_in = (int*)PyArray_DATA(py_row_in);
    int* col_in = (int*)PyArray_DATA(py_col_in);

    long count = 0;
    for (int i = 0; i < natom; i++) {
        long shift = 9 * natom * i;
        for (int j = 0; j < size; j++) {
            row[count] = row_in[j] + shift;
            col[count] = col_in[j] + shift;
            count++;
        }
    }
    Py_RETURN_NONE;
}
