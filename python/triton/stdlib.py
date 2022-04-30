#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Standard library
'''
import triton.code_gen
import triton.language.core as tlcore
tljit = triton.code_gen.jit
constexpr = tlcore.constexpr

# -----------------------
# Standard library
# -----------------------
__all__ = []
def export(func):
    __all__.append(func)
    return func

@export
@tljit
def abs(x):
    return where(x >= 0, x, -x)


@export
@tljit
def cdiv(x, div):
    """
    Computes the ceiling division of :code:`x` by :code:`div`

    :param x: the input number
    :type input: Block
    :param div: the divisor
    :param div: Block
    """
    return (x + div - 1) // div


@export
@tljit
def minimum(x, y):
    """
    Computes the element-wise minimum of :code:`x` and :code:`y`.

    :param input: the first input tensor
    :type input: Block
    :param other: the second input tensor
    :type other: Block
    """
    return tlcore.where(x < y, x, y)


@export
@tljit
def maximum(x, y):
    """
    Computes the element-wise maximum of :code:`x` and :code:`y`.

    :param input: the first input tensor
    :type input: Block
    :param other: the second input tensor
    :type other: Block
    """
    return tlcore.where(x > y, x, y)

def _add_math_1arg_docstr(name):

    def _decorator(func):
        docstr = """
    Computes the element-wise {name} of :code:`x`

    :param x: the input values
    :type x: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator

@export
@tljit
@_add_math_1arg_docstr("sigmoid")
def sigmoid(x):
    return 1 / (1 + tlcore.exp(-x))


@export
@tljit
@_add_math_1arg_docstr("softmax")
def softmax(x, ieee_rounding: constexpr = False):
    z = x - tlcore.max(x, 0)
    num = tlcore.exp(z)
    den = tlcore.sum(num, 0)
    return fdiv(num, den, ieee_rounding)


@export
@tljit
def ravel(x):
    """
    Returns a contiguous flattened view of :code:`x`

    :param x: the input tensor
    :type x: Block
    """
    return tlcore.reshape(x, [x.numel])


@export
@tljit
def swizzle2d(i, j, size_i, size_j, size_g):
    """
    transformes indices of a row-major size_i*size_j matrix into those
    of one where indices are row major for each group of size_j rows.
    For example, for size_i = size_j = 4 and size_g = 2, it will transform
    [[0 , 1 , 2 , 3 ],
     [4 , 5 , 6 , 7 ],
     [8 , 9 , 10, 11],
     [12, 13, 14, 15]]
    into
    [[0, 2,  4 , 6 ],
     [1, 3,  5 , 7 ],
     [8, 10, 12, 14],
     [9, 11, 13, 15]]
    """
    # "unrolled index in array"
    ij = i * size_j + j
    # number of elements in `size_g` groups
    # of `size_j` columns
    size_gj = size_g * size_j
    # index of the group in which (i,j) is
    group_id = ij // size_gj
    # row-index of the first element of this group
    off_i = group_id * size_g
    # last group may have fewer rows
    size_g = minimum(size_i - off_i, size_g)
    # new row and column indices
    new_i = off_i + (ij % size_g)
    new_j = (ij % size_gj) // size_g
    return new_i, new_j


@export
@tljit
def zeros_like(input):
    return tlcore.zeros(input.shape, input.dtype)
