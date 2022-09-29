#!/usr/bin/env python3

from typing import *
import numpy as np

OP_ADD = "add"
OP_SUB = "sub"
OP_MUL = "mul"

# INTERVALS

class Interval:
    def __init__(self, minval: int, maxval: int):
        self.minval = minval
        self.maxval = maxval

    def __add__(self, other):
        return Interval(self.minval + other.minval, self.maxval + other.maxval)

    def __sub__(self, other):
        return Interval(self.minval - other.maxval, self.maxval - other.minval)

    def __mul__(self, other):
        allpairs = [self.minval * other.minval, self.minval * other.maxval, self.maxval * other.minval, self.maxval * other.maxval]
        return Interval(min(allpairs), max(allpairs))

    def contains(self, other):
        return self.minval <= other.minval and self.maxval >= other.maxval

# EXPRESSIONS

class ExpressionNode:
    def __add__(self, other):
        return OpNode(self, other, op=OP_ADD)

    def __sub__(self, other):
        return OpNode(self, other, op=OP_SUB)

    def __mul__(self, other):
        return OpNode(self, other, op=OP_MUL)

    def __getitem__(self, item):
        return IndexingNode(self, item)

## for(i, e)
class ForNode(ExpressionNode):
    def __init__(self, index, extent, expr):
        self.index = index
        self.extent = extent
        self.expr = expr

    def __str__(self):
        return "for {} in {} {{ {} }}".format(self.index, self.extent, self.expr)

## reduce(op, e)
class ReduceNode(ExpressionNode):
    def __init__(self, expr, op=OP_ADD):
        self.expr = expr
        self.op = op

    def __str__(self):
        reduce_str = "sum" if self.op == OP_ADD else "product"
        return "{}({})".format(reduce_str, self.expr)

## e op e
class OpNode(ExpressionNode):
    def __init__(self, expr1, expr2, op=OP_ADD):
        self.expr1 = expr1
        self.expr2 = expr2
        self.op = op

    def __str__(self):
        op_str = None
        if self.op == OP_ADD:
            op_str = "+"
        elif self.op == OP_SUB:
            op_str = "-"
        elif self.op == OP_MUL:
            op_str = "*"
        else:
            assert False, "unknown operator {}".format(self.op)

        return "({} {} {})".format(self.expr1, op_str, self.expr2)

## x[i]
class IndexingNode(ExpressionNode):
    def __init__(self, arr, index_list):
        self.arr = arr
        self.index_list = index_list

    def __str__(self):
        if len(self.index_list) > 0:
            index_str = "".join(["[{}]".format(ind) for ind in self.index_list])
            return "{}{}".format(self.arr, index_str)

        else:
            return self.arr

class VarNode(ExpressionNode):
    def __init__(self, var: str):
        self.var = var

    def __str__(self):
        return str(self.var)

# INDEX EXPRESSIONS

class IndexExpression:
    def __add__(self, other):
        if isinstance(other, int):
            return IndexOpNode(self, IndexLiteralNode(other), op=OP_ADD)

        else:
            return IndexOpNode(self, other, op=OP_ADD)

    def __sub__(self, other):
        if isinstance(other, int):
            return IndexOpNode(self, IndexLiteralNode(other), op=OP_SUB)

        else:
            return IndexOpNode(self, other, op=OP_SUB)

    def __mul__(self, other):
        if isinstance(other, int):
            return IndexOpNode(self, IndexLiteralNode(other), op=OP_MUL)

        else:
            return IndexOpNode(self, other, op=OP_MUL)

class IndexVarNode(IndexExpression):
    def __init__(self, var):
        self.var = var

    def __str__(self):
        return self.var

class IndexLiteralNode(IndexExpression):
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return str(self.val)

class IndexOpNode(IndexExpression):
    def __init__(self, expr1, expr2, op):
        self.expr1 = expr1
        self.expr2 = expr2
        self.op = op

    def __str__(self):
        op_str = None
        if self.op == OP_ADD:
            op_str = "+"
        elif self.op == OP_SUB:
            op_str = "-"
        elif self.op == OP_MUL:
            op_str = "*"
        else:
            assert False, "unknown operator {}".format(self.op)

        return "({} {} {})".format(self.expr1, op_str, self.expr2)


# TRANSFORMATIONS

class TransformNode:
    pass

class FillNode(TransformNode):
    def __init__(self, arr, fill_sizes):
        self.arr = arr
        self.fill_sizes = fill_sizes

    def __str__(self):
        return "fill({}, {})".format(self.arr, self.fill_sizes)

class TransposeNode(TransformNode):
    def __init__(self, arr, perm):
        self.arr = arr
        self.perm = perm

    def __str__(self):
        return "transpose({}, {})".format(self.arr, self.perm)

class PadNode(TransformNode):
    def __init__(self, arr, pad_list):
        self.arr = arr
        self.pad_list = pad_list

    def __str__(self):
        return "pad({}, {})".format(self.arr, self.pad_list)

# TODO support other indexing expressions like operators and constants
def interpret_index(index, ind_store):
    return ind_store[index]

# interpret a program against a given store
def interpret(expr, store, ind_store={}):
    if isinstance(expr, ForNode):
        arrs = []
        for i in range(expr.extent):
            new_ind_store = ind_store.copy()
            new_ind_store[expr.index] = i
            arrs.append(interpret(expr.expr, store, new_ind_store))

        return np.stack(arrs)

    elif isinstance(expr, IndexingNode):
        arr_val = interpret(expr.arr, store, ind_store)
        ind_list_vals = [interpret_index(ind, ind_store) for ind in expr.index_list]
        return arr_val[tuple(ind_list_vals)]

    elif isinstance(expr, ReduceNode):
        val = interpret(expr.expr, store, ind_store)
        if expr.op == OP_ADD:
            return np.add.reduce(val, axis=0)

        elif expr.op == OP_MUL:
            return np.multiply.reduce(val, axis=0)

        else:
            assert False, "unknown operator {}".format(expr.op)

    elif isinstance(expr, OpNode):
        val1 = interpret(expr.expr1, store, ind_store)
        val2 = interpret(expr.expr2, store, ind_store)

        if expr.op == OP_ADD:
            return val1 + val2

        elif expr.op == OP_MUL:
            return val1 * val2

        else:
            assert False, "unknown operator {}".format(expr.op)

    elif isinstance(expr, FillNode):
        cur = interpret(expr.arr, store, ind_store)
        for d in expr.fill_sizes:
            cur = np.stack([cur]*d)

        return cur

    elif isinstance(expr, TransposeNode):
        val = interpret(expr.arr, store, ind_store)
        return val.transpose(tuple(expr.perm))

    elif isinstance(expr, VarNode):
        return store[expr.var]

    else:
        assert False, "interpret: failed to match expression"


# HELPERS

def get_index_vars(ind_expr):
    if isinstance(ind_expr, IndexVarNode):
        return set([ind_expr.var])

    elif isinstance(ind_expr, IndexLiteralNode):
        return set()

    elif isinstance(ind_expr, IndexOpNode):
        return get_index_vars(ind_expr.expr1).union(get_index_vars(ind_expr.expr2))

    else:
        assert False, "get_indices: {} failed to match on index expression forms".format(ind_expr)

def indexpr_to_interval(ind_expr, ind_store):
    if isinstance(ind_expr, IndexVarNode):
        extent = ind_store[ind_expr.var]
        return Interval(extent[0], extent[1])

    elif isinstance(ind_expr, IndexLiteralNode):
        return Interval(ind_expr.val, ind_expr.val)

    elif isinstance(ind_expr, IndexOpNode):
        i1 = indexpr_to_interval(ind_expr.expr1, ind_store)
        i2 = indexpr_to_interval(ind_expr.expr2, ind_store)

        if ind_expr.op == OP_ADD:
            return i1 + i2

        elif ind_expr.op == OP_SUB:
            return i1 - i2

        elif ind_expr.op == OP_MUL:
            return i1 * i2

# convert into index-free representation
def normalize(expr: ExpressionNode, store={}, path=[]): 
    if isinstance(expr, ForNode):
        return normalize(expr.expr, store, path + [("index", (expr.index, expr.extent))])

    elif isinstance(expr, ReduceNode):
        new_expr = normalize(expr.expr, store, path + [("reduce", expr.op)])
        return ReduceNode(new_expr, expr.op)

    elif isinstance(expr, OpNode):
        new_expr1 = normalize(expr.expr1, store, path)
        new_expr2 = normalize(expr.expr2, store, path)
        return OpNode(new_expr1, new_expr2, expr.op)

    elif isinstance(expr, VarNode):
        return expr

    # TODO for now, assume index nodes are scalar (0-dim)
    elif isinstance(expr, IndexingNode):
        # first, compute the required shape of the array
        orig_shape = []
        for ind_expr in expr.index_list:
            ind_vars = get_index_vars(ind_expr)
            if len(ind_vars) == 1:
                orig_shape.append(ind_vars.pop())

            else:
                assert False, "only one index var allowed per dimension"

        required_shape = []
        reduce_ind = 0

        for (tag, val) in path[::-1]:
            if tag == "index":
                required_shape.insert(reduce_ind, val)

            elif tag == "reduce":
                reduce_ind += 1

        # next, compute the transformations from the array's original shape
        # to its required shape
        # - first, generate map of in-scope indices and their extents
        ind_store = dict([data for (tag, data) in path if tag == "index"])

        # - next, compute padding
        pad_list = []
        for i, ind_expr in enumerate(expr.index_list):
            ind_interval = indexpr_to_interval(ind_expr, ind_store)
            dim_extent = store[expr.arr.var][i]
            dim_interval = Interval(dim_extent[0], dim_extent[1])

            pad_min, pad_max = 0, 0
            if dim_interval.minval > ind_interval.minval:
                pad_min = dim_interval.minval - ind_interval.minval

            if dim_interval.maxval < ind_interval.maxval:
                pad_max = ind_interval.maxval - dim_interval.maxval

            pad_list.append((pad_min, pad_max))

        print("pad_list", pad_list)

        # - next, compute fills
        missing_indices = [(index, extent) for (index, extent) in required_shape if index not in orig_shape]
        new_shape = orig_shape[:]
        fill_sizes = []
        for (index, extent) in missing_indices:
            fill_sizes.append(extent[1] - extent[0] + 1)
            new_shape = [index] + new_shape

        print("fill_sizes", fill_sizes)
        print("new_shape", new_shape)
        print("required_shape", required_shape)

        # - finally, compute transpositions
        transpose_perm = list(range(len(required_shape)))
        for i in range(len(new_shape)):
            transpose_perm[i] = new_shape.index(required_shape[i][0])

        # compose transformations in this order: pad, fill, transpose
        pad_expr = expr.arr
        if any(pad_min != 0 or pad_max != 0 for (pad_min, pad_max) in pad_list):
            pad_expr = PadNode(expr.arr, pad_list)

        fill_expr = pad_expr
        if len(fill_sizes) > 0:
            fill_expr = FillNode(expr.arr, fill_sizes)

        return TransposeNode(fill_expr, transpose_perm)

    else:
        assert False, "normalize: failed to match expression"

# check equivalence of two expressions by repeatedly interpreting them against different stores
def check_expr_equiv(expr1, expr2, store_template, n=20):
    for _ in range(n):
        store = {}
        for k, extents in store_template.items():
            shape = tuple(map(lambda extent: extent[1]-extent[0] + 1, extents))
            store[k] = np.random.randint(0, 100, shape)

        out1 = interpret(expr1, store)
        out2 = interpret(expr2, store)

        assert np.all(out1 == out2), "exprs not equal given store {}".format(expr1, expr2, store)

    print("these expressions are equivalent across {} random stores".format(n))
    print(expr1)
    print(expr2)

def ind(var):
    return IndexVarNode(var)

def arr(v):
    return VarNode(v)

def arrsum(expr):
    return ReduceNode(expr, op = OP_ADD)

def arrprod(expr):
    return ReduceNode(expr, op = OP_MUL)

def matvecmul():
    print("TEST: matrix-vector multiply")

    expr = ForNode("i", (0,1),
        arrsum(ForNode("k", (0,1),
            arr("A")[[ind("i"), ind("k")]] * arr("v")[[ind("k")]])
        )
    )

    store = {"A": [(0,1),(0,1)], "v": [(0,1)]}
    norm_expr = normalize(expr, store)

    print(expr)
    print(norm_expr)
    # check_expr_equiv(expr, norm_expr, store)

def matmatmul():
    print("TEST: matrix-matrix multiply")

    expr = ForNode("i", (0,1), ForNode("j", (0,1),
        arrsum(
            ForNode("k", (0,1),
                arr("A")[[ind("i"), ind("k")]] * arr("B")[[ind("k"), ind("j")]]
            )
        )
    ))

    store = {"A": [(0,1),(0,1)], "B": [(0,1),(0,1)]}
    norm_expr = normalize(expr, store)

    print(expr)
    print(norm_expr)
    # check_expr_equiv(expr, norm_expr, store)

    # print("source program:\n", expr)
    # print("index-free representation:\n", norm_expr)

    # store = {
    #     "A": np.array([1,2,3,4]).reshape((2,2)),
    #     "B": np.array([5,6,7,8]).reshape((2,2))
    # }

    # orig_out = interpret(expr, store)
    # norm_out = interpret(norm_expr, store)

    # print("input A:")
    # print(store["A"])
    # print("input B:")
    # print(store["B"])
    # print("output:")
    # print("output of original expr:")
    # print(orig_out)
    # print("output of normalized expr:")
    # print(norm_out)

def imgblur():
    print("TEST: image blur")

    expr = ForNode("x", (0,16), ForNode("y", (0,16),
        arr("img")[[ind("x") - 1, ind("y") - 1]] +
        arr("img")[[ind("x") + 1, ind("y") + 1]]
    ))

    store = {"img": [(0, 16),(0,16)]}
    norm_expr = normalize(expr, store)

    print(expr)
    print(norm_expr)

if __name__ == "__main__":
    matvecmul()
    matmatmul()
    imgblur()