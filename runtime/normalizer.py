#!/usr/bin/env python3

import numpy as np

OP_ADD = "add"
OP_MUL = "mul"

# EXPRESSIONS

class ExpressionNode:
    pass

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
        op_str = "+" if self.op == OP_ADD else "*"
        return "({} {} {})".format(self.expr1, op_str, self.expr2)


## x[i]
class IndexNode(ExpressionNode):
    def __init__(self, var, index_list):
        self.var = var
        self.index_list = index_list

    def __str__(self):
        if len(self.index_list) > 0:
            index_str = "".join(["[{}]".format(ind) for ind in self.index_list])
            return "{}{}".format(self.var, index_str)

        else:
            return self.var

class TransformNode:
    pass

# TRANSFORMATIONS
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

class ArrayNode:
    def __init__(self, arr):
        self.arr = arr

    def __str__(self):
        return str(self.arr)

# interpret a program against a given store
def interpret(expr, store):
    if isinstance(expr, ReduceNode):
        val = interpret(expr.expr, store)
        if expr.op == OP_ADD:
            return np.add.reduce(val, axis=0)

        elif expr.op == OP_MUL:
            return np.multiply.reduce(val, axis=0)

        else:
            assert False, "unknown operator {}".format(expr.op)

    elif isinstance(expr, OpNode):
        val1 = interpret(expr.expr1, store)
        val2 = interpret(expr.expr2, store)

        if expr.op == OP_ADD:
            return val1 + val2

        elif expr.op == OP_MUL:
            return val1 * val2

        else:
            assert False, "unknown operator {}".format(expr.op)

    elif isinstance(expr, FillNode):
        cur = interpret(expr.arr, store)
        for d in expr.fill_sizes:
            cur = np.stack([cur]*d)

        return cur

    elif isinstance(expr, TransposeNode):
        val = interpret(expr.arr, store)
        print(val)
        return val.transpose(tuple(expr.perm))

    elif isinstance(expr, ArrayNode):
        return store[expr.arr]

    else:
        assert False, "interpret: failed to match expression"

# convert into index-free representation
def normalize(expr: ExpressionNode, path=[]):
    if isinstance(expr, ForNode):
        return normalize(expr.expr, path + [("index", (expr.index, expr.extent))])

    elif isinstance(expr, ReduceNode):
        new_expr = normalize(expr.expr, path + [("reduce", expr.op)])
        return ReduceNode(new_expr, expr.op)

    elif isinstance(expr, OpNode):
        new_expr1 = normalize(expr.expr1, path)
        new_expr2 = normalize(expr.expr2, path)
        return OpNode(new_expr1, new_expr2, expr.op)

    # TODO for now, assume index nodes are scalar (0-dim)
    elif isinstance(expr, IndexNode):
        # first, compute the required shape of the array
        orig_shape = expr.index_list[:]
        required_shape = []
        reduce_ind = 0

        for (tag, val) in path[::-1]:
            if tag == "index":
                required_shape.insert(reduce_ind, val)

            elif tag == "reduce":
                reduce_ind += 1

        # next, compute the transformations from the array's original shape
        # to its required shape
        # - first, compute fills
        missing_indices = [(index, extent) for (index, extent) in required_shape if index not in orig_shape]
        new_shape = orig_shape[:]
        fill_sizes = []
        for (index, extent) in missing_indices:
            fill_sizes.append(extent)
            new_shape = [index] + new_shape

        print("new shape:", new_shape)
        print("required shape:", required_shape)

        # - second, compute transpositions
        transpose_perm = list(range(len(required_shape)))
        for i in range(len(new_shape)):
            transpose_perm[i] = new_shape.index(required_shape[i][0])

        if len(fill_sizes) > 0:
            return TransposeNode(FillNode(ArrayNode(expr.var), fill_sizes), transpose_perm)

        else:
            return TransposeNode(ArrayNode(expr.var), transpose_perm)
    else:
        assert False, "normalize: failed to match expression"

def matvecmul():
    # matrix-vector multiply
    expr = ForNode("i", 2, \
        ReduceNode( \
            ForNode("k", 2, \
                OpNode( \
                    IndexNode("A", ["i", "k"]), \
                    IndexNode("v", ["k"]), \
                    op = OP_MUL
                )
            )
        )
    )

    norm_expr = normalize(expr)

    store = {
        "A": np.array([1,2,3,4]).reshape((2,2)),
        "v": np.array([5,6])
    }

    out = interpret(norm_expr, store)

    print("source program:\n", expr)
    print("index-free representation:\n", norm_expr)

    print("input A:")
    print(store["A"])
    print("input v:")
    print(store["v"])
    print("output:")
    print(out)


# matrix-matrix multiply
def matmatmul():
    expr = ForNode("i", 2, ForNode("j", 2, \
        ReduceNode( \
            ForNode("k", 2, \
                OpNode( \
                    IndexNode("A", ["i", "k"]), \
                    IndexNode("B", ["k", "j"]), \
                    op = OP_MUL
                )
            )
        )
    ))

    norm_expr = normalize(expr)

    print("source program:\n", expr)
    print("index-free representation:\n", norm_expr)

    store = {
        "A": np.array([1,2,3,4]).reshape((2,2)),
        "B": np.array([5,6,7,8]).reshape((2,2))
    }

    out = interpret(norm_expr, store)

    print("input A:")
    print(store["A"])
    print("input B:")
    print(store["B"])
    print("output:")
    print(out)

if __name__ == "__main__":
    matvecmul()