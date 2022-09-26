#!/usr/bin/env python3

OP_ADD = "add"
OP_MUL = "mul"

# EXPRESSIONS

## for(i, e)
class ForNode:
    def __init__(self, index, extent, expr):
        self.index = index
        self.extent = extent
        self.expr = expr

    def __str__(self):
        return "for {} in {} { {} }".format(self.index, self.extent, self.expr)

## reduce(op, e)
class ReduceNode:
    def __init__(self, expr, op=OP_ADD):
        self.expr = expr
        self.op = op

    def __str__(self):
        reduce_str = "sum" if op == OP_ADD else "product"
        return "{}({})".format(reduce_str, self.expr)

## e op e
class OpNode:
    def __init__(self, expr1, expr2, op=OP_ADD):
        self.expr1 = expr1
        self.expr2 = expr2
        self.op = op

    def __str__(self):
        op_str = "+" if op == OP_ADD else "*"
        return "({} {} {})".format(self.expr1, op_str, self.expr2)


## x[i]
class IndexNode:
    def __init__(self, var, index_list):
        self.var = var
        self.index_list = index_list

    def __str__(self):
        if len(self.index_list) > 0:
            index_str = ["[{}]".format(ind) for ind in index_list].join("")
            return "{}{}".format(self.var, index_str)

        else:
            return self.var


# TRANSFORMATIONS
class FillNode:
    def __init__(self, arr, fill_sizes):
        self.arr = arr
        self.fill_sizes = fill_sizes

    def __str__(self):
        return "fill({}, {})".format(self.arr, self.fill_sizes)


class TransposeNode:
    def __init__(self, arr, perm):
        self.arr = arr
        self.perm = perm

    def __str__(self):
        return "transpose({}, {}".format(self.arr, self.perm)


def normalize(expr, path=[]):
    if isinstance(expr, ForNode):
        return normalize(expr.expr, path + [("index", (expr.index, expr.extent))])

    elif isinstance(expr, ReduceNode):
        new_expr = normalize(expr.expr, path + [("reduce", expr.op)])
        return ReduceNode(new_expr, expr.op)

    elif isinstance(expr, OpNode):
        new_expr1 = normalize(expr.expr1, path)
        new_expr2 = normalize(expr.expr1, path)
        return OpNode(new_expr1, new_expr2, expr.op)

    # TODO for now, assume index nodes are scalar (0-dim)
    elif isinstance(expr, IndexNode):
        # first, compute the required shape of the array
        orig_shape = expr.index_list[:]
        new_shape = []
        reduce_ind = 0

        for (tag, val) in path[::-1]:
            if tag == "index":
                new_shape.insert(reduce_ind, val)

            elif tag == "reduce":
                reduce_ind += 1

        # next, compute the transformations from the array's original shape
        # to its required shape
        # - first, compute fills
        fill_sizes = []
        if len(orig_shape) < len(new_shape):
            for (index, extent) in new_shape[len(orig_shape):]:
                fill_sizes.append(extent)

        # - second, compute transpositions
        transpose_perm = []
        for i in range(len(orig_shape)):
            transpose_perm[i] = orig_shape.index(new_shape[i])

        if len(fill_sizes) > 0:
            return TransposeNode(FillNode(IndexNode(expr.var, []), fill_sizes), transpose_perm)

        else:
            return TransposeNode(IndexNode(expr.var, []), transpose_perm)
    else:
        assert(False, "normalize: failed to match expression")

