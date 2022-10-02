#!/usr/bin/env python3

from typing import *
import numpy as np

OP_ADD = "add"
OP_SUB = "sub"
OP_MUL = "mul"

# INTERVALS

# Intervals with arithmetic and lattice operations.
# the empty interval is represented as Interval(None, None)
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

    def __eq__(self, other):
        if isinstance(other, Interval):
            return self.minval == other.minval and self.maxval == other.maxval

        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def is_empty(self):
        return self.minval is None and self.maxval is None

    def contains(self, other):
        if self.is_empty():
            return other.is_empty()

        elif other.is_empty():
            return self.is_empty()

        else:
            return self.minval <= other.minval and self.maxval >= other.maxval

    def merge(self, other):
        if self.is_empty():
            return other

        elif other.is_empty():
            return self

        else:
            return Interval(min(self.minval, other.minval), max(self.maxval, other.maxval))

    def __str__(self):
        return "({}, {})".format(self.minval, self.maxval)

    def __repr__(self):
        return "Interval({}, {})".format(self.minval, self.maxval)

EmptyInterval = Interval(None, None)

# EXPRESSIONS

class ExpressionNode:
    cur_id = 0

    @staticmethod
    def get_fresh_id():
        ret = ExpressionNode.cur_id
        ExpressionNode.cur_id += 1
        return ret

    def __init__(self):
        self.id = ExpressionNode.get_fresh_id()

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
        super().__init__()
        self.index = index
        self.extent = extent
        self.expr = expr

    def __str__(self):
        return "for {} in {} {{ {} }}".format(self.index, self.extent, self.expr)

## reduce(op, e)
class ReduceNode(ExpressionNode):
    def __init__(self, expr, op=OP_ADD):
        super().__init__()
        self.expr = expr
        self.op = op

    def __str__(self):
        reduce_str = "sum" if self.op == OP_ADD else "product"
        return "{}({})".format(reduce_str, self.expr)

## e op e
class OpNode(ExpressionNode):
    def __init__(self, expr1, expr2, op=OP_ADD):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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

# represent fill, transpose, and pad transformations
class TransformNode(ExpressionNode):
    def __init__(self, arr, fill_sizes=[], transpose=[], pad_sizes=[], extent_list=[]):
        super().__init__()
        self.arr = arr
        self.fill_sizes = fill_sizes
        self.transpose = transpose
        self.pad_sizes = pad_sizes
        self.extent_list = extent_list

    def __str__(self):
        expr = str(self.arr)

        if len(self.fill_sizes) > 0:
            expr = "fill({}, {})".format(expr, self.fill_sizes)

        if self.transpose != list(range(len(self.transpose))):
            expr = "transpose({}, {})".format(expr, self.transpose)

        if any(pad_min != 0 or pad_max != 0 for (pad_min, pad_max) in self.pad_sizes):
            expr = "pad({}, {})".format(expr, self.pad_sizes)

        return expr

class IntervalVar:
    def __init__(self, var_id):
        self.id = var_id

    def __repr__(self):
        return "IntervalVar({})".format(self.id)

class IntervalConstraint:
    pass

# constraint that rhs contains the lhs
# - lhs must be a constant and rhs must be a variable
class IntervalContainsConstraint(IntervalConstraint):
    def __init__(self, lhs, rhs):
        if isinstance(lhs, Interval) and isinstance(rhs, IntervalVar):
            self.lhs = lhs
            self.rhs = rhs
        
        else:
            assert False, "LHS of IntervalContainsConstraint must be a constant"

# constraint that rhs contains the lhs
# - lhs must be a constant and rhs must be a variable
class IntervalEqualsConstraint(IntervalConstraint):
    def __init__(self, lhs, rhs):
        if isinstance(lhs, IntervalVar) and isinstance(rhs, IntervalVar):
            self.lhs = lhs
            self.rhs = rhs
        
        else:
            assert False, "both LHS and RHS of IntervalEqualsConstraint must be variables"
        
# extent analysis
class ExtentAnalysis:
    def __init__(self):
        # fresh constraint var id counter
        self.cur_var_id = 0

        # constraint counters
        self.constraints = []

        # constraint variables
        self.vars = []

        # the expression nodes that need solutions
        self.nodes = {}

    def fresh_constraint_var(self):
        new_id = self.cur_var_id
        self.cur_var_id += 1
        var = IntervalVar(new_id)
        self.vars.append(var)
        return var

    def add_contains_constraint(self, lhs, rhs):
        self.constraints.append(IntervalContainsConstraint(lhs, rhs))

    def add_equals_constraint(self, lhs, rhs):
        self.constraints.append(IntervalEqualsConstraint(lhs, rhs))

    # generate constraints between extents in an expr
    def collect_constraints(self, expr):
        if isinstance(expr, ForNode):
            assert False, "unify_extents: input must be an index-free expression"

        elif isinstance(expr, IndexingNode):
            assert False, "unify_extents: input must be an index-free expression"

        elif isinstance(expr, ReduceNode):
            (i, extent_list) = self.collect_constraints(expr.expr)
            return (i+1, extent_list)

        elif isinstance(expr, OpNode):
            (i1, extent_list1) = self.collect_constraints(expr.expr1)
            (i2, extent_list2) = self.collect_constraints(expr.expr2)

            assert len(extent_list1[i1:]) == len(extent_list2[i2:])

            for (e1, e2) in zip(extent_list1[i1:], extent_list2[i2:]):
                self.add_equals_constraint(e1, e2)

            # pick one extent list from operands to return
            return (i1, extent_list1)

        elif isinstance(expr, TransformNode):
            extent_vars = []
            for extent in expr.extent_list:
                extent_var = self.fresh_constraint_var()
                extent_vars.append(extent_var)
                self.add_contains_constraint(extent, extent_var)

            self.nodes[expr.id] = extent_vars
            return (0, extent_vars)

    # adjust padding of TransformNodes so that they satisfy
    # the extent solutions computed
    def apply_solution(self, expr, node_solutions):
        if isinstance(expr, ForNode):
            assert False, "apply_solution: input must be an index-free expression"

        elif isinstance(expr, IndexingNode):
            assert False, "apply_solution: input must be an index-free expression"

        elif isinstance(expr, ReduceNode):
            new_expr = self.apply_solution(expr.expr, node_solutions)
            return ReduceNode(new_expr, op=expr.op)

        elif isinstance(expr, OpNode):
            new_expr1 = self.apply_solution(expr.expr1, node_solutions)
            new_expr2 = self.apply_solution(expr.expr2, node_solutions)

            return OpNode(new_expr1, new_expr2, op=expr.op)

        elif isinstance(expr, TransformNode):
            if expr.id in node_solutions:
                new_pad_sizes = []
                for pad, cur_extent, sol_extent in zip(expr.pad_sizes, expr.extent_list, node_solutions[expr.id]):
                    assert sol_extent.contains(cur_extent), "sol_extent should only ADD padding, not remove it"
                    if cur_extent != sol_extent:
                        new_pad_min = (cur_extent.minval - sol_extent.minval) + pad[0]
                        new_pad_max = (sol_extent.maxval - cur_extent.maxval) + pad[1]
                        new_pad_sizes.append((new_pad_min, new_pad_max))

                new_transform = \
                    TransformNode(
                        expr.arr, expr.fill_sizes, expr.transpose,
                        new_pad_sizes, node_solutions[expr.id]
                    )

                return new_transform

            else:
                return expr


    def run(self, expr):
        # first, collect constraints
        self.collect_constraints(expr)

        # next, find solutions
        # initial solution is all variables are EmptyInterval (bottom of lattice)
        solution = dict([(var.id, EmptyInterval) for var in self.vars])

        # next, find fixpoint solution to constraints
        # just implementing a simple linear pass instead of doing
        # the usual dataflow analysis optimizations like
        # keeping track of which constraints to wake when a solution is updated,
        # toposorting the connected components of the graph, etc.
        quiesce = False
        while not quiesce:
            quiesce = True
            for c in self.constraints:
                if isinstance(c, IntervalContainsConstraint):
                    # update RHS solution
                    rhs_sol = solution[c.rhs.id]
                    if not rhs_sol.contains(c.lhs):
                        solution[c.rhs.id] = rhs_sol.merge(c.lhs)
                        quiesce = False

                if isinstance(c, IntervalEqualsConstraint):
                    # update LHS and RHS solution
                    lhs_sol = solution[c.lhs.id]
                    rhs_sol = solution[c.rhs.id]
                    if not (rhs_sol.contains(lhs_sol) and lhs_sol.contains(rhs_sol)):
                        new_sol = rhs_sol.merge(lhs_sol)
                        solution[c.rhs.id] = new_sol
                        solution[c.lhs.id] = new_sol
                        quiesce = False

        node_solutions = {}
        for node, extent_vars in self.nodes.items():
            extent_sol = [solution[var.id] for var in extent_vars]
            node_solutions[node] = extent_sol

        # finally, apply solutions by transforming expr nodes
        new_expr = self.apply_solution(expr, node_solutions)
        return new_expr


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

class Normalizer:
    def __init__(self):
        self.extent_analysis = ExtentAnalysis()

    # convert into index-free representation
    def normalize(self, expr: ExpressionNode, store={}, path=[]): 
        if isinstance(expr, ForNode):
            return self.normalize(expr.expr, store, path + [("index", (expr.index, expr.extent))])

        elif isinstance(expr, ReduceNode):
            new_expr = self.normalize(expr.expr, store, path + [("reduce", expr.op)])
            return ReduceNode(new_expr, expr.op)

        elif isinstance(expr, OpNode):
            new_expr1 = self.normalize(expr.expr1, store, path)
            new_expr2 = self.normalize(expr.expr2, store, path)
            return OpNode(new_expr1, new_expr2, expr.op)

        elif isinstance(expr, VarNode):
            return expr

        # TODO for now, assume index nodes are scalar (0-dim)
        elif isinstance(expr, IndexingNode):
            # first, compute the required shape of the array
            required_shape = []
            reduce_ind = 0

            for (tag, val) in path[::-1]:
                if tag == "index":
                    required_shape.insert(reduce_ind, val)

                elif tag == "reduce":
                    reduce_ind += 1

            # next, compute the transformations from the array's original shape
            # to its required shape
            # compute the original shape
            orig_shape = []
            for ind_expr in expr.index_list:
                ind_vars = get_index_vars(ind_expr)
                if len(ind_vars) == 1:
                    orig_shape.append(ind_vars.pop())

                else:
                    assert False, "only one index var allowed per dimension"

            # generate map of in-scope indices and their extents
            ind_store = dict([data for (tag, data) in path if tag == "index"])

            # compute fills
            extent_list = []
            missing_indices = [(index, extent) for (index, extent) in required_shape if index not in orig_shape]
            new_shape = orig_shape[:]
            fill_sizes = []
            for (index, extent) in missing_indices:
                extent_list.append(Interval(extent[0], extent[1]))
                fill_sizes.append(extent[1] - extent[0] + 1)
                new_shape = [index] + new_shape

            # compute padding
            # initialize with padding for filled dimensions, which should always be (0,0)
            pad_sizes = [(0,0) for _ in range(len(fill_sizes))]
            for i, ind_expr in enumerate(expr.index_list):
                ind_interval = indexpr_to_interval(ind_expr, ind_store)
                dim_extent = store[expr.arr.var][i]
                dim_interval = Interval(dim_extent[0], dim_extent[1])

                pad_min, pad_max = 0, 0
                if dim_interval.minval > ind_interval.minval:
                    pad_min = dim_interval.minval - ind_interval.minval

                if dim_interval.maxval < ind_interval.maxval:
                    pad_max = ind_interval.maxval - dim_interval.maxval

                pad_sizes.append((pad_min, pad_max))
                extent_list.append(Interval(dim_extent[0]-pad_min, dim_extent[1]+pad_max))

            # compute transpositions
            transpose = list(range(len(required_shape)))
            for i in range(len(new_shape)):
                transpose[i] = new_shape.index(required_shape[i][0])

            # apply transpositions
            transposed_pad_sizes = [pad_sizes[i] for i in transpose]
            transposed_extent_list = [extent_list[i] for i in transpose]

            print("new_shape", new_shape)
            print("required_shape", required_shape)
            print("fill_sizes", fill_sizes)
            print("pad_list", transposed_pad_sizes)
            print("transpose", transpose)
            print("extent_list", transposed_extent_list)

            return TransformNode(expr.arr, fill_sizes, transpose, transposed_pad_sizes, transposed_extent_list)

        else:
            assert False, "normalize: failed to match expression"

    def run(self, expr, store):
        norm_expr = self.normalize(expr, store)
        norm_expr2 = self.extent_analysis.run(norm_expr)
        return norm_expr2

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
    norm_expr = Normalizer().run(expr, store)

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
    norm_expr = Normalizer().run(expr, store)

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
    norm_expr = Normalizer().run(expr, store)

    print(expr)
    print(norm_expr)

if __name__ == "__main__":
    matvecmul()
    matmatmul()
    imgblur()