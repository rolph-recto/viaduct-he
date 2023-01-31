#!/usr/bin/env python

# TODO:
# - add masks for derived vectors (DONE)
# - add support for fill dims (DONE)
# - add support for inplace reductions

import itertools
import numpy as np

BASE_OFFSET = "__BASE__"

class Shape:
    def __init__(self, exploded, vectorized):
        self.exploded = exploded
        self.vectorized = vectorized

    def __eq__(self, other):
        return self.exploded == other.exploded and self.vectorized == other.vectorized

    def __repr__(self):
        return "Shape({},{})".format(self.exploded, self.vectorized)

class Dim:
    def __init__(self, array, d, index, extent, stride=1, offset=0, pad_left=0, pad_right=0):
        self.array = array
        self.d = d
        self.index = index
        self.extent = extent
        self.stride = stride
        self.offset = offset
        self.pad_left = pad_left
        self.pad_right = pad_right

    def shape(self):
        return (self.extent, self.index)

    # clip the dimension according to a set size
    def clip(self, dim_size):
        # if offset is less than 0, then find the first element in bounds
        new_pad_left, new_pad_right = self.pad_left, self.pad_right
        new_extent = self.extent
        new_offset = self.offset

        if self.offset < 0:
            start = 0
            while start < self.extent:
                if (start * self.stride) + self.offset >= 0:
                    break

                start += 1

            new_pad_left += start
            new_extent -= start
            new_offset = (start * self.stride) + self.offset

        if (self.extent-1)*self.stride + self.offset >= dim_size:
            end = self.extent-1
            while (end * self.stride) + self.offset >= dim_size:
                end -= 1

            end_delta = (self.extent - 1) - end
            new_pad_right += end_delta
            new_extent -= end_delta

        new_extent = 0 if new_extent < 0 else new_extent
        new_dim = Dim(self.array, self.d, self.index, new_extent, self.stride, new_offset, new_pad_left, new_pad_right)
        return new_dim

    # derive `other` dim from `self` dim
    # returns None if not possible, otherwise
    # returns the rotation step needed for the derivation
    # TODO: generalize this properly
    def derive(self, other):
        if isinstance(other, Dim):
            # first, check if derivation is possible
            same_array_stride = \
                self.array == other.array and \
                self.d == other.d and self.stride == other.stride

            correct_offset = \
                self.offset <= other.offset and \
                self.offset % self.stride == other.offset % self.stride

            within_extent = self.extent >= other.extent

            # this seems an unnecessary constraint?
            no_padding = self.pad_left == 0 and self.pad_right == 0

            derivable = same_array_stride and correct_offset and within_extent and no_padding

            if derivable:
                # rotate right
                if other.pad_left > 0 and other.pad_right == 0 \
                    and other.offset == self.offset:
                    defined_interval = (other.pad_left, other.pad_left+other.extent)
                    return (other.pad_left, defined_interval)

                # rotate left
                elif other.offset > self.offset and other.pad_left == 0:
                    rot_steps = int((other.offset - self.offset) / other.stride)
                    if rot_steps == other.pad_right:
                        defined_interval = (0, other.extent)
                        return (-rot_steps, defined_interval)

                # other is the same dimension as self
                elif other.pad_left == 0 and other.pad_right == 0 and self.offset == other.offset:
                    return 0, (0, other.extent)

            # not derivable
            return None, None

        else:
            return None, None

    def __repr__(self):
        if self.pad_left == 0 and self.pad_right == 0:
            return "{}.{}[x:{}, {}x + {}]".format(
                    self.array, self.d, self.extent, self.stride, self.offset)

        else:
            return "{}.{}[x:{}, {}x + {};padl={};padr={}]".format(
                    self.array, self.d, self.extent, self.stride, self.offset,
                    self.pad_left, self.pad_right)

# fill dimension that only repeats elements of inner dimensions
# kind of a dual to a fixed dim
class FillDim:
    def __init__(self, index, extent):
        self.extent = extent

    def shape(self):
        return (self.extent, self.index)

    def derive(self, other):
        if isinstance(other, FillDim):
            if other.extent == self.extent:
                return 0, (0, other.extent)

            else:
                return None, None

        else:
            return None, None

    def __repr__(self):
        return "[{}]".format(self.extent)

class ArrayDim:
    def __init__(self, array, index, extent):
        self.array = array
        self.index = index
        self.extent = extent

    def __repr__(self):
        return "{}.{}:{}".format(self.array, self.index, self.extent)

class EmptyDim:
    def __init__(self, extent):
        self.extent = extent

class ScheduleDim:
    def __init__(self, index, extent, stride=1, offset=0):
        self.index = index
        self.extent = extent
        self.stride = stride
        self.offset = offset

    def __repr__(self):
        return "[x:{};{}x + {}]".format(self.extent, self.stride, self.offset)

class ArrayDimInfo:
    def __init__(self, array, d):
        self.array = array
        self.d = d

    def __repr__(self):
        return "{}.{}".format(self.array, self.d)

class ArrayDimTraversal:
    def __init__(self, array, d, stride, offset):
        self.array = array
        self.d = d
        self.stride = stride
        self.offset = offset

    def __repr__(self):
        return "{}.{}[{}x + {}]".format(self.array, self.d, self.stride, self.offset)

# fixed dimensions stay constant within a vector
class FixedDim:
    def __init__(self, array, d, offset):
        self.array = array
        self.d = d
        self.offset = offset

    def __repr__(self):
        return "{}.{}[{}]".format(self.array, self.d, self.offset)

class ExplodedDimInfo:
    def __init__(self, array_dim, index, index_var, extent):
        self.array_dim = array_dim
        self.index = index
        self.index_var = index_var
        self.extent = extent

    def get_offset_expr(self):
        stride = self.array_dim.stride
        offset = self.array_dim.offset
        lhs = OffsetMul(OffsetLit(stride), self.index_var) if stride != 1 else self.index_var
        if offset != 0:
            return OffsetAdd(lhs, OffsetLit(offset))
             
        else:
            return lhs

    def __repr__(self):
        return "{}[ind={}][var={}][extent={}]".format(self.array_dim, self.index, self.index_var, self.extent)

class IndexVar:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "{}".format(self.name)

class Reduce:
    def __init__(self, index, expr):
        self.index = index
        self.expr = expr

    def __repr__(self):
        return "Reduce({})".format(self.expr)


class OffsetLit:
    def __init__(self, num):
        self.num = num

    def __repr__(self):
        return "{}".format(self.num)

    def __eq__(self, other):
        return isinstance(other, OffsetLit) and self.num == other.num

class OffsetAdd:
    def __init__(self, expr1, expr2):
        self.expr1 = expr1
        self.expr2 = expr2

    def __repr__(self):
        return "({} + {})".format(self.expr1, self.expr2)

class OffsetMul:
    def __init__(self, expr1, expr2):
        self.expr1 = expr1
        self.expr2 = expr2

    def __repr__(self):
        return "({} * {})".format(self.expr1, self.expr2)

class Op:
    def __init__(self, expr1, expr2):
        self.expr1 = expr1
        self.expr2 = expr2

    def __repr__(self):
        return "Op({}, {})".format(self.expr1, self.expr2)

class OpList:
    def __init__(self, exprs):
        self.exprs = exprs

    def __repr__(self):
        return "OpList({})".format(self.exprs)

class Rotate:
    def __init__(self, steps, expr):
        self.steps = steps
        self.expr = expr

    def __repr__(self):
        return "Rot({}, {})".format(self.steps, self.expr)

class Mask:
    def __init__(self, defined_interval):
        self.defined_interval = defined_interval

    def __repr__(self):
        return "Mask({})".format(self.defined_interval)

class CiphertextVar:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

class PlaintextVar:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

class Vector:
    def __init__(self, dims, fixed_dims=[]):
        self.dims = dims
        self.fixed_dims = fixed_dims

    # derive one vector from another
    def derive(self, other):
        self_fixed_dim_set = set()
        for fd in self.fixed_dims:
            self_fixed_dim_set.add((fd.array, fd.d, fd.offset))

        other_fixed_dim_set = set()
        for fd in other.fixed_dims:
            other_fixed_dim_set.add((fd.array, fd.d, fd.offset))

        same_fixed_dim = self_fixed_dim_set == other_fixed_dim_set

        if same_fixed_dim and len(self.dims) == len(other.dims):
            block_size = 1
            total_steps = 0
            defined_interval = []
            for i in reversed(range(len(self.dims))):
                steps, dim_defined_interval = self.dims[i].derive(other.dims[i])
                if steps is not None:
                    total_steps += (block_size * steps)
                    block_size *= self.dims[i].extent
                    defined_interval.insert(0, dim_defined_interval)
                
                else:
                    return None, None

            return total_steps, tuple(defined_interval)

        else:
            return None, None

    def __repr__(self):
        fixed_dim_str = "{{{}}}".format(",".join(map(str, self.fixed_dims))) if len(self.fixed_dims) > 0 else ""
        dim_str = "<{}>".format(",".join(map(str, self.dims)))
        return fixed_dim_str + dim_str

# the schedule must span the entirety of the array
# this means that ALL array dimensions must be present
class Schedule:
    def __init__(self, exploded_dims=[], vectorized_dims=[]):
        self.exploded_dims = exploded_dims
        self.vectorized_dims = vectorized_dims

    def materialize_recur(self, array_shapes, dim_map, exploded_dims):
        # process vectorized dim
        if len(exploded_dims) == 0:
            vector_dims = []
            vec_array = list(dim_map.keys())[0][0]
            seen_dims = set()
            for vectorized_dim in self.vectorized_dims:
                if isinstance(vectorized_dim, Dim):
                    if vec_array != vectorized_dim.array:
                        raise Exception("vector cannot have dimensions from two different arrays")

                    dim = Dim(
                        vectorized_dim.array,
                        vectorized_dim.d,
                        vectorized_dim.index,
                        vectorized_dim.extent,
                        vectorized_dim.stride,
                        vectorized_dim.offset
                    )

                    if (dim.array, dim.d) in dim_map:
                        dim.offset += dim_map[(dim.array, dim.d)]

                    array_dim_extent = array_shapes[vectorized_dim.array][vectorized_dim.d]
                    final_dim = dim.clip(array_dim_extent)

                    vector_dims.append(final_dim)
                    seen_dims.add(vectorized_dim.d)

                elif isinstance(vectorized_dim, FillDim):
                    vector_dims.append(vectorized_dim)

                else:
                    raise Exception("vectorized_dim has to have type Dim or FillDim")

            # compute fixed dims
            fixed_dims = []
            array_dims = list(range(len(array_shapes[vec_array])))
            for d in array_dims:
                if d not in seen_dims:
                    offset = dim_map[(vec_array, d)]
                    fixed_dims.append(FixedDim(vec_array, d, offset))

            return Vector(vector_dims, fixed_dims)

        # process exploded dim
        else:
            exploded_dim, rest_exploded_dims = exploded_dims[0], exploded_dims[1:]
            vector_list = []

            for i in range(exploded_dim.extent):
                new_dim_map = dim_map.copy()

                if isinstance(exploded_dim, Dim):
                    array_dim = (exploded_dim.array, exploded_dim.d)
                    offset = (i * exploded_dim.stride) + exploded_dim.offset
                    if array_dim in new_dim_map:
                        new_dim_map[array_dim] += offset

                    else:
                        new_dim_map[array_dim] = offset

                elif isinstance(exploded_dim, FillDim):
                    pass

                else:
                    raise Exception("exploded_dim has to have type Dim or FillDim")


                vector_list.append(
                    self.materialize_recur(array_shapes, new_dim_map, rest_exploded_dims)
                )

            return vector_list

    def materialize(self, array_shapes):
        # compute shape from schedule
        exploded_shape = list(map(lambda d: d.shape(), self.exploded_dims))
        vectorized_shape = list(map(lambda d: d.shape(), self.vectorized_dims))
        shape = Shape(exploded_shape, vectorized_shape)
        return (self.materialize_recur(array_shapes, {}, self.exploded_dims), shape)

    def materialize2(self, array_shapes):
        # first, process exploded dimensions
        exploded_dims_info = []
        for dim_num, dim in enumerate(self.exploded_dims):
            if isinstance(dim, Dim):
                name = "i{}".format(dim_num + 1)
                offsets = [(i * dim.stride) + dim.offset for i in range(dim.extent)]
                array_info = ArrayDimTraversal(dim.array, dim.d, dim.stride, dim.offset)
                dim_info = ExplodedDimInfo(array_info, dim.index, IndexVar(name), dim.extent)
                exploded_dims_info.append(dim_info)

            elif isinstance(dim, FillDim):
                exploded_dims_info.append(ExplodedDimInfo(None, dim.index, None, dim.extent))

            else:
                raise Exception("exploded dim must have type Dim or FillDim")

        # second, process vectorized dims
        vector_dims = []
        vec_array = None
        seen_dims = set()
        for vectorized_dim in self.vectorized_dims:
            if isinstance(vectorized_dim, Dim):
                if vec_array is None:
                    vec_array = vectorized_dim.array

                if vec_array != vectorized_dim.array:
                    raise Exception("vector cannot have dimensions from two different arrays")

                dim = Dim(
                    vectorized_dim.array,
                    vectorized_dim.d,
                    vectorized_dim.index,
                    vectorized_dim.extent,
                    vectorized_dim.stride,
                    OffsetLit(vectorized_dim.offset)
                )

                for edim in exploded_dims_info:
                    if edim.array_dim is not None \
                        and edim.array_dim.array == dim.array \
                        and edim.array_dim.d == dim.d:
                            if dim.offset == OffsetLit(0):
                                dim.offset = edim.get_offset_expr()

                            else:
                                dim.offset = OffsetAdd(dim.offset, edim.get_offset_expr())

                vector_dims.append(dim)
                seen_dims.add(dim.d)

            elif isinstance(vectorized_dim, FillDim):
                vector_dims.append(vectorized_dim)

            else:
                raise Exception("vectorized_dim has to have type Dim or FillDim")

        # compute fixed dims
        fixed_dims = []
        array_dims = list(range(len(array_shapes[vec_array])))
        for d in array_dims:
            if d not in seen_dims:
                offset = OffsetLit(0)
                for edim in exploded_dim_info:
                    if edim.array_dim is not None \
                        and edim.array_dim.array == vec_array \
                        and edim.array_dim.d == d:
                            if offset == OffsetLit(0):
                                offset = edim.get_offset_expr()

                            else:
                                offset = OffsetAdd(offset, edim.get_offset_expr())

                fixed_dims.append(FixedDim(vec_array, d, offset))

        return exploded_dims_info, Vector(vector_dims, fixed_dims)


def apply_op_list(list1, list2):
    if isinstance(list1, list) and isinstance(list2, list):
        result = []
        for item1, item2 in zip(list1, list2):
            result.append(apply_op_list(item1, item2))

        return result

    elif (not isinstance(list1, list)) and (not isinstance(list2, list)):
        return Op(list1, list2)
    
    else:
        raise Exception("wrong shape lists: {} and {}".format(list1, list2))


class VectorMaterializer:
    def __init__(self, array_shape):
        self.array_shape = array_shape
        self.cur_vector_id = 1
        self.vector_map = {}
        self.parent_map = {}

    def find_parent(self, vector_id):
        vector = self.vector_map[vector_id]
        for k_id, v_parent in self.parent_map.copy().items():
            if v_parent is None:
                k_vector = self.vector_map[k_id]
                steps, _ = k_vector.derive(vector)
                if steps is None:
                    rev_steps, _ = vector.derive(k_vector)
                    if rev_steps is not None:
                        self.parent_map[k_id] = vector_id
                        return None
                    
                else:
                    self.parent_map[vector_id] = k_id
                    return k_id

        return None

    def record_vector(self, vector):
        vector_id = self.cur_vector_id
        self.vector_map[vector_id] = vector
        self.cur_vector_id += 1

        parent = self.find_parent(vector_id)
        self.parent_map[vector_id] = parent
        return vector_id

    def get_parent(self, vector_id):
        parent_id = self.parent_map[vector_id]
        if parent_id is None:
            return vector_id

        else:
            return self.get_parent(parent_id)

    def derive_vector(self, vector_id):
        parent_id = self.get_parent(vector_id)
        parent = self.vector_map[parent_id]
        vector = self.vector_map[vector_id]
        steps, defined_interval = parent.derive(vector)
        return { "vector": parent_id, "rotation_steps": steps, "mask": Mask(defined_interval) }

    def eval_offset(self, index_map, offset_expr):
        if isinstance(offset_expr, OffsetLit):
            return offset_expr.num

        elif isinstance(offset_expr, OffsetAdd):
            num1 = self.eval_offset(index_map, offset_expr.expr1)
            num2 = self.eval_offset(index_map, offset_expr.expr2)
            return num1 + num2

        elif isinstance(offset_expr, OffsetMul):
            num1 = self.eval_offset(index_map, offset_expr.expr1)
            num2 = self.eval_offset(index_map, offset_expr.expr2)
            return num1 * num2

        elif isinstance(offset_expr, IndexVar):
            return index_map[offset_expr.name]

    def resolve_vector_offset(self, index_map, vector):
        new_dims = []
        for dim in vector.dims:
            new_offset = self.eval_offset(index_map, dim.offset)
            new_dim = \
                Dim(dim.array, dim.d, dim.index, dim.extent, \
                    dim.stride, new_offset, dim.pad_left, dim.pad_right) \
                .clip(self.array_shape[dim.d])
            new_dims.append(new_dim)

        new_fixed_dims = []
        for fixed_dim in vector.fixed_dims:
            new_offset = self.eval_offset(fixed_dim.offset)
            new_fixed_dim = FixedDim(fixed_dim.array, fixed_dim.d, new_offset)
            new_fixed_dims.append(new_fixed_dim)

        return Vector(new_dims, new_fixed_dims)

    def derive_vector_list(self, exploded_dims, vector):
        indices = itertools.product(*list(map(lambda d: list(range(d.extent)), exploded_dims)))
        index_vars = list(map(lambda d: d.index_var.name, exploded_dims))
        index_maps = list(map(lambda index: dict(zip(index_vars, index)), indices))

        vector_id_map = {}
        for index_map in index_maps:
            resolved_vector = self.resolve_vector_offset(index_map, vector)
            index_values = tuple(map(lambda var: index_map[var], index_vars))
            vector_id = self.record_vector(resolved_vector)
            vector_id_map[index_values] = vector_id

        vector_data_map = {}
        for index_values, vector_id in vector_id_map.items():
            vector_data_map[index_values] = self.derive_vector(vector_id)

        return index_vars, vector_data_map

    # compute a function over the index vars that describes the relationship
    # between the index vars and the rotation steps
    # this assumes that the rotation steps are a linear function of the index variables
    def compute_rotation_index_expr(self, index_vars, vector_data_map):
        n = len(index_vars)

        # first, calculate the base offset by probing (0, ..., 0)
        base_probe = tuple(0 for _i in range(n))
        base_steps = vector_data_map[base_probe]["rotation_steps"]

        offset_expr = OffsetLit(base_steps) if base_steps != 0 else None

        # next, probe each index
        # to probe the ith index, probe (0, ..., 1, ..., 0)
        # where the 1 is in the ith position of the coordinate
        for i, index_var in enumerate(index_vars):
            probe = tuple((0 if pi != i else 1) for pi in range(n))
            steps = vector_data_map[probe]["rotation_steps"] - base_steps

            if offset_expr is None:
                offset_expr = OffsetMul(OffsetLit(steps), index_var)

            else:
                offset_expr = OffsetAdd(offset_expr, OffsetMul(OffsetLit(steps), index_var))

        return offset_expr

    def materialize(self, exploded_dims, vector):
        index_vars, vector_data_map = self.derive_vector_list(exploded_dims, vector)
        rot_expr = self.compute_rotation_index_expr(index_vars, vector_data_map)
        
        ct_coord_map = {}
        ct_map = {}

        pt_cur_id = 1
        pt_coord_map = {}
        pt_map = {}
        pt_index = {}

        for index_values, data in vector_data_map.items():
            ct_coord_map[index_values] = data["vector"]
            if data["vector"] not in ct_map:
                ct_map[data["vector"]] = self.vector_map[data["vector"]]

            mask_interval = data["mask"]
            if mask_interval in pt_index:
                pt_coord_map[index_values] = pt_index[mask_interval]

            else:
                pt_index[mask_interval] = pt_cur_id
                pt_coord_map[index_values] = pt_cur_id
                pt_map[pt_cur_id] = mask_interval
                pt_cur_id += 1

        abstract_expr = Op(Rotate(rot_expr, CiphertextVar("ct")), PlaintextVar("pt"))
        return abstract_expr, ct_coord_map, ct_map, pt_coord_map, pt_map


class Materializer:
    def __init__(self, array_shapes):
        self.array_shapes = array_shapes
        self.cur_vector_id = 1
        self.vector_map = {}
        self.parent_map = {}

    def find_parent(self, vector_id):
        vector = self.vector_map[vector_id]
        for k_id, v_parent in self.parent_map.copy().items():
            if v_parent is None:
                k_vector = self.vector_map[k_id]
                steps, _ = k_vector.derive(vector)
                if steps is None:
                    rev_steps, _ = vector.derive(k_vector)
                    if rev_steps is not None:
                        self.parent_map[k_id] = vector_id
                        return None
                    
                else:
                    self.parent_map[vector_id] = k_id
                    return k_id

        return None

    def record_vector(self, vector):
        vector_id = self.cur_vector_id
        self.vector_map[vector_id] = vector
        self.cur_vector_id += 1

        parent = self.find_parent(vector_id)
        self.parent_map[vector_id] = parent
        return vector_id

    def get_parent(self, vector_id):
        parent_id = self.parent_map[vector_id]
        if parent_id is None:
            return vector_id

        else:
            return self.get_parent(parent_id)

    def derive_vector(self, vector_id):
        parent_id = self.get_parent(vector_id)
        parent = self.vector_map[parent_id]
        vector = self.vector_map[vector_id]
        steps, defined_interval = parent.derive(vector)

        if steps == 0:
            return vector

        else:
            return Op(Rotate(steps, parent), Mask(defined_interval))

    def map_vector_list(self, vec_list, f):
        if isinstance(vec_list, list):
            new_vec_list = []
            for vector in vec_list:
                new_vec_list.append(self.map_vector_list(vector, f))

            return new_vec_list

        else:
            return f(vec_list)

    def materialize(self, expr):
        # TODO support reducing vectorized dim
        if isinstance(expr, Reduce):
            expr_list, shape = self.materialize(expr.expr)

            nonreduced_dims = []
            reduced_dims = []
            for i, (_, index) in enumerate(shape.exploded):
                if index == expr.index:
                    reduced_dims.append(i)
                else:
                    nonreduced_dims.append(i)

            expr_nparr = np.array(expr_list)
            expr_nparr = expr_nparr.transpose(tuple(nonreduced_dims + reduced_dims))

            nonreduced_shape = tuple([shape.exploded[d][0] for d in nonreduced_dims])
            nonreduced_arr = []
            for ind in np.ndindex(nonreduced_shape):
                vec_arr = None
                if len(ind) < len(expr_nparr.shape):
                    vec_arr = expr_nparr[ind].tolist()
                else:
                    vec_arr = [expr_nparr[ind]]

                if len(vec_arr) > 1:
                    nonreduced_arr.append(OpList(vec_arr))

                else:
                    nonreduced_arr.append(vec_arr[0])

            new_expr_list = np.array(nonreduced_arr).reshape(nonreduced_shape).tolist()
            new_exploded_shape = []

            for d in nonreduced_dims:
                d_extent, d_index = shape.exploded[d]

                if d_index > expr.index:
                    new_exploded_shape.append((d_extent, d_index-1))
                else:
                    new_exploded_shape.append((d_extent, d_index))
                
            return new_expr_list, Shape(new_exploded_shape, shape.vectorized)

        elif isinstance(expr, Op):
            (expr_list1, shape1) = self.materialize(expr.expr1)
            (expr_list2, shape2) = self.materialize(expr.expr2)
            assert(shape1 == shape2)
            return (apply_op_list(expr_list1, expr_list2), shape1)

        elif isinstance(expr, Schedule):
            vector_list, shape = expr.materialize(self.array_shapes)
            vector_id_list = self.map_vector_list(vector_list, self.record_vector)
            derived_vector_list = self.map_vector_list(vector_id_list, self.derive_vector)
            return derived_vector_list, shape


# returns num_ops, num_rots, num_vectors
def measure_schedule(expr):
    if isinstance(expr, Vector):
        return 0, 0, 1

    elif isinstance(expr, Op):
        ops1, rots1, vecs1 = measure_schedule(expr.expr1)
        ops2, rots2, vecs2 = measure_schedule(expr.expr2)
        return ops1 + ops2 + 1, rots1 + rots2, vecs1 + vecs2

    elif isinstance(expr, OpList):
        child_ops, child_rots, child_vecs = measure_schedule(expr.exprs)
        return child_ops + (len(expr.exprs) - 1), child_rots, child_vecs

    elif isinstance(expr, Rotate):
        child_ops, child_rots, child_vecs = measure_schedule(expr.expr)
        return child_ops, child_rots + 1, child_vecs

    elif isinstance(expr, Mask):
        return 0, 0, 1

    elif isinstance(expr, list):
        ops, rots, vecs = 0, 0, 0
        for child in expr:
            child_ops, child_rots, child_vecs = measure_schedule(child)
            ops += child_ops
            rots += child_rots
            vecs += child_vecs

        return ops, rots, vecs

    else:
        raise Exception("measure_schedule: unknown type {}".format(type(expr)))


def test_dim_clip():
    dim1 = Dim("a", 0, 1, 16, 1, 1)
    clip_dim1 = dim1.clip(16)
    print("original: ", dim1)
    print("clipped: ", clip_dim1)

    dim2 = Dim("a", 0, 1, 16, 2, 1)
    clip_dim2 = dim2.clip(16)
    print("original: ", dim2)
    print("clipped: ", clip_dim2)

    dim3 = Dim("a", 0, 1, 16, 2, -1)
    clip_dim3 = dim3.clip(16)
    print("original: ", dim3)
    print("clipped: ", clip_dim3)

    dim4 = Dim("a", 0, 1, 16, 1, -1)
    clip_dim4 = dim1.clip(16)
    print("original: ", dim4)
    print("clipped: ", clip_dim4)

def test_dim_derive():
    dim1 = Dim("a", 0, 1, 16, 1, 0)
    dim2 = Dim("a", 0, 1, 16, 1, 1)
    dim3 = Dim("a", 0, 1, 16, 1, -1)
    derive2 = dim1.derive(dim2.clip(16))
    derive3 = dim1.derive(dim3.clip(16))
    print(derive2, derive3)

    dim1 = Dim("a", 0, 2, 8, 1, 0)
    dim2 = Dim("a", 0, 2, 8, 1, 1)
    derive2 = dim1.derive(dim2.clip(16))
    print(derive2)

def test_vec_derive():
    v1 = Vector([Dim("a", 0, 1, 16, 1, 0).clip(16)])
    v2 = Vector([Dim("a", 0, 1, 16, 1, 1).clip(16)])
    assert(v1.derive(v2) == -1)

    v3 = Vector([Dim("a", 1, 2, 16, 1, 0).clip(16),Dim("a", 0, 1, 16, 1, 0).clip(16)])
    v4 = Vector([Dim("a", 1, 2, 16, 1, 1).clip(16),Dim("a", 0, 1, 16, 1, 1).clip(16)])
    assert(v3.derive(v4) == -17)

    v5 = Vector([Dim("a", 1, 2, 8, 2, 0).clip(16),Dim("a", 0, 1, 16, 1, 0).clip(16)])
    v6 = Vector([Dim("a", 1, 2, 8, 2, 1).clip(16),Dim("a", 0, 1, 16, 1, 1).clip(16)])
    assert(v5.derive(v6) is None)

def test_materialize():
    array_shapes = { "a": [16,4], "b": [16,4] }

    ax_inner = Dim("a", 0, 2, 4, 1, 0)
    ax_outer = Dim("a", 0, 2, 4, 4, 0)
    ay = Dim("a", 1, 1, 4, 1, 0)
    s1 = Schedule([ax_outer], [ay, ax_inner])

    bx_inner = Dim("b", 0, 2, 4, 1, 0)
    bx_outer = Dim("b", 0, 2, 4, 4, 0)
    by = Dim("b", 1, 1, 4, 1, 0)
    s2 = Schedule([bx_outer], [by, bx_inner])

    expr = Reduce(1, Op(s1, s2))
    # expr = Op(s, s)

    mat = Materializer(array_shapes)
    expr_list, shape = mat.materialize(expr)

    ops, rots, vecs = measure_schedule(expr_list)
    print("ops: ", ops, "rots: ", rots, "vecs: ", vecs)

    expr_nparr = np.array(expr_list, dtype=object)
    for ind, val in np.ndenumerate(expr_nparr):
        print(ind, val)

def test_schedule_derive():
    array_shapes = { "img": [16,16], "filt": [3,3] }

    img_x = Dim("img", 0, 3, 16, 1, 0)
    img_y = Dim("img", 1, 4, 16, 1, 0)
    img_j = Dim("img", 0, 1, 3, 1, 0)
    img_k = Dim("img", 1, 2, 3, 1, 0)
    sched_img = Schedule([img_j, img_k], [img_y, img_x])

    filt_j = Dim("filt", 0, 1, 3, 1, 0)
    filt_k = Dim("filt", 1, 2, 3, 1, 0)
    sched_filt = Schedule([filt_j, filt_k], [FillDim(4,16),FillDim(3,16)])

    expr = Reduce(1, Reduce(2, Op(sched_img, sched_filt)))

    # for x. for y. sum(sum(for i. for j. img[x+i][y+j] * filt[i][j]))
    # expr = Op(sched_img, sched_filt)

    # (CT_img * PT_mask) * PT_filter
    # CT_img * (PT_mask * PT_filter)

    mat = Materializer(array_shapes)
    expr_list, shape = mat.materialize(expr)

    ops, rots, vecs = measure_schedule(expr_list)
    print("ops: ", ops, "rots: ", rots, "vecs: ", vecs)

    expr_nparr = np.array(expr_list, dtype=object)
    for ind, val in np.ndenumerate(expr_nparr):
        print(ind, val)


def test_materialize2():
    array_shapes = { "img": [16,16], "filt": [3,3] }
    img_x = Dim("img", 0, 3, 16, 1, 0)
    img_y = Dim("img", 1, 4, 16, 1, 0)
    img_j = Dim("img", 0, 1, 3, 1, 0)
    img_k = Dim("img", 1, 2, 3, 1, 0)
    sched_img = Schedule([img_j, img_k], [img_y, img_x])
    exploded_dims, vector_dims = sched_img.materialize2(array_shapes)
    vmat = VectorMaterializer(array_shapes["img"])
    abstract_expr, ct_coord_map, ct_map, pt_coord_map, pt_map = \
        vmat.materialize(exploded_dims, vector_dims)

    print(abstract_expr)
    for index_values in ct_coord_map.keys():
        print(index_values, "ciphertext", ct_coord_map[index_values], "mask", pt_coord_map[index_values])

def main():
    # test_vec_derive()
    # test_schedule_derive()
    test_materialize2()

if __name__ == "__main__":
    main()

