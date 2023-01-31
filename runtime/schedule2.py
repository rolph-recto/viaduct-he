#!/usr/bin/env python

import itertools

# abstract arrays

class ArrayDim:
    def __init__(self, dim_ref, extent, stride=1, offset=0):
        self.dim_ref = dim_ref
        self.extent = extent
        self.stride = stride
        self.offset = offset

    def __repr__(self):
        array_str = ""
        if self.dim_ref is not None:
            array_str = self.dim_ref.__repr__()

        return "{}:{}[{}x + {}]".format(self.dim_ref, self.extent, self.stride, self.offset)

# fixed dimensions stay constant within a vector
class FixedDim:
    def __init__(self, dim_ref, offset):
        self.dim_ref = dim_ref
        self.offset = offset

    def __repr__(self):
        array_str = ""
        if self.dim_ref is not None:
            array_str = self.dim_ref.__repr__()

        return "{}[{}]".format(array_str, self.offset)

class DimRef:
    def __init__(self, array, index):
        self.array = array
        self.index = index

    def __hash__(self):
        return hash((self.array, self.index))

    def __eq__(self, other):
        if isinstance(other, DimRef):
            return self.array == other.array and self.index == other.index

        raise NotImplemented

    def __repr__(self):
        return "{}.{}".format(self.array, self.index)

class Array:
    def __init__(self, dims, fixed_dims = []):
        self.dims = dims
        self.fixed_dims = fixed_dims

    def __repr__(self):
        return self.dims.__repr__()

# Offset expressions

BASE_OFFSET = "__BASE__"

class OffsetExpr:
    pass

class OffsetLit(OffsetExpr):
    def __init__(self, num):
        self.num = num

    def __repr__(self):
        return "{}".format(self.num)

    def __eq__(self, other):
        return isinstance(other, OffsetLit) and self.num == other.num

    def eval(self, index_map):
        return self.num

class OffsetAdd(OffsetExpr):
    def __init__(self, expr1, expr2):
        self.expr1 = expr1
        self.expr2 = expr2

    def __repr__(self):
        return "({} + {})".format(self.expr1, self.expr2)

    def eval(self, index_map):
        n1 = self.expr1.eval(index_map)
        n2 = self.expr2.eval(index_map)
        return n1 + n2

class OffsetMul(OffsetExpr):
    def __init__(self, expr1, expr2):
        self.expr1 = expr1
        self.expr2 = expr2

    def __repr__(self):
        return "({} * {})".format(self.expr1, self.expr2)

    def eval(self, index_map):
        n1 = self.expr1.eval(index_map)
        n2 = self.expr2.eval(index_map)
        return n1 * n2

class IndexVar(OffsetExpr):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, IndexVar):
            return self.name == other.name

        raise NotImplemented

    def eval(self, index_map):
        return index_map[self.name]

# schedule

class ScheduleDim:
    def __init__(self, index, extent, stride=1, index_var=None):
        self.index = index
        self.extent = extent
        self.stride = stride
        self.index_var = index_var

    def __repr__(self):
        var = self.index_var if self.index_var is not None else "_"
        return "[{}:{};{}{} + {}]".format(var, self.extent, self.stride, var)

    def __eq__(self, other):
        if isinstance(other, ScheduleDim):
            return \
                self.index == other.index \
                and self.extent == other.extent \
                and self.stride == other.stride \
                and self.index_var == other.index_var

        raise NotImplemented

class Schedule:
    def __init__(self, exploded_dims, vectorized_dims):
        self.exploded_dims = exploded_dims
        self.vectorized_dims = vectorized_dims

    def __eq__(self, other):
        if isinstance(other, Schedule):
            same_exploded_num = \
                len(self.exploded_dims) == len(other.exploded_dims)
            same_vectorized_num = \
                len(self.vectorized_dims) == len(other.vectorized_dims)
            equiv_exploded = \
                all(edim1 == edim2 for edim1, edim2 in \
                    zip(self.exploded_dims, other.exploded_dims))
            equiv_vectorized = \
                all(vdim1 == vdim2 for vdim1, vdim2 in \
                    zip(self.vectorized_dims, other.vectorized_dims))

        return same_exploded_num and same_vectorized_num \
                and equiv_exploded and equiv_vectorized

    def build_offset_expr(self, e_stride, exploded_dim, cur_offset):
        offset_factor = exploded_dim.index_var
        offset_expr = cur_offset

        if e_stride != 1:
            offset_factor = OffsetMul(OffsetLit(stride), offset_factor)

        if offset_expr == OffsetLit(0):
            offset_expr = offset_factor

        else:
            offset_expr = OffsetAdd(offset_expr, offset_factor)

        return offset_expr

    def prematerialize_array(self, array):
        vec_array = None

        # first, process exploded dimensions
        exploded_dims_map = {}
        exploded_dims_list = []
        for dim_num, sched_dim in enumerate(self.exploded_dims):
            array_dim = array.dims[sched_dim.index]
            assert(sched_dim.index_var is not None)

            exploded_dim_info = \
                ExplodedDimInfo(
                    sched_dim.index,
                    sched_dim.extent,
                    IndexVar(sched_dim.index_var)
                )
            exploded_dims_list.append(exploded_dim_info)

            # filled dim
            if array_dim.dim_ref is not None:
                if vec_array is None:
                    vec_array = array_dim.dim_ref.array

                if vec_array != array_dim.dim_ref.array:
                    raise Exception("vector cannot have dimensions from two different arrays")

                value = (sched_dim.stride, exploded_dim_info)
                if array_dim.dim_ref not in exploded_dims_map:
                    exploded_dims_map[array_dim.dim_ref] = [value]

                else:
                    exploded_dims_map[array_dim.dim_ref].append(value)
            
            # empty dim
            else:
                pass

        # second, process vectorized dims
        vector_dims = []
        seen_dims = set()
        for sched_dim in self.vectorized_dims:
            array_dim = array.dims[sched_dim.index]
            offset_expr = OffsetLit(array_dim.offset)
            materialized_dim_ref = None

            # filled dim
            if array_dim.dim_ref is not None:
                if vec_array is None:
                    vec_array = array_dim.dim_ref.array

                if vec_array != array_dim.dim_ref.array:
                    raise Exception("vector cannot have dimensions from two different arrays")

                for data in exploded_dims_map.get(array_dim.dim_ref, []):
                    (e_stride, exploded_dim) = data
                    offset_expr = \
                        self.build_offset_expr(e_stride, exploded_dim, offset_expr)

                seen_dims.add(array_dim.dim_ref.index)

            # empty dim
            else:
                pass

            materialized_dim = MaterializedDim(
                array_dim.dim_ref,
                sched_dim.index,
                sched_dim.extent,
                array_dim.stride * sched_dim.stride,
                offset_expr
            )
            vector_dims.append(materialized_dim)

        # compute all dimensions in the array
        array_filled_dims = []
        for dim in array.dims:
            # filled dim
            if dim.dim_ref is not None:
                array_filled_dims.append(dim.dim_ref.index)

            # empty dim
            else:
                pass
            
        # fixed dims have to include the array's fixed dims
        fixed_dims = \
            [FixedDim(dim.dim_ref, OffsetLit(dim.offset)) \
            for dim in array.fixed_dims]

        # if a dim is not a vectorized dim but also not in array.fixed_dims,
        # then it must be a fixed dim of the materialized array also
        for index in array_filled_dims:
            if index not in seen_dims:
                offset_expr = OffsetLit(0)
                dim_ref = DimRef(vec_array, index)
                for data in exploded_dims_map.get(dim_ref, []):
                    (e_stride, exploded_dim) = data

                    offset_expr = \
                        self.build_offset_expr(
                            e_stride,
                            exploded_dim,
                            offset_expr)

                fixed_dims.append(FixedDim(dim_ref, offset_expr))

        return exploded_dims_list, Vector(vector_dims, fixed_dims)

# materialized arrays

class MaterializedDim:
    def __init__(self, dim_ref, index, extent, stride=1, offset=0, pad_left=0, pad_right=0):
        self.dim_ref = dim_ref
        self.index = index
        self.extent = extent
        self.stride = stride
        self.offset = offset
        self.pad_left = pad_left
        self.pad_right = pad_right

    # exclude index in hashing and equality
    def __eq__(self, other):
        if isinstance(other, MaterializedDim):
            return \
                self.dim_ref == other.dim_ref \
                and self.extent == other.extent \
                and self.stride == other.stride \
                and self.offset == other.offset \
                and self.pad_left == other.pad_right

        raise NotImplemented

    def __hash__(self):
        return hash((self.dim_ref, self.extent, self.stride, self.offset, self.pad_left, self.pad_right))

    def __repr__(self):
        array_str = ""
        if self.dim_ref is not None:
            array_str = self.dim_ref.__repr__()

        if self.pad_left == 0 and self.pad_right == 0:
            return "({}){}:{}[{}x + {}]".format(
                    array_str, self.index, self.extent, self.stride, self.offset)

        else:
            return "({}){}:{}[{}x + {};padl={};padr={}]".format(
                    array_str, self.index, self.extent, self.stride, self.offset,
                    self.pad_left, self.pad_right)

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
        new_dim = MaterializedDim(self.dim_ref, self.index, new_extent, self.stride, new_offset, new_pad_left, new_pad_right)
        return new_dim

    # derive `other` dim from `self` dim
    # returns None if not possible, otherwise
    # returns the rotation step needed for the derivation
    # TODO: generalize this properly
    def derive(self, other):
        if isinstance(other, MaterializedDim):
            # first, check if derivation is possible
            same_array_stride = self.dim_ref == other.dim_ref and self.stride == other.stride

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

class ExplodedDimInfo:
    def __init__(self, index, extent, index_var):
        self.index = index
        self.extent = extent
        self.index_var = index_var

    def __eq__(self, other):
        if isinstance(other, ExplodedDimInfo):
            return self.index == other.index \
                    and self.extent == other.extent \
                    and self.index_var == other.index_var

        raise NotImplemented

    def __repr__(self):
        return "{}:{}[{}]".format(self.index, self.extent, self.index_var)

# index-free expr

class IndexFreeExpr:
    def compute_output_schedule(self, sched_map):
        raise NotImplemented
    
    def is_schedule_valid(self, sched_map):
        return self.compute_output_schedule(sched_map) is not None
    
class IFOp(IndexFreeExpr):
    def __init__(self, expr1, expr2):
        self.expr1 = expr1
        self.expr2 = expr2

    def __repr__(self):
        return "Op({}, {})".format(self.expr1, self.expr2)

    def compute_output_schedule(self, sched_map):
        sched1 = self.expr1.compute_output_schedule(sched_map)
        sched2 = self.expr2.compute_output_schedule(sched_map)

        if sched1 is None or sched2 is None:
            return None

        elif sched1 != sched2:
            return None

        else:
            return sched1

class IFReduce(IndexFreeExpr):
    def __init__(self, index, expr):
        self.index = index
        self.expr = expr

    def __repr__(self):
        return "Reduce({}, {})".format(self.index, self.expr)

    # because schedules don't contain materialized dims,
    # reductions of vectorized dims don't change the schedule
    # (cf. if a materialized vectorized dim is reduced, then it becomes empty)
    def compute_output_schedule(self, sched_map):
        sched = self.expr.compute_output_schedule(sched_map)

        new_exploded_dims = []
        for dim in sched.exploded_dims:
            if dim.index == self.index:
                pass

            elif dim.index > self.index:
                new_dim = ScheduleDim(dim.index-1, dim.extent, dim.stride, dim.offset, dim.index_var)
                new_exploded_dims.append(new_dim)

            else:
                new_exploded_dims.append(dim)

        return Schedule(new_exploded_dims, sched.vectorized_dims)
        

class IFArrayRef(IndexFreeExpr):
    def __init__(self, array):
        self.array = array

    def __repr__(self):
        return self.array

    def compute_output_schedule(self, sched_map):
        if self.array in sched_map:
            return sched_map[self.array]

        raise Exception("{} is not in the schedule map".format(self.array))



# abstract expr

class AbstractExpr:
    pass

class AbstractOp(AbstractExpr):
    def __init__(self, expr1, expr2):
        self.expr1 = expr1
        self.expr2 = expr2

    def __repr__(self):
        return "Op({}, {})".format(self.expr1, self.expr2)

class AbstractReduce(AbstractExpr):
    def __init__(self, index_vars, expr):
        self.index_vars = index_vars
        self.expr = expr

    def __repr__(self):
        return "Reduce({}, {})".format(self.index, self.expr)

class AbstractRotate(AbstractExpr):
    def __init__(self, steps, expr):
        self.steps = steps
        self.expr = expr

    def __repr__(self):
        return "Rot({}, {})".format(self.steps, self.expr)

class CiphertextVar(AbstractExpr):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

class PlaintextVar(AbstractExpr):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

# concrete expr

class Expr:
    pass

class Rotate(Expr):
    def __init__(self, steps, expr):
        self.steps = steps
        self.expr = expr

    def __repr__(self):
        return "Rotate({},{})".format(self.steps, self.expr)

class Op(Expr):
    def __init__(self, expr1, expr2):
        self.expr1 = expr1
        self.expr2 = expr2

    def __repr__(self):
        return "Op({},{})".format(self.expr1, self.expr2)

class OpList(Expr):
    def __init__(self, expr_list):
        self.expr_list = expr_list

    def __repr__(self):
        return "Op({})".format(self.expr_list)

class Vector(Expr):
    def __init__(self, dims, fixed_dims=[]):
        self.dims = dims
        self.fixed_dims = fixed_dims

    # derive one vector from another
    def derive(self, other):
        self_fixed_dim_set = set()
        for fd in self.fixed_dims:
            self_fixed_dim_set.add((fd.dim_ref, fd.offset))

        other_fixed_dim_set = set()
        for fd in other.fixed_dims:
            other_fixed_dim_set.add((fd.dim_ref, fd.offset))

        same_fixed_dim = self_fixed_dim_set == other_fixed_dim_set

        if same_fixed_dim and len(self.dims) == len(other.dims):
            block_size = 1
            total_steps = 0
            dim_extents = []
            defined_interval = []
            for i in reversed(range(len(self.dims))):
                steps, dim_defined_interval = self.dims[i].derive(other.dims[i])
                if steps is not None:
                    total_steps += (block_size * steps)
                    block_size *= self.dims[i].extent
                    dim_extent = self.dims[i].extent + self.dims[i].pad_left + self.dims[i].pad_right
                    dim_extents.insert(0, dim_extent)
                    defined_interval.insert(0, dim_defined_interval)
                
                else:
                    return None, None, None

            return total_steps, tuple(dim_extents), tuple(defined_interval)

        else:
            return None, None, None

    def __repr__(self):
        fixed_dim_str = "{{{}}}".format(",".join(map(str, self.fixed_dims))) if len(self.fixed_dims) > 0 else ""
        dim_str = "<{}>".format(",".join(map(str, self.dims)))
        return fixed_dim_str + dim_str

class Mask(Expr):
    def __init__(self, extents, defined_interval):
        self.extents = extents
        self.defined_interval = defined_interval

    def __eq__(self, other):
        if isinstance(other, Mask):
            return \
                self.extents == other.extents \
                and self.defined_interval == other.defined_interval

        raise NotImplemented

    def __hash__(self):
        return hash((self.extents, self.defined_interval))

    def __repr__(self):
        dims_str = \
            ", ".join(map( \
                lambda data: "({},{}):{}".format(data[1][0], data[1][1], data[0]),
                zip(self.extents, self.defined_interval)))
        return "Mask([{}])".format(dims_str)

# materialization

class ArrayMaterializer:
    # materialize the prematerialized array
    def materialize(self, exploded_dims, vector, array, registry):
        raise NotImplemented

    # check if the prematerialized array can be processed by this materializer
    def can_process(self, exploded_dims, vector, array, registry):
        raise NotImplemented

class VectorRegistry:
    def __init__(self):
        # map from variables to coordinate maps
        self.ct_vars = {}
        self.ct_var_id = 1

        self.pt_vars = {}
        self.pt_var_id = 1

    def fresh_ciphertext_var(self):
        name = "ct{}".format(self.ct_var_id)
        self.ct_var_id += 1
        return CiphertextVar(name)

    def fresh_plaintext_var(self):
        name = "pt{}".format(self.pt_var_id)
        self.pt_var_id += 1
        return PlaintextVar(name)

    def set_ciphertext_coord_map(self, ct_var, coord_map):
        self.ct_vars[ct_var.name] = coord_map

    def set_plaintext_coord_map(self, pt_var, coord_map):
        self.pt_vars[pt_var.name] = coord_map

    def get_ciphertext_coord_map(self, ct_var):
        return self.ct_vars.get(ct_var.name, None) 

    def get_plaintext_coord_map(self, pt_var):
        return self.pt_vars.get(pt_var.name, None) 

class DefaultArrayMaterializer(ArrayMaterializer):
    def __init__(self):
        self.cur_vector_id = 1
        self.vector_map = {}
        self.parent_map = {}

    def find_parent(self, vector_id):
        vector = self.vector_map[vector_id]
        for k_id, v_parent in self.parent_map.copy().items():
            if v_parent is None:
                k_vector = self.vector_map[k_id]
                steps, _, _ = k_vector.derive(vector)
                if steps is None:
                    rev_steps, _, _ = vector.derive(k_vector)
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
        steps, dim_extents, defined_interval = parent.derive(vector)
        return {
            "vector": parent_id,
            "rotation_steps": steps,
            "mask": Mask(dim_extents, defined_interval)
        }

    def resolve_vector_offset(self, index_map, vector, array):
        new_dims = []
        for dim in vector.dims:
            new_offset = dim.offset.eval(index_map)
            new_dim = \
                MaterializedDim(dim.dim_ref, dim.index, dim.extent, \
                    dim.stride, new_offset, dim.pad_left, dim.pad_right) \
                .clip(array.dims[dim.index].extent)
            new_dims.append(new_dim)

        new_fixed_dims = []
        for fixed_dim in vector.fixed_dims:
            new_offset = fixed_dim.offset.eval(index_map)
            new_fixed_dim = FixedDim(fixed_dim.dim_ref, new_offset)
            new_fixed_dims.append(new_fixed_dim)

        return Vector(new_dims, new_fixed_dims)

    def derive_vector_list(self, exploded_dims, vector, array):
        sorted_var_extents = \
            sorted( \
                map(lambda d: (d.index_var.name, d.extent), exploded_dims),
                key=lambda p: p[0])
        index_values_list = itertools.product(*list(map(lambda p: range(p[1]), sorted_var_extents)))
        index_vars = list(map(lambda p: p[0], sorted_var_extents))

        vector_id_map = {}
        for index_values in index_values_list:
            index_map = dict(zip(index_vars, index_values))
            resolved_vector = self.resolve_vector_offset(index_map, vector, array)
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
        print("vector data map", vector_data_map)

        # next, probe each index
        # to probe the ith index, probe (0, ..., 1, ..., 0)
        # where the 1 is in the ith position of the coordinate
        for i, index_var in enumerate(index_vars):
            probe = tuple((0 if pi != i else 1) for pi in range(n))
            steps = vector_data_map[probe]["rotation_steps"] - base_steps

            if offset_expr is None:
                offset_expr = OffsetMul(OffsetLit(steps), IndexVar(index_var))

            else:
                offset_expr = OffsetAdd(offset_expr, OffsetMul(OffsetLit(steps), IndexVar(index_var)))

        return offset_expr

    def materialize(self, exploded_dims, vector, array, registry):
        index_vars, vector_data_map = self.derive_vector_list(exploded_dims, vector, array)
        rot_expr = self.compute_rotation_index_expr(index_vars, vector_data_map)
        
        ct_coord_map = {}
        pt_coord_map = {}

        for index_values, data in vector_data_map.items():
            ct_coord_map[index_values] = self.vector_map[data["vector"]]
            pt_coord_map[index_values] = data["mask"]

        ct_var = registry.fresh_ciphertext_var()
        pt_var = registry.fresh_plaintext_var()

        registry.set_ciphertext_coord_map(ct_var, ct_coord_map)
        registry.set_plaintext_coord_map(pt_var, pt_coord_map)

        abstract_expr = AbstractOp(AbstractRotate(rot_expr, ct_var), pt_var)
        return abstract_expr

    # the default array materializer can process any array with any valid schedule
    def can_process(self, exploded_dims, vector, array):
        return True


class Materializer:
    def __init__(self, array_materializers, array_map):
        # list of array materializers
        # that will be tried in sequence
        self.array_materializers = array_materializers

        # arrays to be materialized
        self.array_map = array_map

        self.registry = VectorRegistry()

    def materialize(self, expr, schedule_map):
        if isinstance(expr, IFOp):
            exploded_dims1, mat_expr1 = self.materialize(expr.expr1, schedule_map)
            exploded_dims2, mat_expr2 = self.materialize(expr.expr2, schedule_map)

            assert(len(exploded_dims1) == len(exploded_dims2))
            assert(all(d1 == d2 for d1, d2 in zip(exploded_dims1, exploded_dims2)))

            return exploded_dims1, AbstractOp(mat_expr1, mat_expr2)

        # TODO: implement reduction for vectorized dims
        elif isinstance(expr, IFReduce):
            exploded_dims, abstract_expr = self.materialize(expr.expr)

            reduced_dims = []
            new_exploded_dims = []
            for dim in exploded_dims:
                if dim.index == expr.index:
                    reduced_dims.append(dim)

                elif dim.index > expr.index:
                    new_dim = ExplodedDimInfo(dim.index-1, dim.extent, dim.index_var)
                    new_exploded_dims.append(new_dim)

                else:
                    new_exploded_dims.append(dim)

            return new_exploded_dims, AbstractReduce(reduced_dims, abstract_expr)

        elif isinstance(expr, IFArrayRef):
            array = self.array_map[expr.array]
            sched = schedule_map[expr.array]
            exploded_dims, vector = sched.prematerialize_array(array)

            for materializer in self.array_materializers:
                if materializer.can_process(exploded_dims, vector, array):
                    abstract_expr = \
                        materializer.materialize(exploded_dims, vector, array, self.registry)

                    return exploded_dims, abstract_expr

            raise Exception("no registered materializer can process array")

# concretization
class Concretizer:
    def __init__(self, registry):
        self.registry = registry

    def concretize(self, index_vars, abstract_expr):
        if isinstance(abstract_expr, CiphertextVar):
            coord_map = self.registry.get_ciphertext_coord_map(abstract_expr)
            return coord_map

        elif isinstance(abstract_expr, PlaintextVar):
            coord_map = self.registry.get_plaintext_coord_map(abstract_expr)
            return coord_map

        elif isinstance(abstract_expr, AbstractRotate):
            expr_map = self.concretize(index_vars, abstract_expr.expr)
            result_map = {}
            for index_values, expr in expr_map.items():
                index_map = dict(zip(map(lambda p: p[0], index_vars), index_values))
                steps = abstract_expr.steps.eval(index_map)

                if steps != 0:
                    result_map[index_values] = Rotate(steps, expr)

                else:
                    result_map[index_values] = expr

            return result_map

        elif isinstance(abstract_expr, AbstractOp):
            expr_map1 = self.concretize(index_vars, abstract_expr.expr1)
            expr_map2 = self.concretize(index_vars, abstract_expr.expr2)
            assert(expr_map1.keys() == expr_map2.keys())

            result_map = {}
            for index_values, expr1 in expr_map1.items():
                expr2 = expr_map2[index_values]
                result_map[index_values] = Op(expr1, expr2)

            return result_map
                
        elif isinstance(abstract_expr, AbstractReduce):
            # add reduced index vars when processing child expr
            child_index_var_extents = \
                sorted(
                    abstract_expr.index_vars + index_vars,
                    key=lambda p: p[0])


            var_list = list(map(lambda p: p[0], child_index_var_extents))
            reduced_vars = list(map(lambda p: p[0], abstract_expr.index_vars))
            reduced_vars_pos = [var_list.index(var) for var in reduced_vars]

            output_vars = list(map(lambda p: p[0], index_vars))
            output_vars_pos = [var_list.index(var) for var in output_vars]

            expr_map = self.concretize(child_index_var_extents, abstract_expr.expr)
            result_map = {}

            reduced_index_values_list = \
                list(itertools.product(*list(map(lambda p: p[1], abstract_expr.index_vars))))

            output_index_values_list = \
                itertools.product(*list(map(lambda p: p[1], index_vars)))

            if len(output_vars) > 0:
                for output_index_values in output_index_values_list:
                    sum_list = []
                    for reduced_index_values in reduced_index_values_list:
                        index_values = tuple(0 for _ in range(len(var_list)))

                        for i in range(len(output_vars)):
                            index_values[output_vars_pos[i]] = output_index_values[i]

                        for i in range(len(reduced_vars)):
                            index_values[reduced_vars_pos[i]] = reduced_index_values[i]

                        sum_list.append(expr_map[index_values])

                    assert(len(sum_list) > 0)

                    if len(sum_list) > 1:
                        result_map[output_index_values] = OpList(sum_list)

                    else:
                        result_map[output_index_values] = sum_list[0]

                return result_map

            else:
                sum_list = []
                for reduced_index_values in reduced_index_values_list:
                    index_values = tuple(0 for _ in range(len(var_list)))
                    sum_list.append(expr_map[reduced_index_values])

                assert(len(sum_list) > 0)

                if len(sum_list) > 1:
                    return OpList(sum_list)

                else:
                    return sum_list[0]


# tests

def test_materializer():
    img = Array([
        ArrayDim(DimRef("img", 1), 16, 1, 0),
        ArrayDim(DimRef("img", 2), 16, 1, 0),
        ArrayDim(DimRef("img", 1), 3, 1, 0),
        ArrayDim(DimRef("img", 2), 3, 1, 0),
    ])

    sched = Schedule(
        # exploded
        [
            ScheduleDim(2, 3, 1, "i"),
            ScheduleDim(3, 3, 1, "j")
        ],

        # vectorized
        [
            ScheduleDim(0, 16, 1),
            ScheduleDim(1, 16, 1)
        ]
    )

    exploded_dims, vector = sched.prematerialize_array(img)
    mat = DefaultArrayMaterializer()
    abstract_expr, ct_coord_map, ct_map, pt_coord_map, pt_map = \
        mat.materialize(exploded_dims, vector, img)

    print(exploded_dims, vector)
    print(abstract_expr)
    for index_vals, ct_ref in ct_coord_map.items():
        print(index_vals, "ct: ", ct_map[ct_ref], "pt: ", pt_map[pt_coord_map[index_vals]])

def test_materializer2():
    array_map = {
        "img": Array([
            ArrayDim(DimRef("img", 1), 16, 1, 0),
            ArrayDim(DimRef("img", 2), 16, 1, 0),
            ArrayDim(DimRef("img", 1), 3, 1, 0),
            ArrayDim(DimRef("img", 2), 3, 1, 0),
        ])
    }

    sched_map = {
        "img": Schedule(
            # exploded
            [
                ScheduleDim(2, 3, 1, "i"),
                ScheduleDim(3, 3, 1, "j")
            ],
            # vectorized
            [
                ScheduleDim(0, 16, 1),
                ScheduleDim(1, 16, 1)
            ]
        )
    }

    expr = IFOp(IFArrayRef("img"), IFArrayRef("img"))
    print(expr.is_schedule_valid(sched_map))

    mat = Materializer([DefaultArrayMaterializer()], array_map)
    exploded_dims, abstract_expr = mat.materialize(expr, sched_map)
    print(exploded_dims)
    print(abstract_expr)

def test_concretizer():
    array_map = {
        "img1": Array([
            ArrayDim(DimRef("img", 0), 16, 1, 0),
            ArrayDim(DimRef("img", 1), 16, 1, 0),
            ArrayDim(DimRef("img", 0), 3, 1, 0),
            ArrayDim(DimRef("img", 1), 3, 1, 0),
        ]),

        "f1": Array([
            ArrayDim(None, 16, 1, 0),
            ArrayDim(None, 16, 1, 0),
            ArrayDim(DimRef("f", 0), 3, 1, 0),
            ArrayDim(DimRef("f", 1), 3, 1, 0),
        ]),
    }

    sched_map = {
        "img1": Schedule(
            # exploded
            [
                ScheduleDim(2, 3, 1, "i"),
                ScheduleDim(3, 3, 1, "j")
            ],
            # vectorized
            [
                ScheduleDim(0, 16, 1, "x"),
                ScheduleDim(1, 16, 1, "y")
            ]
        ),
        "f1": Schedule(
            # exploded
            [
                ScheduleDim(2, 3, 1, "i"),
                ScheduleDim(3, 3, 1, "j")
            ],
            # vectorized
            [
                ScheduleDim(0, 16, 1, "x"),
                ScheduleDim(1, 16, 1, "y")
            ]
        ),
    }

    expr = IFOp(IFArrayRef("img1"), IFArrayRef("f1"))
    print(expr.is_schedule_valid(sched_map))

    mat = Materializer([DefaultArrayMaterializer()], array_map)
    exploded_dims, abstract_expr = mat.materialize(expr, sched_map)

    index_vars = list(map(lambda d: (d.index_var.name, d.extent), exploded_dims))
    con = Concretizer(mat.registry)
    expr_map = con.concretize(index_vars, abstract_expr)

    for index_values, expr in expr_map.items():
        print(index_values, expr)

if __name__ == "__main__":
    # test_materializer()
    # test_materializer2()
    test_concretizer()

