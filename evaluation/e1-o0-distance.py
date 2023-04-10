import sys
import itertools
import json
import numpy as np
from time import time
from typing import *

from seal import *

class ArrayPreprocessing:
    def apply(self, arr):
        raise NotImplemented


class Roll(ArrayPreprocessing):
    def __init__(self, dim_i, dim_j):
        self.dim_i = dim_i
        self.dim_j = dim_j

    def apply(self, arr):
        slices = []
        for ind in range(arr.shape[self.dim_j]):
            j_slice = np.take(arr, ind, axis=self.dim_j)
            roll_dim = self.dim_i if self.dim_i < self.dim_j else self.dim_i - 1
            perm_j_slice = np.roll(j_slice, -ind, axis=roll_dim)
            slices.append(perm_j_slice)

        return np.stack(slices, axis=self.dim_j)

    def __str__(self):
        return "Roll({},{})".format(self.dim_i, self.dim_j)


class VectorDimContent:
    def __init__(self):
        pass

    def offset_left(self):
        raise NotImplemented

    def size(self):
        raise NotImplemented


class FilledDim(VectorDimContent):
    def __init__(self, dim, extent, stride=1, oob_left=0, oob_right=0, pad_left=0, pad_right=0):
        super().__init__()
        self.dim = dim
        self.extent = extent
        self.stride = stride
        self.oob_left = oob_left
        self.oob_right = oob_right
        self.pad_left = pad_left
        self.pad_right = pad_right

    def offset_left(self):
        return self.oob_left + self.pad_left

    def size(self):
        return self.pad_left + self.oob_left + self.extent + self.oob_right + self.pad_right


class EmptyDim(VectorDimContent):
    def __init__(self, extent, pad_left=0, pad_right=0, oob_right=0):
        super().__init__()
        self.extent = extent
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.oob_right = oob_right

    def offset_left(self):
        return self.pad_left

    def size(self):
        return self.pad_left + self.extent + self.oob_right + self.pad_right


class AbstractVector:
    def __init__(self, size: int, array=None):
        self.size = size
        if array is None:
            self.array = np.zeros((size,))

        else:
            self.array = array

    def __str__(self):
        return str(self.array)

    def create(self, size, array):
        pass

    def validate_operand(self, y):
        pass

class CiphertextVector(AbstractVector):
    def __init__(self, size: int, array: Ciphertext):
        super().__init__(size, array)

    def create(self, size, array):
        return CiphertextVector(size, array)

    def validate_operand(self, y):
        return isinstance(y, CiphertextVector) or isinstance(y, PlaintextVector)


class PlaintextVector(AbstractVector):
    def __init__(self, size: int, array: Plaintext):
        super().__init__(size, array)

    def create(self, size, array):
        return CiphertextVector(size, array)

    def validate_operand(self, y):
        return isinstance(y, CiphertextVector) or isinstance(y, PlaintextVector)


class NativeVector(AbstractVector):
    def __init__(self, size: int, array: np.ndarray):
        super().__init__(size, array)

    def create(self, size, array):
        return CiphertextVector(size, array)

    def validate_operand(self, y):
        return isinstance(y, NativeVector)


class AbstractArray:
    def __init__(self, size: int, extents: List[int], default):
        if len(extents) == 0:
            self.single = default

        else:
            self.map = {}
            coords = itertools.product(*map(lambda e: tuple(range(e)), extents))
            for coord in coords:
                self.map[coord] = default

    def set(self, coord: List[int], val):
        if len(coord) == 0:
            self.single = val

        else:
            self.map[tuple(coord)] = val

    def get(self, coord=[]):
        if len(coord) == 0:
            return self.single

        else:
            return self.map[tuple(coord)]

    def create_vector(self, size):
        pass

    def show(self):
        if hasattr(self, "single"):
            print(self.single)

        else:
            for coord, val in self.map.items():
                print("{} => {}", coord, val)


class CiphertextArray(AbstractArray):
    def __init__(self, size: int, extents: List[int], default: CiphertextVector):
        super().__init__(size, extents, default)

    def create_vector(self, size: int):
        return CiphertextVector(size)


class PlaintextArray(AbstractArray):
    def __init__(self, size: int, extents: List[int], default: PlaintextVector):
        super().__init__(size, extents, default)

    def create_vector(self, size: int):
        return PlaintextVector(size)


class NativeArray(AbstractArray):
    def __init__(self, size: int, extents: List[int], default: NativeVector):
        super().__init__(size, extents, default)

    def create_vector(self, size: int):
        return NativeVector(size)


class SEALWrapper:
    def __init__(self, seal, size, client_inputs={}, server_inputs={}):
        self.seal = seal
        self.size = size
        self.client_inputs = client_inputs
        self.server_inputs = server_inputs
        self.client_arrays = {}
        self.server_arrays = {}
        self.client_buffer = {}
        self.server_buffer = {}
        self.party = None
        self.server_start_time = 0
        self.server_exec_duration = 0

    def start_server_exec(self):
        self.server_start_time = time()

    def end_server_exec(self):
        self.server_exec_duration = (time() - self.server_start_time) * 1000

    def make_native_full(self, value):
        return NativeVector(self.size, np.full(self.size, value))

    def make_pt_full(self, value):
        return self.encode_vec(NativeVector(self.size, np.full(self.size, value)))

    def make_ct_full(self, value):
        return self.encrypt(self.encode_vec(NativeVector(self.size, np.full(self.size, value))))

    def set_party(self, party):
        self.party = party

    def client_input(self, name):
        assert(self.party == "client")
        self.client_arrays[(name, "")] = self.client_inputs[name]

    def server_input(self, name):
        assert(self.party == "server")
        self.server_arrays[(name, "")] = self.server_inputs[name]

    def output_ciphertext_vector(self, vec):
        noise = self.invariant_noise_budget(vec)
        decrypted = self.seal["decryptor"].decrypt(vec.array)
        decoded = self.seal["batch_encoder"].decode(decrypted)
        # with np.printoptions(threshold=1000):
        with np.printoptions(threshold=np.inf):
            print("output: ", decoded[0:self.size])
            print("noise budget: ", noise)

    def client_output(self, arr):
        assert(self.party == "client")

        if isinstance(arr, CiphertextArray):
            if hasattr(arr, "single"):
                self.output_ciphertext_vector(arr.single)

            else:
                for coord, val in arr.map.items():
                    print("coord ", coord)
                    self.output_ciphertext_vector(val)

        elif isinstance(arr, CiphertextVector):
            self.output_ciphertext_vector(arr)

    def client_send(self, name, arr):
        assert(self.party == "client")
        vec = arr.get([])
        assert(isinstance(vec, NativeVector))

        # encode into plaintext
        encoded = self.encode_vec(vec)

        # encrypt into ciphertext
        self.client_buffer[name] = self.encrypt(encoded)

    def server_recv(self, name):
        assert(self.party == "server")
        return self.vec_to_array(self.client_buffer[name])

    def server_send(self, name, arr):
        assert(self.party == "server")
        assert(isinstance(arr, CiphertextArray))
        self.server_buffer[name] = arr

    def client_recv(self, name):
        assert(self.party == "client")
        return self.server_buffer[name]

    def vec_to_array(self, vec):
        arr = None
        if isinstance(vec, NativeVector):
            arr = NativeArray(self.size, [], vec)

        elif isinstance(vec, CiphertextVector):
            arr = CiphertextArray(self.size, [], vec)

        elif isinstance(vec, PlaintextVector):
            arr = PlaintextArray(self.size, [], vec)

        else:
            raise Exception("unknown type")

        return arr

    def get_array(self, name, preprocess=None):
        preprocess_str = str(preprocess) if preprocess is not None else ""

        arrays = None
        if self.party == "server":
            arrays = self.server_arrays

        elif self.party == "client":
            arrays = self.client_arrays

        else:
            raise Exception("party not set")

        if not (name, preprocess_str) in arrays:
            arr = arrays[(name, "")]
            parr = preprocess.apply(arr)
            arrays[(name, preprocess_str)] = parr
            return parr

        else:
            return arrays[(name, preprocess_str)]

    def native_array(self, extents: List[List[int]], default=0):
        return NativeArray(self.size, extents, self.make_native_full(default))

    def ciphertext_array(self, extents, default=0):
        return CiphertextArray(self.size, extents, self.make_ct_full(default))

    def plaintext_array(self, extents, default=0):
        return PlaintextArray(self.size, extents, self.make_pt_full(default))

    def build_vector(self, name: str, preprocess: ArrayPreprocessing, src_offset: List[int], dims: List[VectorDimContent]) -> NativeArray:
        array = self.get_array(name, preprocess)
        if len(dims) == 0:
            npvec = np.full(self.size, array[tuple(src_offset)])
            vec = NativeVector(self.size, npvec)
            return self.vec_to_array(vec)

        else:
            dst_offset = []
            dst_shape = []
            stride_map = {}
            coords = []
            dst_size = 1
            for i, dim in enumerate(dims):
                dst_offset.append(dim.offset_left())
                dst_shape.append(dim.size())
                dst_size *= dim.size()
                coords.append(range(dim.extent))

                if isinstance(dim, FilledDim):
                    stride_map[i] = (dim.dim, dim.stride)

            dst = np.zeros(dst_shape, dtype=int)
            for coord in itertools.product(*coords):
                src_coord = src_offset[:]
                dst_coord = dst_offset[:]
                for i, coord in enumerate(coord):
                    dst_coord[i] += coord
                    if i in stride_map:
                        (src_dim, stride) = stride_map[i]
                        src_coord[src_dim] += coord * stride

                dst_coord_tup = tuple(dst_coord)
                src_coord_tup = tuple(src_coord)
                src_in_bounds = all(map(lambda t: t[0] > t[1], zip(array.shape, src_coord_tup)))
                if src_in_bounds:
                    dst[dst_coord_tup] = array[tuple(src_coord)]

                else:
                    dst[dst_coord_tup] = 0

            dst_flat = dst.reshape(dst_size)

            # should we actually repeat? or just pad with zeros?
            repeats = int(self.size / dst_size)
            if self.size % dst_size != 0:
                repeats += 1

            dst_vec = np.tile(dst_flat, repeats)[:self.size]
            return self.vec_to_array(NativeVector(self.size, dst_vec))


    def const(self, const: int):
        return self.vec_to_array(NativeVector(self.size, np.array(self.size * [const])))

    def mask(self, mask_vec: List[Tuple[int, int, int]]):
        mask_size = 1
        mask_acc = None
        clip = lambda val, zero, lo, hi, i: val if lo <= i and i <= hi else zero
        while len(mask_vec) > 0:
            extent, lo, hi = mask_vec.pop()
            mask_size = mask_size * extent
            if mask_acc is None:
                lst = [clip(1,0,lo,hi,i) for i in range(extent)]
                mask_acc = np.array(lst)

            else:
                lst = [clip(mask_acc,np.zeros(mask_acc.shape),lo,hi,i) for i in range(extent)]
                mask_acc = np.stack(lst)

        mask_flat = mask_acc.reshape(mask_size)
        repeats = int(self.size / mask_size)
        if self.size % mask_size != 0:
            repeats += 1

        mask_final = np.tile(mask_flat, repeats)[:self.size]
        return self.vec_to_array(NativeVector(self.size, mask_final))

    def set(self, arr, coord, val):
        arr.set(coord, val)

    def encode_vec(self, vec):
        assert(isinstance(vec, NativeVector))
        # BFV as 2xn vectors, so repeat array to encode
        encoded = self.seal["batch_encoder"].encode(np.tile(vec.array, 2))
        return PlaintextVector(vec.size, encoded)

    def encode(self, arr: NativeArray, coord: List[int]):
        arr.set(coord, self.encode_vec(arr.get(coord)))

    def encrypt(self, x: PlaintextVector):
        encrypted = self.seal["encryptor"].encrypt(x.array)
        return CiphertextVector(x.size, encrypted)

    def add(self, x: CiphertextVector, y: CiphertextVector):
        sum = self.seal["evaluator"].add(x.array, y.array)
        return CiphertextVector(x.size, sum)

    def add_plain(self, x: CiphertextVector, y: PlaintextVector):
        sum = self.seal["evaluator"].add_plain(x.array, y.array)
        return CiphertextVector(x.size, sum)

    def add_inplace(self, x: CiphertextVector, y: CiphertextVector):
        self.seal["evaluator"].add_inplace(x.array, y.array)

    def add_plain_inplace(self, x: CiphertextVector, y: PlaintextVector):
        self.seal["evaluator"].add_plain_inplace(x.array, y.array)

    def add_native(self, x: NativeVector, y: NativeVector):
        return NativeVector(x.size, x.array + y.array)

    def add_native_inplace(self, x: NativeVector, y: NativeVector):
        x.array = x.array + y.array

    def subtract(self, x: CiphertextVector, y: CiphertextVector):
        diff = self.seal["evaluator"].sub(x.array, y.array)
        return CiphertextVector(x.size, diff)

    def subtract_plain(self, x: CiphertextVector, y: PlaintextVector):
        diff = self.seal["evaluator"].sub_plain(x.array, y.array)
        return CiphertextVector(x.size, diff)

    def subtract_inplace(self, x: CiphertextVector, y: CiphertextVector):
        self.seal["evaluator"].sub_inplace(x.array, y.array)

    def subtract_plain_inplace(self, x: CiphertextVector, y: PlaintextVector):
        self.seal["evaluator"].sub_plain_inplace(x.array, y.array)

    def subtract_native(self, x: NativeVector, y: NativeVector):
        return NativeVector(x.size, x.array - y.array)

    def subtract_native_inplace(self, x: NativeVector, y: NativeVector):
        x.array = x.array - y.array

    def multiply(self, x: CiphertextVector, y: CiphertextVector):
        product = self.seal["evaluator"].multiply(x.array, y.array)
        return CiphertextVector(x.size, product)

    def multiply_plain(self, x: CiphertextVector, y: PlaintextVector):
        product = self.seal["evaluator"].multiply_plain(x.array, y.array)
        return CiphertextVector(x.size, product)

    def multiply_inplace(self, x: CiphertextVector, y: CiphertextVector):
        self.seal["evaluator"].multiply_inplace(x.array, y.array)

    def multiply_plain_inplace(self, x: CiphertextVector, y: PlaintextVector):
        self.seal["evaluator"].multiply_plain_inplace(x.array, y.array)

    def multiply_native(self, x: NativeVector, y: NativeVector):
        return NativeVector(x.size, x.array * y.array)

    def multiply_native_inplace(self, x: NativeVector, y: NativeVector):
        x.array = x.array * y.array

    def rotate_rows(self, amt: int, x: CiphertextVector):
        rotated = self.seal["evaluator"].rotate_rows(x.array, -amt, self.seal["galois_keys"])
        return CiphertextVector(x.size, rotated)

    def rotate_rows_inplace(self, amt: int, x: CiphertextVector):
        self.seal["evaluator"].rotate_rows_inplace(x.array, -amt, self.seal["galois_keys"])

    def rotate_rows_native(self, amt: int, x: NativeVector):
        rotated = x.array[[(i+amt) % self.size for i in range(self.size)]]
        return NativeVector(x.size, rotated)

    def rotate_rows_native_inplace(self, amt: int, x: NativeVector):
        x.array = x.array[[(i+amt) % self.size for i in range(self.size)]]

    def relinearize_inplace(self, x: CiphertextVector):
        self.seal["evaluator"].relinearize_inplace(x.array, self.seal["relin_keys"])

    def invariant_noise_budget(self, x: CiphertextVector):
        return self.seal["decryptor"].invariant_noise_budget(x.array)


### START GENERATED CODE
def client_pre(wrapper):
    wrapper.client_input("point")
    v_point_1 = wrapper.build_vector("point", None, [0], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    wrapper.client_send("v_point_1", v_point_1)

def client_post(wrapper):
    __out = wrapper.client_recv("__out")
    wrapper.client_output(__out)

def server(wrapper):
    wrapper.server_input("tests")
    v_tests_1 = wrapper.build_vector("tests", Roll(1,0), [0, 53], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_2 = wrapper.build_vector("tests", Roll(1,0), [0, 27], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_3 = wrapper.build_vector("tests", Roll(1,0), [0, 33], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_4 = wrapper.build_vector("tests", Roll(1,0), [0, 55], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_5 = wrapper.build_vector("tests", Roll(1,0), [0, 45], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_6 = wrapper.build_vector("tests", Roll(1,0), [0, 35], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_7 = wrapper.build_vector("tests", Roll(1,0), [0, 15], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_8 = wrapper.build_vector("tests", Roll(1,0), [0, 12], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_9 = wrapper.build_vector("tests", Roll(1,0), [0, 59], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_10 = wrapper.build_vector("tests", Roll(1,0), [0, 7], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_11 = wrapper.build_vector("tests", Roll(1,0), [0, 17], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_12 = wrapper.build_vector("tests", Roll(1,0), [0, 25], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_13 = wrapper.build_vector("tests", Roll(1,0), [0, 34], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_14 = wrapper.build_vector("tests", Roll(1,0), [0, 0], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_15 = wrapper.build_vector("tests", Roll(1,0), [0, 19], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_16 = wrapper.build_vector("tests", Roll(1,0), [0, 62], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_17 = wrapper.build_vector("tests", Roll(1,0), [0, 4], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_18 = wrapper.build_vector("tests", Roll(1,0), [0, 32], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_19 = wrapper.build_vector("tests", Roll(1,0), [0, 46], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_20 = wrapper.build_vector("tests", Roll(1,0), [0, 11], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_21 = wrapper.build_vector("tests", Roll(1,0), [0, 16], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_22 = wrapper.build_vector("tests", Roll(1,0), [0, 36], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_23 = wrapper.build_vector("tests", Roll(1,0), [0, 47], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_24 = wrapper.build_vector("tests", Roll(1,0), [0, 6], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_25 = wrapper.build_vector("tests", Roll(1,0), [0, 50], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_26 = wrapper.build_vector("tests", Roll(1,0), [0, 29], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_27 = wrapper.build_vector("tests", Roll(1,0), [0, 56], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_28 = wrapper.build_vector("tests", Roll(1,0), [0, 61], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_29 = wrapper.build_vector("tests", Roll(1,0), [0, 13], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_30 = wrapper.build_vector("tests", Roll(1,0), [0, 21], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_31 = wrapper.build_vector("tests", Roll(1,0), [0, 30], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_32 = wrapper.build_vector("tests", Roll(1,0), [0, 28], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_33 = wrapper.build_vector("tests", Roll(1,0), [0, 54], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_34 = wrapper.build_vector("tests", Roll(1,0), [0, 20], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_35 = wrapper.build_vector("tests", Roll(1,0), [0, 37], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_36 = wrapper.build_vector("tests", Roll(1,0), [0, 57], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_37 = wrapper.build_vector("tests", Roll(1,0), [0, 41], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_38 = wrapper.build_vector("tests", Roll(1,0), [0, 39], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_39 = wrapper.build_vector("tests", Roll(1,0), [0, 43], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_40 = wrapper.build_vector("tests", Roll(1,0), [0, 42], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_41 = wrapper.build_vector("tests", Roll(1,0), [0, 60], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_42 = wrapper.build_vector("tests", Roll(1,0), [0, 58], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_43 = wrapper.build_vector("tests", Roll(1,0), [0, 10], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_44 = wrapper.build_vector("tests", Roll(1,0), [0, 1], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_45 = wrapper.build_vector("tests", Roll(1,0), [0, 63], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_46 = wrapper.build_vector("tests", Roll(1,0), [0, 5], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_47 = wrapper.build_vector("tests", Roll(1,0), [0, 22], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_48 = wrapper.build_vector("tests", Roll(1,0), [0, 23], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_49 = wrapper.build_vector("tests", Roll(1,0), [0, 31], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_50 = wrapper.build_vector("tests", Roll(1,0), [0, 51], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_51 = wrapper.build_vector("tests", Roll(1,0), [0, 52], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_52 = wrapper.build_vector("tests", Roll(1,0), [0, 3], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_53 = wrapper.build_vector("tests", Roll(1,0), [0, 40], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_54 = wrapper.build_vector("tests", Roll(1,0), [0, 9], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_55 = wrapper.build_vector("tests", Roll(1,0), [0, 48], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_56 = wrapper.build_vector("tests", Roll(1,0), [0, 49], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_57 = wrapper.build_vector("tests", Roll(1,0), [0, 24], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_58 = wrapper.build_vector("tests", Roll(1,0), [0, 26], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_59 = wrapper.build_vector("tests", Roll(1,0), [0, 2], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_60 = wrapper.build_vector("tests", Roll(1,0), [0, 8], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_61 = wrapper.build_vector("tests", Roll(1,0), [0, 44], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_62 = wrapper.build_vector("tests", Roll(1,0), [0, 18], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_63 = wrapper.build_vector("tests", Roll(1,0), [0, 38], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_tests_64 = wrapper.build_vector("tests", Roll(1,0), [0, 14], [FilledDim(0, 64, 1, 0, 0, 0, 0)])
    v_point_1 = wrapper.server_recv("v_point_1")
    const_neg1 = wrapper.const(-1)
    wrapper.start_server_exec()
    wrapper.encode(v_tests_30, [])
    wrapper.encode(v_tests_50, [])
    wrapper.encode(v_tests_41, [])
    wrapper.encode(v_tests_52, [])
    wrapper.encode(v_tests_10, [])
    wrapper.encode(v_tests_56, [])
    wrapper.encode(v_tests_43, [])
    wrapper.encode(v_tests_57, [])
    wrapper.encode(v_tests_27, [])
    wrapper.encode(v_tests_26, [])
    wrapper.encode(v_tests_23, [])
    wrapper.encode(v_tests_24, [])
    wrapper.encode(v_tests_8, [])
    wrapper.encode(v_tests_61, [])
    wrapper.encode(v_tests_3, [])
    wrapper.encode(v_tests_12, [])
    wrapper.encode(v_tests_5, [])
    wrapper.encode(v_tests_2, [])
    wrapper.encode(v_tests_51, [])
    wrapper.encode(v_tests_28, [])
    wrapper.encode(v_tests_13, [])
    wrapper.encode(v_tests_15, [])
    wrapper.encode(v_tests_33, [])
    wrapper.encode(v_tests_29, [])
    wrapper.encode(v_tests_60, [])
    wrapper.encode(v_tests_46, [])
    wrapper.encode(v_tests_47, [])
    wrapper.encode(v_tests_9, [])
    wrapper.encode(v_tests_42, [])
    wrapper.encode(v_tests_59, [])
    wrapper.encode(v_tests_31, [])
    wrapper.encode(v_tests_21, [])
    wrapper.encode(v_tests_55, [])
    wrapper.encode(v_tests_25, [])
    wrapper.encode(v_tests_16, [])
    wrapper.encode(v_tests_64, [])
    wrapper.encode(v_tests_22, [])
    wrapper.encode(v_tests_54, [])
    wrapper.encode(v_tests_4, [])
    wrapper.encode(v_tests_45, [])
    wrapper.encode(v_tests_14, [])
    wrapper.encode(v_tests_11, [])
    wrapper.encode(v_tests_1, [])
    wrapper.encode(v_tests_18, [])
    wrapper.encode(v_tests_32, [])
    wrapper.encode(v_tests_20, [])
    wrapper.encode(v_tests_17, [])
    wrapper.encode(v_tests_53, [])
    wrapper.encode(v_tests_37, [])
    wrapper.encode(v_tests_63, [])
    wrapper.encode(v_tests_7, [])
    wrapper.encode(v_tests_48, [])
    wrapper.encode(v_tests_49, [])
    wrapper.encode(v_tests_38, [])
    wrapper.encode(v_tests_19, [])
    wrapper.encode(v_tests_34, [])
    wrapper.encode(v_tests_62, [])
    wrapper.encode(v_tests_58, [])
    wrapper.encode(v_tests_6, [])
    wrapper.encode(v_tests_39, [])
    wrapper.encode(v_tests_40, [])
    wrapper.encode(v_tests_36, [])
    wrapper.encode(v_tests_44, [])
    wrapper.encode(v_tests_35, [])
    wrapper.encode(const_neg1, [])
    pt1 = wrapper.plaintext_array([64], 0)
    wrapper.set(pt1, [0], v_tests_14.get())
    wrapper.set(pt1, [1], v_tests_44.get())
    wrapper.set(pt1, [2], v_tests_59.get())
    wrapper.set(pt1, [3], v_tests_52.get())
    wrapper.set(pt1, [4], v_tests_17.get())
    wrapper.set(pt1, [5], v_tests_46.get())
    wrapper.set(pt1, [6], v_tests_24.get())
    wrapper.set(pt1, [7], v_tests_10.get())
    wrapper.set(pt1, [8], v_tests_60.get())
    wrapper.set(pt1, [9], v_tests_54.get())
    wrapper.set(pt1, [10], v_tests_43.get())
    wrapper.set(pt1, [11], v_tests_20.get())
    wrapper.set(pt1, [12], v_tests_8.get())
    wrapper.set(pt1, [13], v_tests_29.get())
    wrapper.set(pt1, [14], v_tests_64.get())
    wrapper.set(pt1, [15], v_tests_7.get())
    wrapper.set(pt1, [16], v_tests_21.get())
    wrapper.set(pt1, [17], v_tests_11.get())
    wrapper.set(pt1, [18], v_tests_62.get())
    wrapper.set(pt1, [19], v_tests_15.get())
    wrapper.set(pt1, [20], v_tests_34.get())
    wrapper.set(pt1, [21], v_tests_30.get())
    wrapper.set(pt1, [22], v_tests_47.get())
    wrapper.set(pt1, [23], v_tests_48.get())
    wrapper.set(pt1, [24], v_tests_57.get())
    wrapper.set(pt1, [25], v_tests_12.get())
    wrapper.set(pt1, [26], v_tests_58.get())
    wrapper.set(pt1, [27], v_tests_2.get())
    wrapper.set(pt1, [28], v_tests_32.get())
    wrapper.set(pt1, [29], v_tests_26.get())
    wrapper.set(pt1, [30], v_tests_31.get())
    wrapper.set(pt1, [31], v_tests_49.get())
    wrapper.set(pt1, [32], v_tests_18.get())
    wrapper.set(pt1, [33], v_tests_3.get())
    wrapper.set(pt1, [34], v_tests_13.get())
    wrapper.set(pt1, [35], v_tests_6.get())
    wrapper.set(pt1, [36], v_tests_22.get())
    wrapper.set(pt1, [37], v_tests_35.get())
    wrapper.set(pt1, [38], v_tests_63.get())
    wrapper.set(pt1, [39], v_tests_38.get())
    wrapper.set(pt1, [40], v_tests_53.get())
    wrapper.set(pt1, [41], v_tests_37.get())
    wrapper.set(pt1, [42], v_tests_40.get())
    wrapper.set(pt1, [43], v_tests_39.get())
    wrapper.set(pt1, [44], v_tests_61.get())
    wrapper.set(pt1, [45], v_tests_5.get())
    wrapper.set(pt1, [46], v_tests_19.get())
    wrapper.set(pt1, [47], v_tests_23.get())
    wrapper.set(pt1, [48], v_tests_55.get())
    wrapper.set(pt1, [49], v_tests_56.get())
    wrapper.set(pt1, [50], v_tests_25.get())
    wrapper.set(pt1, [51], v_tests_50.get())
    wrapper.set(pt1, [52], v_tests_51.get())
    wrapper.set(pt1, [53], v_tests_1.get())
    wrapper.set(pt1, [54], v_tests_33.get())
    wrapper.set(pt1, [55], v_tests_4.get())
    wrapper.set(pt1, [56], v_tests_27.get())
    wrapper.set(pt1, [57], v_tests_36.get())
    wrapper.set(pt1, [58], v_tests_42.get())
    wrapper.set(pt1, [59], v_tests_9.get())
    wrapper.set(pt1, [60], v_tests_41.get())
    wrapper.set(pt1, [61], v_tests_28.get())
    wrapper.set(pt1, [62], v_tests_16.get())
    wrapper.set(pt1, [63], v_tests_45.get())
    __out = wrapper.ciphertext_array([], 0)
    __reduce_1 = wrapper.ciphertext_array([], 0)
    for i1 in range(64):
        instr1 = wrapper.rotate_rows((0 + i1), v_point_1.get())
        wrapper.subtract_plain_inplace(instr1, pt1.get([i1]))
        wrapper.multiply_inplace(instr1, instr1)
        wrapper.relinearize_inplace(instr1)
        wrapper.add_inplace(instr1, __reduce_1.get())
        wrapper.set(__reduce_1, [], instr1)
    
    wrapper.set(__out, [], __reduce_1.get())
    wrapper.end_server_exec()
    wrapper.server_send("__out", __out)

### END GENERATED CODE

init_start_time = time()

input_str = None
client_inputs = {}
server_inputs = {}
if len(sys.argv) >= 2:
    with open(sys.argv[1]) as f:
        input_str = f.read()

else:
    input_str = sys.stdin.read()

inputs = json.loads(input_str)
for key, val in inputs["client"].items():
    client_inputs[key] = np.array(val)

for key, val in inputs["server"].items():
    server_inputs[key] = np.array(val)

vec_size = 2048

seal = {}
seal["parms"] = EncryptionParameters(scheme_type.bfv)

poly_modulus_degree = vec_size * 2
seal["parms"].set_poly_modulus_degree(poly_modulus_degree)
seal["parms"].set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
seal["parms"].set_plain_modulus(PlainModulus.Batching(poly_modulus_degree, 20))
seal["context"] = SEALContext(seal["parms"])
seal["keygen"] = KeyGenerator(seal["context"])
seal["secret_key"] = seal["keygen"].secret_key()
seal["public_key"] = seal["keygen"].create_public_key()
seal["relin_keys"] = seal["keygen"].create_relin_keys()
seal["galois_keys"] = seal["keygen"].create_galois_keys()

seal["encryptor"] = Encryptor(seal["context"], seal["public_key"])
seal["evaluator"] = Evaluator(seal["context"])
seal["decryptor"] = Decryptor(seal["context"], seal["secret_key"])

seal["batch_encoder"] = BatchEncoder(seal["context"])

wrapper = SEALWrapper(seal, vec_size, client_inputs, server_inputs)

init_end_time = time()
print("init duration: {}ms".format((init_end_time - init_start_time)*1000))

wrapper.set_party("client")
client_pre(wrapper)

wrapper.set_party("server")
server(wrapper)

wrapper.set_party("client")
client_post(wrapper)

print("exec duration: {:.0f}ms".format(wrapper.server_exec_duration))