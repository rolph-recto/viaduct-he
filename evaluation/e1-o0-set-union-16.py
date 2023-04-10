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
        rotated = self.seal["evaluator"].rotate_rows(x.array, amt, self.seal["galois_keys"])
        return CiphertextVector(x.size, rotated)

    def rotate_rows_inplace(self, amt: int, x: CiphertextVector):
        self.seal["evaluator"].rotate_rows_inplace(x.array, amt, self.seal["galois_keys"])

    def rotate_rows_native(self, amt: int, x: NativeVector):
        rotated = x.array[[(i-amt) % self.size for i in range(self.size)]]
        return NativeVector(x.size, rotated)

    def rotate_rows_native_inplace(self, amt: int, x: NativeVector):
        x.array = x.array[[(i-amt) % self.size for i in range(self.size)]]

    def relinearize_inplace(self, x: CiphertextVector):
        self.seal["evaluator"].relinearize_inplace(x.array, self.seal["relin_keys"])

    def invariant_noise_budget(self, x: CiphertextVector):
        return self.seal["decryptor"].invariant_noise_budget(x.array)


### START GENERATED CODE
def client_pre(wrapper):
    wrapper.client_input("a_id")
    wrapper.client_input("a_data")
    wrapper.client_input("b_id")
    wrapper.client_input("b_data")
    v_a_id_1 = wrapper.build_vector("a_id", None, [0, 0], [FilledDim(0, 16, 1, 0, 0, 0, 0), FilledDim(1, 4, 1, 0, 0, 0, 0), EmptyDim(16, 0, 0, 0)])
    wrapper.client_send("v_a_id_1", v_a_id_1)
    v_b_id_1 = wrapper.build_vector("b_id", None, [0, 0], [EmptyDim(16, 0, 0, 0), FilledDim(1, 4, 1, 0, 0, 0, 0), FilledDim(0, 16, 1, 0, 0, 0, 0)])
    wrapper.client_send("v_b_id_1", v_b_id_1)
    v_b_data_1 = wrapper.build_vector("b_data", None, [0], [FilledDim(0, 16, 1, 0, 0, 0, 0)])
    wrapper.client_send("v_b_data_1", v_b_data_1)
    v_a_data_1 = wrapper.build_vector("a_data", None, [0], [FilledDim(0, 16, 1, 0, 0, 0, 0)])
    wrapper.client_send("v_a_data_1", v_a_data_1)

def client_post(wrapper):
    __out = wrapper.client_recv("__out")
    wrapper.client_output(__out)

def server(wrapper):
    v_a_id_1 = wrapper.server_recv("v_a_id_1")
    v_b_id_1 = wrapper.server_recv("v_b_id_1")
    v_b_data_1 = wrapper.server_recv("v_b_data_1")
    v_a_data_1 = wrapper.server_recv("v_a_data_1")
    const_1 = wrapper.const(1)
    const_neg1 = wrapper.const(-1)
    wrapper.start_server_exec()
    wrapper.encode(const_1, [])
    wrapper.encode(const_neg1, [])
    a_sum = wrapper.ciphertext_array([], 0)
    instr1 = wrapper.rotate_rows(-8, v_a_data_1.get())
    wrapper.add_inplace(instr1, v_a_data_1.get())
    instr3 = wrapper.rotate_rows(-4, instr1)
    wrapper.add_inplace(instr1, instr3)
    instr5 = wrapper.rotate_rows(-2, instr1)
    wrapper.add_inplace(instr1, instr5)
    instr7 = wrapper.rotate_rows(-1, instr1)
    wrapper.add_inplace(instr1, instr7)
    wrapper.set(a_sum, [], instr1)
    b_sum = wrapper.ciphertext_array([], 0)
    instr13 = wrapper.subtract(v_a_id_1.get(), v_b_id_1.get())
    wrapper.multiply_inplace(instr13, instr13)
    wrapper.relinearize_inplace(instr13)
    wrapper.multiply_plain_inplace(instr13, const_neg1.get())
    wrapper.add_plain_inplace(instr13, const_1.get())
    instr17 = wrapper.rotate_rows(-32, instr13)
    wrapper.multiply_inplace(instr13, instr17)
    wrapper.relinearize_inplace(instr13)
    instr19 = wrapper.rotate_rows(-16, instr13)
    wrapper.multiply_inplace(instr13, instr19)
    wrapper.relinearize_inplace(instr13)
    wrapper.multiply_plain_inplace(instr13, const_neg1.get())
    wrapper.add_plain_inplace(instr13, const_1.get())
    instr23 = wrapper.rotate_rows(-512, instr13)
    wrapper.multiply_inplace(instr13, instr23)
    wrapper.relinearize_inplace(instr13)
    instr25 = wrapper.rotate_rows(-256, instr13)
    wrapper.multiply_inplace(instr13, instr25)
    wrapper.relinearize_inplace(instr13)
    instr27 = wrapper.rotate_rows(-128, instr13)
    wrapper.multiply_inplace(instr13, instr27)
    wrapper.relinearize_inplace(instr13)
    instr29 = wrapper.rotate_rows(-64, instr13)
    wrapper.multiply_inplace(instr13, instr29)
    wrapper.relinearize_inplace(instr13)
    wrapper.multiply_inplace(instr13, v_b_data_1.get())
    wrapper.relinearize_inplace(instr13)
    instr32 = wrapper.rotate_rows(-8, instr13)
    wrapper.add_inplace(instr13, instr32)
    instr34 = wrapper.rotate_rows(-4, instr13)
    wrapper.add_inplace(instr13, instr34)
    instr36 = wrapper.rotate_rows(-2, instr13)
    wrapper.add_inplace(instr13, instr36)
    instr38 = wrapper.rotate_rows(-1, instr13)
    wrapper.add_inplace(instr13, instr38)
    wrapper.set(b_sum, [], instr13)
    __out = wrapper.ciphertext_array([], 0)
    instr42 = wrapper.add(a_sum.get(), b_sum.get())
    wrapper.set(__out, [], instr42)
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

vec_size = 8192

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