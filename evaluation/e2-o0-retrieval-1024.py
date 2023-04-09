import sys
import itertools
import json
import numpy as np
from time import time

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
        for ind in range(arr.shape[self.dim_i]):
            i_slice = np.take(arr, ind, axis=self.dim_i)
            roll_dim = self.dim_j if self.dim_j < self.dim_i else self.dim_j - 1
            perm_i_slice = np.roll(i_slice, -ind, axis=roll_dim)
            slices.append(perm_i_slice)

        return np.stack(slices, axis=self.dim_i)

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
    def __init__(self, size: int, extents: list[int], default):
        if len(extents) == 0:
            self.single = default

        else:
            self.map = {}
            coords = itertools.product(*map(lambda e: tuple(range(e)), extents))
            for coord in coords:
                self.map[coord] = default

    def set(self, coord: list[int], val):
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
        if self.single is not None:
            print(self.single)

        else:
            for coord, val in self.map.items():
                print("{} => {}", coord, val)


class CiphertextArray(AbstractArray):
    def __init__(self, size: int, extents: list[int], default: CiphertextVector):
        super().__init__(size, extents, default)

    def create_vector(self, size: int):
        return CiphertextVector(size)


class PlaintextArray(AbstractArray):
    def __init__(self, size: int, extents: list[int], default: PlaintextVector):
        super().__init__(size, extents, default)

    def create_vector(self, size: int):
        return PlaintextVector(size)


class NativeArray(AbstractArray):
    def __init__(self, size: int, extents: list[int], default: NativeVector):
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
            if arr.single is not None:
                self.output_ciphertext_vector(arr.single)

            else:
                for coord, val in self.map.items():
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

    def native_array(self, extents: list[list[int]], default=0):
        return NativeArray(self.size, extents, self.make_native_full())

    def ciphertext_array(self, extents, default=0):
        return CiphertextArray(self.size, extents, self.make_ct_full(default))

    def plaintext_array(self, extents, default=0):
        return PlaintextArray(self.size, extents, self.make_pt_zeros())

    def build_vector(self, name: str, preprocess: ArrayPreprocessing, src_offset: list[int], dims: list[VectorDimContent]) -> NativeArray:
        array = self.get_array(name, preprocess)
        if len(dims) == 0:
            npvec = np.zeros((self.size,))
            npvec[0] = array[tuple(src_offset)]
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

    def mask(self, mask: list[(int, int, int)]):
        raise NotImplemented

    def set(self, arr, coord, val):
        arr.set(coord, val)

    def encode_vec(self, vec):
        assert(isinstance(vec, NativeVector))
        # BFV as 2xn vectors, so repeat array to encode
        encoded = self.seal["batch_encoder"].encode(np.tile(vec.array, 2))
        return PlaintextVector(vec.size, encoded)

    def encode(self, arr: NativeArray, coord: list[int]):
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
    wrapper.client_input("values")
    wrapper.client_input("query")
    wrapper.client_input("keys")
    v_keys_1 = wrapper.build_vector("keys", None, [0, 0], [FilledDim(1, 5, 2, 0, 3, 0, 0), FilledDim(0, 1024, 1, 0, 0, 0, 0)])
    wrapper.client_send("v_keys_1", v_keys_1)
    v_keys_2 = wrapper.build_vector("keys", None, [0, 1], [FilledDim(1, 5, 2, 0, 3, 0, 0), FilledDim(0, 1024, 1, 0, 0, 0, 0)])
    wrapper.client_send("v_keys_2", v_keys_2)
    v_query_1 = wrapper.build_vector("query", None, [1], [FilledDim(0, 5, 2, 0, 3, 0, 0), EmptyDim(1024, 0, 0, 0)])
    wrapper.client_send("v_query_1", v_query_1)
    v_query_2 = wrapper.build_vector("query", None, [0], [FilledDim(0, 5, 2, 0, 3, 0, 0), EmptyDim(1024, 0, 0, 0)])
    wrapper.client_send("v_query_2", v_query_2)
    v_values_1 = wrapper.build_vector("values", None, [0], [FilledDim(0, 1024, 1, 0, 0, 0, 0)])
    wrapper.client_send("v_values_1", v_values_1)

def client_post(wrapper):
    __out = wrapper.client_recv("__out")
    wrapper.client_output(__out)

def server(wrapper):
    v_keys_1 = wrapper.server_recv("v_keys_1")
    v_keys_2 = wrapper.server_recv("v_keys_2")
    v_query_1 = wrapper.server_recv("v_query_1")
    v_query_2 = wrapper.server_recv("v_query_2")
    v_values_1 = wrapper.server_recv("v_values_1")
    const_1 = wrapper.const(1)
    const_neg1 = wrapper.const(-1)
    wrapper.start_server_exec()
    wrapper.encode(const_1, [])
    wrapper.encode(const_neg1, [])
    ct2 = wrapper.ciphertext_array([2])
    wrapper.set(ct2, [0], v_keys_1.get())
    wrapper.set(ct2, [1], v_keys_2.get())
    ct1 = wrapper.ciphertext_array([2])
    wrapper.set(ct1, [0], v_query_2.get())
    wrapper.set(ct1, [1], v_query_1.get())
    mask = wrapper.ciphertext_array([])
    __reduce_1 = wrapper.ciphertext_array([], 1)
    for i9i in range(2):
        instr3 = wrapper.subtract(ct1.get([i9i]), ct2.get([i9i]))
        wrapper.multiply_inplace(instr3, instr3)
        wrapper.relinearize_inplace(instr3)
        wrapper.multiply_plain_inplace(instr3, const_neg1.get())
        wrapper.add_plain_inplace(instr3, const_1.get())
        wrapper.multiply_inplace(instr3, __reduce_1.get())
        wrapper.relinearize_inplace(instr3)
        wrapper.set(__reduce_1, [], instr3)
    
    instr9 = wrapper.rotate_rows(-4096, __reduce_1.get())
    wrapper.multiply_inplace(instr9, __reduce_1.get())
    wrapper.relinearize_inplace(instr9)
    instr11 = wrapper.rotate_rows(-2048, instr9)
    wrapper.multiply_inplace(instr9, instr11)
    wrapper.relinearize_inplace(instr9)
    instr13 = wrapper.rotate_rows(-1024, instr9)
    wrapper.multiply_inplace(instr9, instr13)
    wrapper.relinearize_inplace(instr9)
    wrapper.set(mask, [], instr9)
    __out = wrapper.ciphertext_array([])
    instr17 = wrapper.multiply(v_values_1.get(), mask.get())
    wrapper.relinearize_inplace(instr17)
    instr18 = wrapper.rotate_rows(-512, instr17)
    wrapper.add_inplace(instr17, instr18)
    instr20 = wrapper.rotate_rows(-256, instr17)
    wrapper.add_inplace(instr17, instr20)
    instr22 = wrapper.rotate_rows(-128, instr17)
    wrapper.add_inplace(instr17, instr22)
    instr24 = wrapper.rotate_rows(-64, instr17)
    wrapper.add_inplace(instr17, instr24)
    instr26 = wrapper.rotate_rows(-32, instr17)
    wrapper.add_inplace(instr17, instr26)
    instr28 = wrapper.rotate_rows(-16, instr17)
    wrapper.add_inplace(instr17, instr28)
    instr30 = wrapper.rotate_rows(-8, instr17)
    wrapper.add_inplace(instr17, instr30)
    instr32 = wrapper.rotate_rows(-4, instr17)
    wrapper.add_inplace(instr17, instr32)
    instr34 = wrapper.rotate_rows(-2, instr17)
    wrapper.add_inplace(instr17, instr34)
    instr36 = wrapper.rotate_rows(-1, instr17)
    wrapper.add_inplace(instr17, instr36)
    wrapper.set(__out, [], instr17)
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
