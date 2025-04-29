import numpy as np
import math
from .mask_const import *
from .stencil import get_id2stage, get_stages, get_num_halo_points

def preprocess_sptrsv(data, tile_x, tile_y, stencil_type, dims, schedule_scheme):
    # A_valid is a flag to indicate when the pipeline should insert a bubble
    matrix_value, A_valid = preprocessPGC_sptrsv(data["size"], data['A'], tile_x, tile_y, stencil_type, dims, schedule_scheme)

    num_tile_x = math.ceil(data["size"][0] / tile_x)
    num_tile_y = math.ceil(data["size"][1] / tile_y)
    sche_dict, sche_seq = get_sche_seq(num_tile_x, num_tile_y, schedule_scheme)

    matrix_diag, right_hand_side, vec_index = preprocess_domain_data_sptrsv(data, tile_x, tile_y, sche_seq)
    halo_x, halo_y, b_valid = preprocess_halo_data_sptrsv(data, tile_x, tile_y, stencil_type, dims, sche_seq, sche_dict)

    data = {
        "size": data["size"], "A": matrix_value,
        "diag_A": matrix_diag, "b": right_hand_side,
        "b_valid": b_valid, "ijk": vec_index,
        "A_valid": A_valid, "halo_x": halo_x,
        "halo_y": halo_y, "x": np.zeros(data["size"], dtype=object)
    }
    return data


def calc_bubbles(stages, diag_len):
    return stages - 1
    # Insert at most stage - 1 bubbles
    # return max(0, min(stages - 1, stages + 20 - diag_len))

def get_bubbles(num_tile_x, num_tile_y, stages, schedule_scheme):
    if schedule_scheme == "wavefront":
        return sum([calc_bubbles(stages, min(num_tile_x, d + 1) - max(0, d + 1 - num_tile_y)) for d in range(num_tile_x + num_tile_y - 1)])
    else:
        return num_tile_x * num_tile_y * (stages - 1)

def preprocessPGC_sptrsv(size, data_A, tile_x, tile_y, stencil_type, dims, schedule_scheme):
    x, y, z = size
    n = x * y * z
    num_tile_x = math.ceil(x / tile_x)
    num_tile_y = math.ceil(y / tile_y)
    num_tiles = num_tile_x * num_tile_y
    stencil_length = data_A.shape[-1]
    stages = get_stages(stencil_type, dims, "sptrsv")
    id2stage = get_id2stage(dims, stencil_type, "sptrsv")

    bubbles = get_bubbles(num_tile_x, num_tile_y, stages, schedule_scheme)
    dim_shape_A = (num_tiles * z + bubbles, tile_x, tile_y, stencil_length)
    matrix_value = np.zeros(dim_shape_A)
    A_valid = np.ones(num_tiles * z + bubbles, dtype=np.bool)

    if schedule_scheme == "wavefront":
        dim_0 = 0
        for d in range(num_tile_y + num_tile_x - 1):
            diag_len = min(num_tile_x, d + 1) - max(0, d + 1 - num_tile_y)
            for out_i in range(max(0, d + 1 - num_tile_y), min(num_tile_x, d + 1)):
                out_j = d - out_i
                for k in range(z):
                    for in_i in range(tile_x):
                        for in_j in range(tile_y):
                            total_i = out_i * tile_x + in_i
                            total_j = out_j * tile_y + in_j
                            addr = total_i * y * z + total_j * z + k
                            if addr < n:
                                for l in range(stencil_length):
                                    assert(dim_0 + id2stage[l] < dim_shape_A[0])
                                    matrix_value[dim_0 + id2stage[l]][in_i][in_j][l] = data_A[addr][l]
                    dim_0 += 1

            bubbles = calc_bubbles(stages, diag_len)
            for i in range(bubbles):
                A_valid[dim_0 + i] = False
            dim_0 += bubbles
    else:
        dim_0 = 0
        for out_i in range(num_tile_x):
            for out_j in range(num_tile_y):
                for k in range(z):
                    for in_i in range(tile_x):
                        for in_j in range(tile_y):
                            total_i = out_i * tile_x + in_i
                            total_j = out_j * tile_y + in_j
                            addr = total_i * y * z + total_j * z + k
                            if addr < n:
                                for l in range(stencil_length):
                                    assert(dim_0 + id2stage[l] < dim_shape_A[0])
                                    matrix_value[dim_0 + id2stage[l]][in_i][in_j][l] = data_A[addr][l]
                    dim_0 += 1

                bubbles = stages - 1
                for i in range(bubbles):
                    A_valid[dim_0 + i] = False
                dim_0 += bubbles

    return matrix_value, A_valid

def get_sche_seq(num_tile_x, num_tile_y, schedule_scheme):
    sche_dict = {}
    sche_list = []
    idx = 0
    if schedule_scheme == "wavefront":
        for d in range(num_tile_x + num_tile_y - 1):
            for out_i in range(max(0, d + 1 - num_tile_y), min(num_tile_x, d + 1)):
                out_j = d - out_i
                sche_dict[(out_i, out_j)] = idx
                sche_list.append((out_i, out_j))
                idx += 1
    else:
        for out_i in range(num_tile_x):
            for out_j in range(num_tile_y):
                sche_dict[(out_i, out_j)] = idx
                sche_list.append((out_i, out_j))
                idx += 1

    return sche_dict, sche_list

def preprocess_halo_data_sptrsv(data, tile_x, tile_y, stencil_type, dims, sche_seq, sche_dict):
    x, y, z = data["size"]
    num_tile_x = math.ceil(x / tile_x)
    num_tile_y = math.ceil(y / tile_y)
    num_tiles = num_tile_x * num_tile_y
    dim_shape = (num_tiles * z, tile_x, tile_y)
    # padding for halo data
    padd_x, padd_y = get_num_halo_points(stencil_type, tile_x, tile_y, "sptrsv")
    halo_x = np.zeros((num_tiles * z, padd_x), dtype=object)
    halo_y = np.zeros((num_tiles * z, padd_y), dtype=object)
    # check the update status of b
    b_valid = np.zeros(dim_shape, dtype=np.int16)

    tile_idx = 0
    for out_i, out_j in sche_seq:
        for k in range(z):
            dim_0 = tile_idx * z + k
            if stencil_type == 0: # Star7P
                # Star7P: padd_x = padd_y = base
                for p in range(padd_x):
                    halo_tile_x = out_i + 1
                    halo_tile_y = out_j if p < tile_y else out_j + 1

                    if halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                        halo_tile_idx = sche_dict[(halo_tile_x, halo_tile_y)]
                        halo_dim_0 = halo_tile_idx * z + k
                    else:
                        halo_dim_0 = -1

                    halo_x[dim_0][p] = (halo_dim_0, 0, p % tile_y)
                    if halo_dim_0 >= 0:
                        b_valid[halo_x[dim_0][p]] |= OUT_I_M # out_i

                for p in range(padd_y):
                    halo_tile_x = out_i
                    halo_tile_y = out_j + 1

                    if halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                        halo_tile_idx = sche_dict[(halo_tile_x, halo_tile_y)]
                        halo_dim_0 = halo_tile_idx * z + k
                    else:
                        halo_dim_0 = -1
                    halo_y[dim_0][p] = (halo_dim_0, p % tile_x, 0)
                    if halo_dim_0 >= 0:
                        b_valid[halo_y[dim_0][p]] |= OUT_J_M # out_j

            elif stencil_type == 1: # Star13P
                # in and agg alternate mapping halo_x[0]->in, halo_x[1]->agg, ...
                # padd_x = padd_y = 2 * base
                for p in range(padd_x):
                    halo_tile_x = out_i + 1
                    halo_tile_y = out_j

                    if halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                        halo_tile_idx = sche_dict[(halo_tile_x, halo_tile_y)]
                        halo_dim_0 = halo_tile_idx * z + k
                    else:
                        halo_dim_0 = -1

                    halo_x[dim_0][p] = (halo_dim_0, p % 2, (p // 2) % tile_y)
                    if halo_dim_0 >= 0:
                        if p % 2 == 0:
                            b_valid[halo_x[dim_0][p]] |= OUT_I_M # out_i
                        else:
                            b_valid[halo_x[dim_0][p]] |= AGG_I_M # agg_i

                for p in range(padd_y):
                    halo_tile_x = out_i
                    halo_tile_y = out_j + 1

                    if halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                        halo_tile_idx = sche_dict[(halo_tile_x, halo_tile_y)]
                        halo_dim_0 = halo_tile_idx * z + k
                    else:
                        halo_dim_0 = -1

                    halo_y[dim_0][p] = (halo_dim_0, (p // 2) % tile_x, p % 2)
                    if halo_dim_0 >= 0:
                        if p % 2 == 0:
                            b_valid[halo_y[dim_0][p]] |= OUT_J_M # out_j
                        else:
                            b_valid[halo_y[dim_0][p]] |= AGG_J_M # agg_j

            elif stencil_type == 2: # diamond13P
                # Diamond13P: padd_x = base + 1, padd_y = base
                for p in range(padd_x):
                    halo_tile_x = out_i + 1
                    halo_tile_y = out_j if p < tile_y else out_j + 1

                    if halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                        halo_tile_idx = sche_dict[(halo_tile_x, halo_tile_y)]
                        halo_dim_0 = halo_tile_idx * z + k
                    else:
                        halo_dim_0 = -1

                    halo_x[dim_0][p] = (halo_dim_0, 0, p % tile_y)
                    if halo_dim_0 >= 0:
                        if p != tile_y: # out_i
                            b_valid[halo_x[dim_0][p]] |= OUT_I_M
                        if p != 0: # agg_i
                            b_valid[halo_x[dim_0][p]] |= AGG_I_M

                for p in range(padd_y):
                    halo_tile_x = out_i
                    halo_tile_y = out_j + 1

                    if halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                        halo_tile_idx = sche_dict[(halo_tile_x, halo_tile_y)]
                        halo_dim_0 = halo_tile_idx * z + k
                    else:
                        halo_dim_0 = -1

                    halo_y[dim_0][p] = (halo_dim_0, p % tile_x, 0)
                    if halo_dim_0 >= 0:
                        b_valid[halo_y[dim_0][p]] |= OUT_J_M # out_j

            elif stencil_type == 3: # Box27P
                # in和agg交替映射 halo_y[0]->in, halo_y[1]->agg, ...
                # padd_x = base + 1, padd_y = 2 * base
                for p in range(padd_x):
                    halo_tile_x = out_i + 1
                    halo_tile_y = out_j if p < tile_y else out_j + 1

                    if halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                        halo_tile_idx = sche_dict[(halo_tile_x, halo_tile_y)]
                        halo_dim_0 = halo_tile_idx * z + k
                    else:
                        halo_dim_0 = -1

                    halo_x[dim_0][p] = (halo_dim_0, 0, p % tile_y)
                    if halo_dim_0 >= 0:
                        if p != tile_y: # out_i
                            b_valid[halo_x[dim_0][p]] |= OUT_I_M
                        if p != 0: # agg_i
                            b_valid[halo_x[dim_0][p]] |= AGG_I_M

                for p in range(padd_y):
                    halo_tile_x = out_i if p < 2 * tile_x - 1 else out_i + 1
                    halo_tile_y = out_j + 1

                    if halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                        halo_tile_idx = sche_dict[(halo_tile_x, halo_tile_y)]
                        halo_dim_0 = halo_tile_idx * z + k
                    else:
                        halo_dim_0 = -1

                    halo_y[dim_0][p] = (halo_dim_0, ((p + 1) // 2) % tile_x, p % 2)
                    if halo_dim_0 >= 0:
                        if p % 2 == 0:
                            b_valid[halo_y[dim_0][p]] |= OUT_J_M # out_j
                        else:
                            b_valid[halo_y[dim_0][p]] |= AGG_J_M # agg_j
        tile_idx += 1

    return halo_x, halo_y, b_valid

def preprocess_domain_data_sptrsv(data, tile_x, tile_y, sche_seq):
    x, y, z = data["size"]
    n = x * y * z
    num_tile_x = math.ceil(x / tile_x)
    num_tile_y = math.ceil(y / tile_y)
    # store the tiling result in a 3D tensor
    num_tiles = num_tile_x * num_tile_y
    dim_shape = (num_tiles * z, tile_x, tile_y)
    matrix_diag = np.zeros(dim_shape)
    right_hand_side = np.ones(dim_shape)
    vec_index = np.zeros(dim_shape, dtype=object)
    # domain data
    tile_idx = 0

    for out_i, out_j in sche_seq:
        for k in range(z):
            dim_0 = tile_idx * z + k
            for in_i in range(tile_x):
                for in_j in range(tile_y):
                    total_i = out_i * tile_x + in_i
                    total_j = out_j * tile_y + in_j
                    addr = total_i * y * z + total_j * z + k
                    if addr < n:
                        matrix_diag[dim_0][in_i][in_j] = data["diag_a"][addr]
                        right_hand_side[dim_0][in_i][in_j] = data["b"][addr]
                    else:
                        matrix_diag[dim_0][in_i][in_j] = 1
                        right_hand_side[dim_0][in_i][in_j] = 0
                    vec_index[dim_0][in_i][in_j] = (total_i, total_j, k)
        tile_idx += 1
    return matrix_diag, right_hand_side, vec_index
