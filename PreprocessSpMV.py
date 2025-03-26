import numpy as np
import math
import random
from mask_const import *
from stencil import get_id2stage, get_stages, get_num_halo_points, get_stencil_points

def preprocess_spmv(data, tile_x, tile_y, stencil_type, dims):
    matrix_value, A_valid = preprocessPGC_spmv(data["size"], data['A'], tile_x, tile_y, stencil_type, dims)
    result = np.zeros(data["size"])

    x, b, vec_index = preprocess_domain_data_spmv(data, tile_x, tile_y)
    
    halo_x, halo_y, halo_x_inv, halo_y_inv, b_valid = preprocess_halo_data_spmv(data, tile_x, tile_y, stencil_type, dims)

    processed_data = {
        "size": data["size"], "A": matrix_value, "A_valid": A_valid, 
        "b": b, "b_valid": b_valid, 
        "x": x, "ijk": vec_index,
        "halo_x": halo_x, "halo_y": halo_y, "halo_x_inv": halo_x_inv, "halo_y_inv": halo_y_inv,
        "result": result, "gt": data["gt"]
    }     
    return processed_data

def preprocessPGC_spmv(size, data_A, tile_x, tile_y, stencil_type, dims):
    x, y, z = size
    n = x * y * z
    num_tile_x = math.ceil(x / tile_x)
    num_tile_y = math.ceil(y / tile_y)
    num_tiles = num_tile_x * num_tile_y
    stencil_length = data_A.shape[-1]

    id2stage = get_id2stage(dims, stencil_type, "spmv")
    stages = get_stages(dims, stencil_type, "spmv")

    dim_shape_A = (num_tiles * z + stages - 1, tile_x, tile_y, stencil_length)
    matrix_value = np.zeros(dim_shape_A)
    A_valid = np.ones(dim_shape_A[0], dtype=np.bool)
    for i in range(stages - 1):
        A_valid[i + num_tiles * z] = False

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

    return matrix_value, A_valid


def preprocess_halo_data_spmv(data, tile_x, tile_y, stencil_type, dims):
    x, y, z = data["size"]
    num_tile_x = math.ceil(x / tile_x)
    num_tile_y = math.ceil(y / tile_y)
    num_tiles = num_tile_x * num_tile_y
    dim_shape = (num_tiles * z, tile_x, tile_y)
    # padding for halo data
    padd_x, padd_y, padd_x_inv, padd_y_inv = get_num_halo_points(stencil_type, tile_x, tile_y, "spmv")
    # halo data store the index of b (tile_idx, x_id_in_tile, y_id_in_tile)
    halo_x = np.zeros((num_tiles * z, padd_x), dtype=object)
    halo_y = np.zeros((num_tiles * z, padd_y), dtype=object)
    halo_x_inv = np.zeros((num_tiles * z, padd_x_inv), dtype=object)
    halo_y_inv = np.zeros((num_tiles * z, padd_y_inv), dtype=object)
    
    # check the update status of b
    b_valid = np.full(dim_shape, CUR_M, dtype=np.int16)

    for out_i in range(num_tile_x):
        for out_j in range(num_tile_y):
            tile_idx = out_i * num_tile_y + out_j
            for k in range(z):
                dim_0 = tile_idx * z + k
                if stencil_type == 0: # Star7P
                    # HEU_X
                    for p in range(padd_x):
                        halo_tile_x = out_i + 1
                        halo_tile_y = out_j

                        if halo_tile_x >= 0 and halo_tile_y >= 0 and halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                            halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                            halo_dim_0 = halo_tile_idx * z + k
                        else:
                            halo_dim_0 = -1

                        halo_x[dim_0][p] = (halo_dim_0, 0, p % tile_y)
                        if halo_dim_0 >= 0:
                            b_valid[halo_x[dim_0][p]] |= OUT_I_M # out_i

                    # HEU_Y
                    for p in range(padd_y):
                        halo_tile_x = out_i
                        halo_tile_y = out_j + 1

                        if halo_tile_x >= 0 and halo_tile_y >= 0 and halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                            halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                            halo_dim_0 = halo_tile_idx * z + k
                        else:
                            halo_dim_0 = -1
                        halo_y[dim_0][p] = (halo_dim_0, p % tile_x, 0)
                        if halo_dim_0 >= 0:
                            b_valid[halo_y[dim_0][p]] |= OUT_J_M # out_j
                    
                    # HEU_X_INV
                    for p in range(padd_x_inv):
                        halo_tile_x = out_i - 1
                        halo_tile_y = out_j

                        if halo_tile_x >= 0 and halo_tile_y >= 0 and halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                            halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                            halo_dim_0 = halo_tile_idx * z + k
                        else:
                            halo_dim_0 = -1

                        halo_x_inv[dim_0][p] = (halo_dim_0, tile_x - 1, p % tile_y)
                        if halo_dim_0 >= 0:
                            b_valid[halo_x_inv[dim_0][p]] |= OUT_I_INV_M # out_i_inv
                    
                    # HEU_Y_INV
                    for p in range(padd_y_inv):
                        halo_tile_x = out_i
                        halo_tile_y = out_j - 1

                        if halo_tile_x >= 0 and halo_tile_y >= 0 and halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                            halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                            halo_dim_0 = halo_tile_idx * z + k
                        else:
                            halo_dim_0 = -1

                        halo_y_inv[dim_0][p] = (halo_dim_0, p % tile_x, tile_y - 1)
                        if halo_dim_0 >= 0:
                            b_valid[halo_y_inv[dim_0][p]] |= OUT_J_INV_M # out_j_inv

                elif stencil_type == 1: # Star13P
                    # in和agg交替映射 halo_x[0]->in, halo_x[1]->agg, ...
                    # padd_x = padd_y = 2 * base
                    for p in range(padd_x):
                        dx = tile_x if p % 2 == 0 else tile_x + 1
                        dy = p // 2

                        halo_tile_x = out_i + dx // tile_x
                        halo_tile_y = out_j + dy // tile_y

                        if halo_tile_x >= 0 and halo_tile_y >= 0 and halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                            halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                            halo_dim_0 = halo_tile_idx * z + k
                        else:
                            halo_dim_0 = -1
                        halo_x[dim_0][p] = (halo_dim_0, dx % tile_x, dy % tile_y)

                        if halo_dim_0 >= 0:
                            if p % 2 == 0:
                                b_valid[halo_x[dim_0][p]] |= OUT_I_M # out_i
                            else:
                                b_valid[halo_x[dim_0][p]] |= AGG_I_M # agg_i

                    for p in range(padd_y):
                        dx = p // 2
                        dy = tile_y if p % 2 == 0 else tile_y + 1

                        halo_tile_x = out_i + dx // tile_x
                        halo_tile_y = out_j + dy // tile_y

                        if halo_tile_x >= 0 and halo_tile_y >= 0 and halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                            halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                            halo_dim_0 = halo_tile_idx * z + k
                        else:
                            halo_dim_0 = -1
                        halo_y[dim_0][p] = (halo_dim_0, dx % tile_x, dy % tile_y)

                        if halo_dim_0 >= 0:
                            if p % 2 == 0:
                                b_valid[halo_y[dim_0][p]] |= OUT_J_M # out_j
                            else:
                                b_valid[halo_y[dim_0][p]] |= AGG_J_M # agg_j

                    for p in range(padd_x_inv):
                        dx = -1 if p % 2 == 0 else -2
                        dy = p // 2

                        halo_tile_x = out_i + dx // tile_x
                        halo_tile_y = out_j + dy // tile_y

                        if halo_tile_x >= 0 and halo_tile_y >= 0 and halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                            halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                            halo_dim_0 = halo_tile_idx * z + k
                        else:
                            halo_dim_0 = -1
                        halo_x_inv[dim_0][p] = (halo_dim_0, dx % tile_x, dy % tile_y)

                        if halo_dim_0 >= 0:
                            if p % 2 == 0:
                                b_valid[halo_x_inv[dim_0][p]] |= OUT_I_INV_M # out_i_inv
                            else:
                                b_valid[halo_x_inv[dim_0][p]] |= AGG_I_INV_M # agg_i_inv

                    for p in range(padd_y_inv):
                        dx = p // 2
                        dy = -1 if p % 2 == 0 else -2

                        halo_tile_x = out_i + dx // tile_x
                        halo_tile_y = out_j + dy // tile_y

                        if halo_tile_x >= 0 and halo_tile_y >= 0 and halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                            halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                            halo_dim_0 = halo_tile_idx * z + k
                        else:
                            halo_dim_0 = -1
                        halo_y_inv[dim_0][p] = (halo_dim_0, dx % tile_x, dy % tile_y)

                        if halo_dim_0 >= 0:
                            if p % 2 == 0:
                                b_valid[halo_y_inv[dim_0][p]] |= OUT_J_INV_M # out_j_inv
                            else:
                                b_valid[halo_y_inv[dim_0][p]] |= AGG_J_INV_M # agg_j_inv

                elif stencil_type == 2: # diamond13P
                    for p in range(padd_x):
                        dx = tile_x
                        dy = p - 1

                        halo_tile_x = out_i + dx // tile_x
                        halo_tile_y = out_j + dy // tile_y

                        if halo_tile_x >= 0 and halo_tile_y >= 0 and halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                            halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                            halo_dim_0 = halo_tile_idx * z + k
                        else:
                            halo_dim_0 = -1
                        halo_x[dim_0][p] = (halo_dim_0, dx % tile_x, dy % tile_y)

                        if halo_dim_0 >= 0:
                            if p != 0:
                                b_valid[halo_x[dim_0][p]] |= OUT_I_M # out_i
                            if p != tile_y:
                                b_valid[halo_x[dim_0][p]] |= AGG_I_M # agg_i

                    for p in range(padd_y):
                        dx = p
                        dy = tile_y

                        halo_tile_x = out_i + dx // tile_x
                        halo_tile_y = out_j + dy // tile_y

                        if halo_tile_x >= 0 and halo_tile_y >= 0 and halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                            halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                            halo_dim_0 = halo_tile_idx * z + k
                        else:
                            halo_dim_0 = -1
                        halo_y[dim_0][p] = (halo_dim_0, dx % tile_x, dy % tile_y)

                        if halo_dim_0 >= 0:
                            b_valid[halo_y[dim_0][p]] |= OUT_J_M # out_j

                    for p in range(padd_x_inv):
                        dx = -1
                        dy = p

                        halo_tile_x = out_i + dx // tile_x
                        halo_tile_y = out_j + dy // tile_y

                        if halo_tile_x >= 0 and halo_tile_y >= 0 and halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                            halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                            halo_dim_0 = halo_tile_idx * z + k
                        else:
                            halo_dim_0 = -1
                        halo_x_inv[dim_0][p] = (halo_dim_0, dx % tile_x, dy % tile_y)
                        # print(halo_x_inv[dim_0][p])

                        if halo_dim_0 >= 0:
                            if p != tile_y:
                                b_valid[halo_x_inv[dim_0][p]] |= OUT_I_INV_M # out_i_inv
                            if p != 0:
                                b_valid[halo_x_inv[dim_0][p]] |= AGG_I_INV_M # agg_i_inv

                    for p in range(padd_y_inv):
                        dx = p
                        dy = -1

                        halo_tile_x = out_i + dx // tile_x
                        halo_tile_y = out_j + dy // tile_y

                        if halo_tile_x >= 0 and halo_tile_y >= 0 and halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                            halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                            halo_dim_0 = halo_tile_idx * z + k
                        else:
                            halo_dim_0 = -1
                        halo_y_inv[dim_0][p] = (halo_dim_0, dx % tile_x, dy % tile_y)

                        if halo_dim_0 >= 0:
                            b_valid[halo_y_inv[dim_0][p]] |= OUT_J_INV_M # out_j_inv

                elif stencil_type == 3: # Box27P
                    # in和agg交替映射 halo_y[0]->in, halo_y[1]->agg, ...
                    # padd_x = base + 1, padd_y = 2 * base
                    for p in range(padd_x):
                        dx = tile_x
                        dy = p - 1

                        halo_tile_x = out_i + dx // tile_x
                        halo_tile_y = out_j + dy // tile_y

                        if halo_tile_x >= 0 and halo_tile_y >= 0 and halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                            halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                            halo_dim_0 = halo_tile_idx * z + k
                        else:
                            halo_dim_0 = -1
                        halo_x[dim_0][p] = (halo_dim_0, dx % tile_x, dy % tile_y)

                        if halo_dim_0 >= 0:
                            if p != 0: # out_i
                                b_valid[halo_x[dim_0][p]] |= OUT_I_M
                            if p != tile_y: # agg_i
                                b_valid[halo_x[dim_0][p]] |= AGG_I_M

                    for p in range(padd_y):
                        dx = p
                        dy = tile_y

                        halo_tile_x = out_i + dx // tile_x
                        halo_tile_y = out_j + dy // tile_y

                        if halo_tile_x >= 0 and halo_tile_y >= 0 and halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                            halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                            halo_dim_0 = halo_tile_idx * z + k
                        else:
                            halo_dim_0 = -1
                        halo_y[dim_0][p] = (halo_dim_0, dx % tile_x, dy % tile_y)

                        if halo_dim_0 >= 0:
                            if p != tile_x:
                                b_valid[halo_y[dim_0][p]] |= OUT_J_M # out_j
                            if p != 0:
                                b_valid[halo_y[dim_0][p]] |= AGG_J_M # agg_j
                    
                    for p in range(padd_x_inv):
                        dx = -1
                        dy = p

                        halo_tile_x = out_i + dx // tile_x
                        halo_tile_y = out_j + dy // tile_y

                        if halo_tile_x >= 0 and halo_tile_y >= 0 and halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                            halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                            halo_dim_0 = halo_tile_idx * z + k
                        else:
                            halo_dim_0 = -1
                        halo_x_inv[dim_0][p] = (halo_dim_0, dx % tile_x, dy % tile_y)
                        
                        if halo_dim_0 >= 0:
                            if p != tile_y:
                                b_valid[halo_x_inv[dim_0][p]] |= OUT_I_INV_M # out_i_inv
                            if p != 0:
                                b_valid[halo_x_inv[dim_0][p]] |= AGG_I_INV_M # agg_i_inv
                    
                    for p in range(padd_y_inv):
                        dx = p - 1
                        dy = -1

                        halo_tile_x = out_i + dx // tile_x
                        halo_tile_y = out_j + dy // tile_y

                        if halo_tile_x >= 0 and halo_tile_y >= 0 and halo_tile_x < num_tile_x and halo_tile_y < num_tile_y:
                            halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                            halo_dim_0 = halo_tile_idx * z + k
                        else:
                            halo_dim_0 = -1
                        halo_y_inv[dim_0][p] = (halo_dim_0, dx % tile_x, dy % tile_y)
                        
                        if halo_dim_0 >= 0:
                            if p != 0:
                                b_valid[halo_y_inv[dim_0][p]] |= OUT_J_INV_M # out_j_inv
                            if p != tile_x:
                                b_valid[halo_y_inv[dim_0][p]] |= AGG_J_INV_M # agg_j_inv

    return halo_x, halo_y, halo_x_inv, halo_y_inv, b_valid

def preprocess_domain_data_spmv(data, tile_x, tile_y):
    x, y, z = data["size"]
    n = x * y * z
    num_tile_x = math.ceil(x / tile_x)
    num_tile_y = math.ceil(y / tile_y)
    # store the tiling result in a 3D tensor
    num_tiles = num_tile_x * num_tile_y
    dim_shape = (num_tiles * z, tile_x, tile_y)
    x = np.ones(dim_shape)
    b = np.zeros(dim_shape)
    vec_index = np.zeros(dim_shape, dtype=object)
    # domain data
    for out_i in range(num_tile_x):
        for out_j in range(num_tile_y):
            tile_idx = out_i * num_tile_y + out_j
            for k in range(z):
                dim_0 = tile_idx * z + k
                for in_i in range(tile_x):
                    for in_j in range(tile_y):
                        total_i = out_i * tile_x + in_i
                        total_j = out_j * tile_y + in_j
                        addr = total_i * y * z + total_j * z + k
                        if addr < n:
                            x[dim_0][in_i][in_j] = data["x"][addr]
                            b[dim_0][in_i][in_j] = data["b"][addr]
                        else:
                            x[dim_0][in_i][in_j] = 0
                            b[dim_0][in_i][in_j] = 0
                        vec_index[dim_0][in_i][in_j] = (total_i, total_j, k)

    return x, b, vec_index