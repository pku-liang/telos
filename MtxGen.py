# 读入命令行参数, 4-5个数字, 代表stencil类型, 维数, x, y, z
import numpy as np
import math
import random
from stencil import get_affine_stencil_points, get_stencil_points

def get_linear_system_sptrsv(dim, stencil_type, size):
    grid_size = math.prod(size)
    if dim == 3:
        x, y, z = size
    else:
        x, y = size
        z = 1

    stencil_points = get_affine_stencil_points(dim, stencil_type)
    stencil_length = len(stencil_points)
    matrix_value = np.zeros((grid_size, stencil_length))
    matrix_diag = np.zeros(grid_size)
    right_hand_side = np.ones(grid_size)

    for k in range(x):
        for j in range(y):
            for i in range(z):
                # only lower triangular
                sum = 0
                idx = k * y * z + j * z + i
                for l in range(stencil_length):
                    if dim == 2:
                        dx, dy = stencil_points[l]
                        dz = 0
                    else:
                        dx, dy, dz = stencil_points[l]
                    x_new = k + dx
                    y_new = j + dy
                    z_new = i + dz
                    idx_new = x_new * y * z + y_new * z + z_new
                    if x_new >= 0 and y_new >= 0 and z_new >= 0:
                        rand_num = -random.randint(1, 10)
                        matrix_value[idx_new][l] = rand_num
                        sum += rand_num
                        
                matrix_diag[idx] = -sum + 1.0

    data = { "size": (x, y, z), "A": matrix_value, "diag_a": matrix_diag, "b": right_hand_side }
    return data


def get_linear_system_spmv(dim, stencil_type, size):
    grid_size = math.prod(size)
    if dim == 3:
        x, y, z = size
    else:
        x, y = size
        z = 1

    stencil_points = get_stencil_points(dim, stencil_type)
    stencil_length = len(stencil_points)
    matrix_value = np.zeros((grid_size, stencil_length))
    b = np.zeros(grid_size)
    rhs_gt = np.zeros((x, y, z))
    x_arr = np.random.randint(1, 10, size=grid_size)

    for k in range(x):
        for j in range(y):
            for i in range(z):
                # sum = 0
                idx = k * y * z + j * z + i
                for l in range(stencil_length):
                    if dim == 2:
                        dx, dy = stencil_points[l]
                        dz = 0
                    else:
                        dx, dy, dz = stencil_points[l]
                    x_new = k - dx
                    y_new = j - dy
                    z_new = i - dz
                    idx_new = x_new * y * z + y_new * z + z_new
                    if x_new >= 0 and y_new >= 0 and z_new >= 0 and x_new < x and y_new < y and z_new < z:
                        rand_num = random.randint(1, 10)
                        matrix_value[idx_new][l] = rand_num
                        rhs_gt[k][j][i] += rand_num * x_arr[idx_new]

    data = { "size": (x, y, z), "A": matrix_value, "x": x_arr, "b": b, "gt": rhs_gt}
    return data