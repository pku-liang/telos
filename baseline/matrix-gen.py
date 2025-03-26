# 读入命令行参数, 4-5个数字, 代表stencil类型, 维数, x, y, z

import sys
import numpy as np

dim = int(sys.argv[1])
stencil_type = int(sys.argv[2])
x = int(sys.argv[3])
y = int(sys.argv[4])
if dim == 3:
    z = int(sys.argv[5])
else:
    z = 1

grid_size = x * y * z

stencil_points = []

if stencil_type == 0:
    # 3D-Star-7P/2D-Star-5P
    if dim == 2:
        stencil_points = [
            (0, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (0, 1),
        ]
    elif dim == 3:
        stencil_points = [
            (0, 0, -1),
            (0, -1, 0),
            (-1, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        ]
elif stencil_type == 1:
    # 3D-Star-13P/2D-Star-9P
    if dim == 2:
        stencil_points = [
            (0, -2),
            (0, -1),
            (-2, 0),
            (-1, 0),
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (0, 2),
        ]
    elif dim == 3:
        stencil_points = [
            (0, 0, -2),
            (0, 0, -1),
            (0, -2, 0),
            (0, -1, 0),
            (-2, 0, 0),
            (-1, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (0, 1, 0),
            (0, 2, 0),
            (0, 0, 1),
            (0, 0, 2),
        ]
elif stencil_type == 2:
    # 3D-Diamond-13P / 2D-Diamond-7P
    if dim == 2:
        stencil_points = [
            (0, -1),
            (1, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
        ]
    elif dim == 3:
        stencil_points = [
            (0, 0, -1),
            (1, 0, -1),
            (0, 1, -1),
            (0, -1, 0),
            (1, -1, 0),
            (-1, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
            (-1, 1, 0),
            (0, 1, 0),
            (0, -1, 1),
            (-1, 0, 1),
            (0, 0, 1),
        ]
elif stencil_type == 3:
    # 3D-Box-27P / 2D-Box-9P
    if dim == 2:
        stencil_points = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ]
    elif dim == 3:
        stencil_points = [
            (-1, -1, -1),
            (0, -1, -1),
            (1, -1, -1),
            (-1, 0, -1),
            (0, 0, -1),
            (1, 0, -1),
            (-1, 1, -1),
            (0, 1, -1),
            (1, 1, -1),
            (-1, -1, 0),
            (0, -1, 0),
            (1, -1, 0),
            (-1, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
            (-1, 1, 0),
            (0, 1, 0),
            (1, 1, 0),
            (-1, -1, 1),
            (0, -1, 1),
            (1, -1, 1),
            (-1, 0, 1),
            (0, 0, 1),
            (1, 0, 1),
            (-1, 1, 1),
            (0, 1, 1),
            (1, 1, 1),
        ]

stencil_length = len(stencil_points)
matrix_value = np.zeros((grid_size, stencil_length // 2 + 1))

for i in range(z):
    for j in range(y):
        for k in range(x):
            # only lower triangular
            cnt = 0
            for l in range(stencil_length // 2):
                if dim == 2:
                    dx, dy = stencil_points[l]
                    dz = 0
                else:
                    dx, dy, dz = stencil_points[l]
                x_new = k + dx
                y_new = j + dy
                z_new = i + dz
                if (
                    x_new >= 0
                    and x_new < x
                    and y_new >= 0
                    and y_new < y
                    and z_new >= 0
                    and z_new < z
                ):
                    matrix_value[i * x * y + j * x + k][l] = -1.0
                    cnt += 1
                else:
                    matrix_value[i * x * y + j * x + k][l] = 0.0
            matrix_value[i * x * y + j * x + k][stencil_length // 2] = cnt + 1.0

print(matrix_value)
