# affine_diamond: (x, y, z) -> (x, x + y, x + y + z)
# affine_box: (x, y, z) -> (x, x + y, 2x + y + z)

import math
import numpy as np
from collections import Counter

def affine_count(size, tile_X, tile_Y, affine_type):
    dim = len(size)
    if dim == 3:
        x, y, z = size
        tile_X_n = math.ceil(x / tile_X)
        tile_Y_n = math.ceil((x + y) / tile_Y)
        max_z = (x + y + z) if affine_type == "diamond" else (2 * x + y + z)
        tile_z_max = np.full((tile_X_n, tile_Y_n), -1)
        tile_z_min = np.full((tile_X_n, tile_Y_n), max_z)
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    x_new = i
                    y_new = i + j
                    z_new = i + j + k if affine_type == "diamond" else 2 * i + j + k
                    tile_z_max[x_new // tile_X][y_new // tile_Y] = max(tile_z_max[x_new // tile_X][y_new // tile_Y], z_new)
                    tile_z_min[x_new // tile_X][y_new // tile_Y] = min(tile_z_min[x_new // tile_X][y_new // tile_Y], z_new)

        non_empty_tile = 0
        after_dim0_size = 0
        counter = Counter()
        for i in range(tile_X_n):
            for j in range(tile_Y_n):
                if tile_z_max[i][j] != -1:
                    non_empty_tile += 1
                    height = (tile_z_max[i][j] - tile_z_min[i][j]) + 1
                    after_dim0_size += height
                    counter.update([height])

        before_dim0_size = tile_X_n * math.ceil(y / tile_Y) * z
        additional_cost = 100 * (after_dim0_size - before_dim0_size) / before_dim0_size
        print(counter)
        print(f"Total tiles: {tile_X_n * tile_Y_n}, non-empty tiles: {non_empty_tile}")
        # print(f"Before affine: dim0 size={before_dim0_size}")
        print(f"After affine: dim0 size={after_dim0_size}")
        # print(f"Additional Cost: {100 * (dim0_size - before_dim0_size) / before_dim0_size:.2f}%")
    else:
        x, y = size
        tile_X_n = math.ceil(x / tile_X)
        tile_Y_n = math.ceil((x + y) / tile_Y)
        tile_arr = np.zeros((tile_X_n, tile_Y_n))
        for i in range(x):
            for j in range(y):
                x_new = i
                y_new = i + j
                tile_arr[x_new // tile_X][y_new // tile_Y] += 1

        non_empty_tile = 0
        for i in range(tile_X_n):
            for j in range(tile_Y_n):
                if tile_arr[i][j] > 0:
                    non_empty_tile += 1

        before_dim0_size = tile_X_n * math.ceil(y / tile_Y)
        after_dim0_size = non_empty_tile
        additional_cost = 100 * (non_empty_tile - before_dim0_size) / before_dim0_size
        # print(f"Tiles before affine: {before_dim0_size}")
        # print(f"Tiles after affine: {}")
        # print(f"Additional cost: {:.2f}%")
    return before_dim0_size, after_dim0_size, additional_cost

if __name__ == "__main__":
    csv_path = "data/affine_2d.csv"
    csv_file = open(csv_path, "w")
    tile_X = tile_Y = 8
    csv_file.write("Size, 2D-Diamond(Box), 3D-Diamond, 3D-Box\n")
    size_list = [1024, 2048, 3072, 4096]
    for i in size_list:
        _, _, additional_cost_2d = affine_count((i, i), tile_X, tile_Y, "box")
        csv_file.write(f"{i}, {additional_cost_2d:.2f}%\n")
        csv_file.flush()

    # for i in range(32, 512 + 1, 32):
    #     print(f"Running {i}")
    #     _, _, additional_cost_2d = affine_count((i, i), tile_X, tile_Y, "box")
    #     _, _, additional_cost_3d_diamond = affine_count((i, i, i), tile_X, tile_Y, "diamond")
    #     _, _, additional_cost_3d_box = affine_count((i, i, i), tile_X, tile_Y, "box")
    #     csv_file.write(f"{i}, {additional_cost_2d:.2f}%, {additional_cost_3d_diamond:.2f}%, {additional_cost_3d_box:.2f}%\n")
    #     csv_file.flush()