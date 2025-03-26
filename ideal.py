import json
import math
from MtxGen import get_num_domain_points
import sys
import os

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

for stencil_type in [0, 3]:
    for n in [512, 1024, 2048, 4096, 8192]:
        stencil_points = get_num_domain_points(stencil_type, 3)
        vec_lanes = 4
        mul_num = stencil_points / vec_lanes

        mul_num = math.ceil(stencil_points / vec_lanes)
        div_num = 1
        add_num = [1, 2, 1, 1]

        delay = 1 * add_num[stencil_type] + \
                1 * mul_num + \
                2 * div_num

        size = [n, n]
        tile_x = 8
        tile_y = 8
        ideal_cycles = math.ceil(math.prod(size) / (tile_x * tile_y)) * delay
        wallclock = ideal_cycles / (0.5 * 1e9) * 1000

        mem = ((stencil_points + 1) * 100.56 + 133.50) * math.prod(size)
        print(f"Stencil {stencil_type} {n}x{n}x{n}: {ideal_cycles} cycles, {wallclock} ms, mem {mem / 1e12} J")
