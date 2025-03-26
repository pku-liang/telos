from calc_tile import compute_SymGS_tiles
import configs_64bit as configs_64bit
import pandas as pd
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_stencil_str(dims, stencil_type):
    stencil_str_2d = ["2D-Star-5P", "2D-Star-9P", "2D-Diamond-7P", "2D-Box-9P"]
    stencil_str_3d= ["3D-Star-7P", "3D-Star-13P", "3D-Diamond-13P", "3D-Box-27P"]
    if dims == 3:
        return stencil_str_3d[stencil_type]
    else:
        return stencil_str_2d[stencil_type]

def get_stencil_points(stencil_type, dims):
    points_2d = [2, 3, 3, 4]
    points_3d = [3, 6, 6, 13]
    if dims == 3:
        return points_3d[stencil_type]
    else:
        return points_2d[stencil_type]

def sim_alrescha_sptrsv(stencil_type, dims, gs_tiles, gemv_tiles, tile_size, grid_size,
                        cache_miss_cnt, cache_hit_cnt):
    delay = 1
    latency_cycles = delay * gs_tiles * tile_size + gemv_tiles * tile_size

    # mem model
    num_points = get_stencil_points(stencil_type, dims)
    mem_read = grid_size * (num_points + 2) + tile_size * cache_miss_cnt
    # mem_read = gs_tiles * tile_size * (tile_size - 1) / 2 + \
    #             gemv_tiles * (tile_size ** 2)
    mem_write = grid_size

    # sram model
    cache_read = cache_hit_cnt * tile_size
    cache_write = (gs_tiles + cache_miss_cnt) * tile_size
    fifo_read = 2 * grid_size
    fifo_write = 2 * grid_size
    sram_read = cache_read + fifo_read
    sram_write = cache_write + fifo_write

    add = grid_size * num_points
    mul = grid_size * num_points
    # add = gs_tiles * tile_size * (tile_size + 1) / 2 + gemv_tiles * (tile_size ** 2)
    # mul = gs_tiles * tile_size * (tile_size - 1) / 2 + gemv_tiles * (tile_size ** 2)
    div = grid_size

    mem_energy = mem_read * configs_64bit.dram_read_energy_per_data + \
                mem_write * configs_64bit.dram_write_energy_per_data
    sram_energy = sram_read * configs_64bit.alres_cache_read_energy + \
                sram_write * configs_64bit.alres_cache_write_energy
    op_energy = add * configs_64bit.adder_energy + mul * configs_64bit.multiplier_energy + div * configs_64bit.divisor_energy
    # total_energy = mem_energy + sram_energy + op_energy

    return latency_cycles, mem_energy, sram_energy, op_energy, sram_read + sram_write

def sim_task(size, stencil_type, tile_size):
    dims = len(size)
    grid_size = size[0] * size[1] * (size[2] if len(size) == 3 else 1)
    gemv_tiles, symgs_tiles, cache_miss_cnt, cache_hit_cnt = compute_SymGS_tiles(stencil_type, size, tile_size)
    _, sptrsv_mem_energy, sptrsv_sram_energy, sptrsv_op_energy, sram_access = \
        sim_alrescha_sptrsv(stencil_type, dims, symgs_tiles,
                            gemv_tiles, tile_size, grid_size,
                            cache_miss_cnt, cache_hit_cnt)

    return symgs_tiles, gemv_tiles, sptrsv_mem_energy, sptrsv_sram_energy,\
        sptrsv_op_energy, sram_access, cache_hit_cnt, cache_miss_cnt

def run_sim(size_list, csv_path, max_procs):
    tile_size = 8
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        tasks = {}
        for size in size_list:
            for stencil_type in range(4):
                tasks[executor.submit(sim_task, size, stencil_type, tile_size)] = (size, stencil_type, tile_size)

        with open(csv_path, "w") as outfile:
            outfile.write("Grid Size, Stencil Type, SpTRSV D_SymGS Tiles, SpTRSV GEMV Tiles,\
                  cache hit, cache miss,\
                  Mem Energy(pJ), SRAM Energy(pJ), Computation Energy(pJ), SRAM Access\n")
            for task in as_completed(tasks):
                size, stencil_type, tile_size = tasks[task]
                symgs_tiles, gemv_tiles, sptrsv_mem_energy, sptrsv_sram_energy,\
                    sptrsv_op_energy, sram_access, cache_hit, cache_miss = task.result()
                dims = len(size)
                outfile.write(f"{size[0]}, {get_stencil_str(dims, stencil_type)}, {symgs_tiles}, {gemv_tiles},\
                            {cache_hit}, {cache_miss},\
                            {sptrsv_mem_energy}, {sptrsv_sram_energy}, {sptrsv_op_energy}, \
                            {sram_access}\n""")
                outfile.flush()


if __name__ == "__main__":
    csv_path = sys.argv[1]
    size_list = [
        [64, 64, 64],
        [256, 256, 256],
        [1024, 1024],
        [4096, 4096],
    ]
    run_sim(size_list, csv_path, 64)
