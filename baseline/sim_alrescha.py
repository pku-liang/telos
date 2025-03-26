from calc_tile import compute_SpMV_tiles
from calc_tile import compute_SymGS_tiles
import configs_64bit as configs_64bit
import pandas as pd

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
    sram_read = 2 * grid_size + gemv_tiles * tile_size + grid_size
    sram_write = gs_tiles * tile_size + 2 * grid_size
    add = grid_size * get_stencil_points(stencil_type, dims)
    mul = grid_size * get_stencil_points(stencil_type, dims)
    # add = gs_tiles * tile_size * (tile_size + 1) / 2 + gemv_tiles * (tile_size ** 2)
    # mul = gs_tiles * tile_size * (tile_size - 1) / 2 + gemv_tiles * (tile_size ** 2)
    div = grid_size

    mem_energy = mem_read * configs_64bit.dram_read_energy_per_data + \
                mem_write * configs_64bit.dram_write_energy_per_data
    sram_energy = sram_read * configs_64bit.alres_cache_read_energy + \
                sram_write * configs_64bit.alres_cache_write_energy
    op_energy = add * configs_64bit.adder_energy + mul * configs_64bit.multiplier_energy + div * configs_64bit.divisor_energy
    total_energy = mem_energy + sram_energy + op_energy

    return latency_cycles, mem_energy, sram_energy, op_energy, sram_read + sram_write

def sim_alrescha_spmv(gemv_tiles, tile_size, grid_size):
    latency_cycles = gemv_tiles * tile_size

    mem_read = gemv_tiles * (tile_size ** 2) + grid_size
    mem_write = grid_size

    sram_read = gemv_tiles * tile_size + grid_size
    sram_write = 2 * grid_size

    add = gemv_tiles * (tile_size ** 2)
    mul = add

    mem_energy = mem_read * configs_64bit.dram_read_energy_per_data + \
                mem_write * configs_64bit.dram_write_energy_per_data
    sram_energy = sram_read * configs_64bit.alres_cache_read_energy + \
                sram_write * configs_64bit.alres_cache_read_energy
    op_energy = add * configs_64bit.adder_energy + mul * configs_64bit.multiplier_energy
    total_energy = mem_energy + sram_energy + op_energy

    return latency_cycles, total_energy

def sim_alrescha_vector_op(grid_size, tile_size):
    vadd_n = dot_n = 3
    vadd_cycles = dot_cycles = 2 * grid_size / tile_size
    total_cycles = vadd_n * vadd_cycles + dot_n * dot_cycles

    dot_mem_read = 2 * grid_size
    vadd_mem_read = 2 * grid_size
    vadd_mem_write = grid_size
    mem_read = vadd_mem_read * vadd_n + dot_mem_read * dot_n
    mem_write = vadd_mem_write * vadd_n
    mem_energy = mem_read * configs_64bit.dram_read_energy_per_data + \
                mem_write * configs_64bit.dram_write_energy_per_data

    vadd_add = dot_add = grid_size
    vadd_mul = dot_mul = grid_size
    add = vadd_n * vadd_add + dot_n * dot_add
    mul = vadd_n * vadd_mul + dot_n * dot_mul
    op_energy = add * configs_64bit.adder_energy + mul * configs_64bit.multiplier_energy

    total_energy = mem_energy + op_energy
    return total_cycles, total_energy

import sys

if __name__ == "__main__":
    csv_path = sys.argv[1]
    size_lists = [
        # [16, 16],
        # [128, 128],
        # [512, 512],
        [64, 64, 64],
        [256, 256, 256],
        [1024, 1024],
        # [2048, 2048],
        [4096, 4096],
        # [8192, 8192],
        # [32, 32, 32],
        # [128, 128, 128],
        # [192, 192, 192],
        # [3072, 3072]
    ]
    tile_size = 8
    # clock freq (GHz)
    clock_freq = 2.5
    outfile = open(csv_path, "w")
    outfile.write("Grid Size, Stencil Type, SpTRSV D_SymGS Tiles, SpTRSV GEMV Tiles,\
                  SpTRSV Cycles, SpTRSV Latency(ms), SpTRSV Energy(pJ),\
                  Mem Energy(pJ), Mem Energy Ratio, SRAM Energy(pJ), SRAM Energy Ratio,\
                  Computation Energy(pJ), Computation Energy Ratio, SRAM Access\n")

    df = pd.read_csv("./alrescha_tile.csv", sep="\t")
    for size in size_lists:
        for stencil_type in range(4):
            dims = len(size)
            grid_size = size[0] * size[1] * (size[2] if len(size) == 3 else 1)

            # line = df[(df["Size"] == size[0]) & (df["StencilType"] == stencil_type)]
            # symgs_tiles = line.iloc[0]['SymGS']
            # gemv_tiles = line.iloc[0]['GEMV']
            # spmv_gemv_tiles = line.iloc[0]['SpMV_GEMV']

            gemv_tiles, symgs_tiles, cache_miss_cnt, cache_hit_cnt = compute_SymGS_tiles(stencil_type, size, tile_size)

            # spmv_gemv_tiles = compute_SpMV_tiles(stencil_type, size_list, tile_size)
            # print(f"SpTRSV: D_SymGS tiles: {symgs_tiles} GEMV tiles: {gemv_tiles}")
            # print(f"SpMV: GEMV tiles: {spmv_gemv_tiles}")

            sptrsv_cycles, sptrsv_mem_energy, sptrsv_sram_energy, sptrsv_op_energy, sram_access = \
                sim_alrescha_sptrsv(stencil_type, dims, symgs_tiles,
                                    gemv_tiles, tile_size, grid_size,
                                    cache_miss_cnt, cache_hit_cnt)

            total_energy = sptrsv_mem_energy + sptrsv_sram_energy + sptrsv_op_energy
            mem_ratio = sptrsv_mem_energy / total_energy
            sram_ratio = sptrsv_sram_energy / total_energy
            op_ratio = sptrsv_op_energy / total_energy
            # spmv_cycles, spmv_energy = sim_alrescha_spmv(spmv_gemv_tiles, tile_size, grid_size)
            # vec_cycles, vec_energy = sim_alrescha_vector_op(grid_size, tile_size)

            sptrsv_ms = sptrsv_cycles / clock_freq / 1000 / 1000

            # solver modeling
            # solver_cycles_perit = 2 * sptrsv_cycles + spmv_cycles + vec_cycles
            # solver_ms_perit = solver_cycles_perit / clock_freq / 1000 / 1000
            # solver_energy_perit = 2 * sptrsv_energy + spmv_energy + vec_energy

            # print("Modeling ALRESCHA SpTRSV")
            # print(f"Latency cycles: {sptrsv_cycles} cycles")
            # print(f"Total energy: {sptrsv_energy / 1e9} mJ")

            # print("Modeling ALRESCHA PCG Solver")
            # print(f"Latency cycles: {solver_cycles_perit} cycles")
            # print(f"Total energy: {solver_energy_perit / 1e9} mJ")
            # print(f"Latency times: {solver_ms_perit} ms")
            outfile.write(f"{size[0]}, {get_stencil_str(dims, stencil_type)}, {symgs_tiles},\
                        {gemv_tiles}, {sptrsv_cycles}, {sptrsv_ms}, {total_energy},\
                        {sptrsv_mem_energy}, {100 * mem_ratio:.2f}%, {sptrsv_sram_energy}, {100 * sram_ratio:.2f}%,\
                        {sptrsv_op_energy}, {100 * op_ratio:.2f}%,\
                        {sram_access}\n""")
            outfile.flush()
    outfile.close()