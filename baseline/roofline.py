import math
import configs_32bit as config
from stencil import get_stencil_points
from sim_alrescha import get_stencil_str


vadd_flops = 38.33
dot_flops = 57.5
peak_flops = 102.4
bw = 460

def sim_roof_spmv(stencil_length, grid_size):
    grid_n = math.prod(grid_size)

    spmv_add = spmv_mul = stencil_length * grid_n
    dram_read = (stencil_length + 1) * grid_n
    dram_write = grid_n

    ci = stencil_length / ((stencil_length + 2) * 4)
    flops = max(ci * bw, peak_flops)
    spmv_latency = stencil_length * grid_n / flops

    dram_energy = dram_read * config.dram_read_energy_per_data + \
                    dram_write * config.dram_write_energy_per_data
    op_energy = spmv_add * config.adder_energy + \
                spmv_mul * config.multiplier_energy
    total_energy = dram_energy + op_energy
    return spmv_latency, total_energy


def sim_vector_op(grid_size):
    grid_n = math.prod(grid_size)

    vadd_latency = grid_n / vadd_flops
    dot_latency = grid_n / dot_flops
    total_latency = vadd_latency + dot_latency

    dram_read = 4 * grid_n
    dram_write = grid_n
    add = 2 * grid_n
    mul = 2 * grid_n
    dram_energy = dram_read * config.dram_read_energy_per_data + \
                    dram_write * config.dram_write_energy_per_data
    op_energy = add * config.adder_energy + \
                mul * config.multiplier_energy

    total_energy = dram_energy + op_energy
    return total_latency, total_energy

if __name__ == "__main__":
    size_list = [
        (32, 32, 32),
        (64, 64, 64),
        (96, 96, 96),
        # (128, 128, 128),
        # (256, 256, 256),
        (256, 256),
        (512, 512),
        (768, 768),
        (1024, 1024),
    ]
    with open("./sim_roof_spmv.csv", "w") as csv_file:
        csv_file.write("size,stencil,latency(ms),energy(pJ)\n")
        for size in size_list:
            for stencil in range(4):
                dims = len(size)
                stencil_length = len(get_stencil_points(stencil, dims, False))
                spmv_latency, spmv_energy = sim_roof_spmv(stencil_length, size)
                stencil_str = get_stencil_str(dims, stencil)
                csv_file.write(f"{size[0]},{stencil_str},{spmv_latency / 1e6:.3f},{spmv_energy:.3f}\n")

    # sptrsv_result_list = [
    #     (11887, 118798999.14000002),
    #     (95663, 950391283.3000001),
    #     (765871, 7603189522.020001),
    #     (40457, 397957826.46),
    #     (161801, 1591824517.02),
    #     (647177, 6367291279.26),
    # ]
    # clock_freq = 0.4
    # for grid_size, sptrsv_result in zip(size_list, sptrsv_result_list):
    #     print(grid_size)
    #     sptrsv_cycles, sptrsv_energy = sptrsv_result
    #     spmv_latency, spmv_energy = sim_roof_spmv(grid_size)
    #     vec_latency, vec_energy = sim_vector_op(grid_size)
    #     sptrsv_latency = sptrsv_cycles / clock_freq

    #     total_latency = 2 * sptrsv_latency + spmv_latency + 3 * vec_latency
    #     total_energy = 2 * sptrsv_energy + spmv_energy + 3 * vec_energy
    #     # print(f"Sptrsv Latency: {sptrsv_latency / 1e6} ms, Spmv Latency: {spmv_latency / 1e6} ms, Vector Latency: {vec_latency / 1e6} ms")
    #     print(f"Total Latency: {total_latency / 1e6} ms")
    #     # print(f"SpTRSV Energy: {2 * sptrsv_energy / 1e9} mJ, SpMV Energy: {spmv_energy / 1e9} mJ, Vec Energy: {3 * vec_energy / 1e9} mJ")
    #     print(f"Total Energy: {total_energy / 1e9} mJ")