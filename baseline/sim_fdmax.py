import configs_32bit as configs
from math import ceil

def get_gs2d_tiles(size, tile_size):
    x, y = size
    tile_x, tile_y = tile_size
    num_tile_x = ceil(x / tile_x)
    num_tile_y = ceil(y / tile_y)
    crit_delay = 1
    halo_delay = tile_x * crit_delay

    total_tiles = 0
    for d in range(num_tile_y + num_tile_x - 1):
        diag_len = min(num_tile_x, d + 1) - max(0, d + 1 - num_tile_y)
        total_tiles += max(halo_delay, diag_len)

    return total_tiles
    # print(f"tiles before padding: {num_tile_x * num_tile_y}, tiles after padding: {total_tiles}")

def sim_gs2d(size, tile_size):
    gx, gy = size
    tx, ty = tile_size

    nx = ceil(gx / tx)
    ny = ceil(gy / ty)

    grid_n = gx * gy
    halo_n = nx * ny * (tx + ty)

    latency = get_gs2d_tiles(size, tile_size)

    cur_buf_access = 2 * (grid_n + halo_n)
    offset_buf_access = 2 * grid_n
    halo_buf_access = 2 * halo_n
    next_buf_access = 2 * grid_n

    mem_read = 2 * (halo_n + grid_n)
    mem_write = halo_n + grid_n

    add = 7 * grid_n + halo_n
    mul = 4 * grid_n + halo_n

    mem_energy = mem_read * configs.dram_read_energy_per_data + \
                mem_write * configs.dram_write_energy_per_data
    sram_energy = cur_buf_access * configs.gs_cur_buf_access_energy + \
                    offset_buf_access * configs.gs_offset_buf_access_energy + \
                    halo_buf_access * configs.gs_halo_buf_access_energy + \
                    next_buf_access * configs.gs_next_buf_access_energy

    op_energy = add * configs.adder_energy + mul * configs.multiplier_energy


    halo_energy = 2 * halo_n +

    total_energy = mem_energy + sram_energy + op_energy

    return latency, total_energy

def sim_gs3d(size, tile_size):
    gx, gy, gz = size
    tx, ty = tile_size
    nx = ceil(gx / tx)
    ny = ceil(gy / ty)

    latency = nx * ny * gz + 15 - 1

    grid_n = gx * gy * gz
    halo_n = nx * ny * gz * (tx + ty)

    mem_read = 2 * halo_n + grid_n
    mem_write = halo_n + grid_n

    cur_buf_access = 2 * (grid_n + halo_n)
    halo_buf_access = 4 * halo_n
    next_buf_access = 2 * grid_n

    add = 7 * grid_n + halo_n
    mul = 4 * grid_n + halo_n

    mem_energy = mem_read * configs.dram_read_energy_per_data + \
                mem_write * configs.dram_write_energy_per_data
    sram_energy = cur_buf_access * configs.gs_cur_buf_access_energy + \
                    halo_buf_access * configs.gs_halo_buf_access_energy + \
                    next_buf_access * configs.gs_next_buf_access_energy

    op_energy = add * configs.adder_energy + mul * configs.multiplier_energy
    total_energy = mem_energy + sram_energy + op_energy
    return latency, total_energy

def sim_fdmax(size, tile_size):
    gx, gy = size
    tx, ty = tile_size
    nx = ceil(gx / tx)
    ny = ceil(gy / ty)
    grid_n = gx * gy
    latency = nx * ny * (tx + 2)

    mem_read = 2 * nx * gy * (tx + 2)
    mem_write = grid_n

    sram_access = 2 * (mem_read + mem_write)
    fifo_access = 4 * nx * ny * tx

    add = 7 * grid_n + gx * ny
    mul = 4 * grid_n

    mem_energy = mem_read * configs.dram_read_energy_per_data + \
                mem_write * configs.dram_write_energy_per_data
    sram_energy = sram_access * configs.fdmax_sram_access_energy
    fifo_energy = fifo_access * configs.fdmax_fifo_access_energy

    op_energy = add * configs.adder_energy + mul * configs.multiplier_energy
    total_energy = mem_energy + sram_energy + fifo_energy + op_energy

    return latency, total_energy

def sim_spadix(size, tile_size):
    gx, gy, gz = size
    tx, ty, tz = tile_size
    nx = ceil(gx / tx)
    ny = ceil(gy / ty)
    nz = ceil(gz / tz)
    grid_n = gx * gy * gz

    latency = 2 * nx * ny * nz * (tz + 2)

    mem_read = nx * (tx + 2) * gy * nz * (tz + 2)
    mem_write = grid_n

    sram_access = 2 * (mem_read + mem_write)
    fifo_access = 4 * gx * ny * nz * (tz + 2)

    add = 7 * grid_n + gx * ny * gz
    mul = 4 * grid_n + gx * ny * gz

    mem_energy = mem_read * configs.dram_read_energy_per_data + \
                mem_write * configs.dram_write_energy_per_data
    sram_energy = sram_access * configs.spadix_sram_access_energy
    fifo_energy = fifo_access * configs.spadix_fifo_access_energy
    op_energy = add * configs.adder_energy + mul * configs.multiplier_energy

    total_energy = mem_energy + sram_energy + fifo_energy + op_energy

    return latency, total_energy

def sim_2d():
    size_2d_list = [
        [1024, 1024],
        [2048, 2048],
        [4096, 4096]
    ]
    gs_n = 204
    jacobi_n = 398
    hybrid_n = 301

    fdmax_tile_n = (512, 64)
    gs2d_tile_n = (8, 8)
    for size in size_2d_list:
        gs2d_cycles, gs2d_energy = sim_gs2d(size, gs2d_tile_n)
        fdmax_cycles, fdmax_energy = sim_fdmax(size, fdmax_tile_n)
        # gs_latency = gs2d_cycles / clock_freq / 1e6
        # jacobi_latency = jacobi_n * fdmax_cycles / clock_freq / 1e6
        # hybrid_latency = hybrid_n * fdmax_cycles / clock_freq / 1e6
        gs_latency = gs2d_cycles / clock_freq / 1e6
        jacobi_latency = fdmax_cycles / clock_freq / 1e6
        hybrid_latency = fdmax_cycles / clock_freq / 1e6
        gs_energy = gs2d_energy / 1e9
        jacobi_energy = fdmax_energy / 1e9
        hybrid_energy = fdmax_energy / 1e9
        print(f"Size: {size}")
        print(f"Cycles: GS {gs2d_cycles}, FDMAX {fdmax_cycles}")
        print(f"Latency: GS {gs_latency} ms, FDMAX-J {jacobi_latency} ms, FDMAX-H {hybrid_latency} ms")
        print(f"Energy: GS {gs_energy} mJ, FDMAX-J {jacobi_energy} mJ, FDMAX-H {hybrid_energy} mJ")


def sim_3d():
    size_3d_list = [
        [64, 64, 64],
        [128, 128, 128],
        [256, 256, 256],
    ]
    spadix_tile_n = (8, 16, 64)
    gs3d_tile_n = (8, 8)

    gs_n = 261
    rb_n = 384
    jacobi_n = 511
    for size in size_3d_list:
        gs3d_cycles, gs3d_energy = sim_gs3d(size, gs3d_tile_n)
        spadix_cycles, spadix_energy = sim_spadix(size, spadix_tile_n)

        gs_latency = gs3d_cycles / clock_freq / 1e6
        jacobi_latency = spadix_cycles / clock_freq / 1e6
        rb_latency = spadix_cycles / clock_freq / 1e6

        gs_energy = gs3d_energy / 1e9
        rb_energy = spadix_energy / 1e9
        jacobi_energy = spadix_energy / 1e9
        print(f"Size: {size}")
        print(f"Latency: GS {gs_latency} ms, SPADIX-RB {rb_latency} ms, SPADIX-J {jacobi_latency} ms")
        print(f"Energy: GS {gs_energy} mJ, SPADIX-RB {rb_energy} mJ, SPADIX-J {jacobi_energy} mJ")


if __name__ == "__main__":
    clock_freq = 0.2
    sim_2d()
    sim_3d()

