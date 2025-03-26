# Bytes
data_width = 8

# DRAM cfg
# DRAM Type: HBM2 8GB 2stacks 16channels 128bit/channel 460GB/s
dram_bandwidth = 460
dram_read_energy_per_data = 50.28
dram_write_energy_per_data = 66.75

# before scale (8*8)
# 4KB 32banks 64bit (1r 1w)
# 2KB 1bank (512 entries) 32bit (1r 1w)
# 15KB 8banks 32*10bit (1r 1w)
# 1KB 8banks 32bit (1r 1w)
gs_cur_buf_access_energy = 3.73
gs_offset_buf_access_energy = 2.95
gs_next_buf_access_energy = 2.95
gs_halo_buf_access_energy = 2.03

fdmax_sram_access_energy = 2.95
fdmax_fifo_access_energy = 1.14
spadix_sram_access_energy = 3.87
spadix_fifo_access_energy = 1.03

# after scale (12 * 12)
# fdmax_sram_access_energy = 5.96
# fdmax_fifo_access_energy = 1.14
# spadix_sram_access_energy = 7.6
# spadix_fifo_access_energy = 1.35

# 32bit op
adder_energy = 0.37
multiplier_energy = 1.22