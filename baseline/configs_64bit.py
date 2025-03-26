# Bytes
data_width = 8

# DRAM cfg
# DRAM Type: HBM2 8GB 2stacks 16channels 128bit/channel 512GB/s
dram_bandwidth = 460
dram_read_energy_per_data = 100.53
dram_write_energy_per_data = 133.5

# SRAM access energy (per data)
alres_cache_read_energy = 12.07
alres_cache_write_energy = 12.24

# 20KB 8banks 64*10bit (1r 1w)
gs_cur_buf_access_energy = 13.5
# 16KB 32banks 128bit (1r 1w)
gs_offset_buf_access_energy = 7.93
gs_next_buf_access_energy = 7.93
# 4KB 32banks 64bit (1r 1w)
gs_halo_buf_access_energy = 5.9

# 8KB 32banks 128bit (1r 1w)
fdmax_sram_access_energy = 7.9
# 4KB 1bank (512 entries) 64bit (1r 1w)
fdmax_fifo_access_energy = 2.4

# 30KB 8banks 64*10bit (1r 1w)
spadix_sram_access_energy = 13.8
# 2KB 8banks 64bit (1r 1w)
spadix_fifo_access_energy = 2.9

# fdmax_sram_read_energy = 2.55
# fdmax_sram_write_energy = 2.60
# fdmax_fifo_read_energy = 1.26
# fdmax_fifo_write_energy = 1.34
# spadix_sram_read_energy = 5.11
# spadix_sram_write_energy = 5.22
# spadix_fifo_read_energy = 0.29
# spadix_fifo_write_energy = 0.39

# 64bit op
adder_energy = 0.79
multiplier_energy = 5.63
divisor_energy = 50.71
