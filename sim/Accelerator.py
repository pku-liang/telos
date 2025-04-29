import simpy
import math
import progressbar
from loguru import logger
from .Memory import *
from .PEArray import *
from .HEU import *
from .info import *
from .Container import *
import numpy as np

class Accelerator:
    def __init__(self, env, cfg, data, progressbar=False):
        self.env = env
        self.cfg = cfg
        self.data = data
        self.progressbar = progressbar
        self.compute_type = cfg["ComputeType"]

        # Memory
        # Due to different data depths, DRAM is set separately for A
        self.domain_spmat_data = DomainSpMatData(env, cfg, data)
        self.domain_spmat_dram = DRAM(env, "domain_spmat_data", cfg["Mem"]["SpMat_DRAM_BW"], self.domain_spmat_data)
        self.buffers = Buffers(env, cfg)
        # PE array
        self.tile_X = cfg["Arch"]["NumPEs"][0]
        self.tile_Y = cfg["Arch"]["NumPEs"][1]
        self.boundary_x = [BoundaryPorts(env) for _ in range(self.tile_Y)]
        self.boundary_y = [BoundaryPorts(env) for _ in range(self.tile_X)]

        if self.compute_type == "sptrsv":
            self.domain_data = DomainData(env, cfg, data)
            self.domain_dram = DRAM(env, "domain_data", cfg["Mem"]["Domain_DRAM_BW"], self.domain_data)

            self.halo_data_X = HaloData(env, cfg, data, 0)
            self.halo_dram_X = DRAM(env, "halo_x", cfg["Mem"]["Halo_DRAM_BW"], self.halo_data_X)
            self.halo_data_Y = HaloData(env, cfg, data, 1)
            self.halo_dram_Y = DRAM(env, "halo_y", cfg["Mem"]["Halo_DRAM_BW"], self.halo_data_Y)

            self.PE_Array = PEArray(env, cfg, self.buffers, self.domain_data, self.domain_spmat_data, [self.boundary_x, self.boundary_y])

            self.HEU_X = HEU(env, cfg, self.buffers, self.halo_data_X, self.boundary_x, 0)  # X:0, Y:1
            self.HEU_Y = HEU(env, cfg, self.buffers, self.halo_data_Y, self.boundary_y, 1)
        else:
            self.domain_data = DomainDataSpMV(env, cfg, data)
            self.domain_dram = DRAM(env, "domain_data", cfg["Mem"]["Domain_DRAM_BW"], self.domain_data)

            self.halo_data_X = HaloDataSpMV(env, cfg, data, 0)
            self.halo_dram_X = DRAM(env, "halo_x", cfg["Mem"]["Halo_DRAM_BW"], self.halo_data_X)
            self.halo_data_Y = HaloDataSpMV(env, cfg, data, 1)
            self.halo_dram_Y = DRAM(env, "halo_y", cfg["Mem"]["Halo_DRAM_BW"], self.halo_data_Y)
            self.halo_data_X_inv = HaloDataSpMV(env, cfg, data, 2)
            self.halo_dram_X_inv = DRAM(env, "halo_x_inv", cfg["Mem"]["Halo_DRAM_BW"], self.halo_data_X_inv)
            self.halo_data_Y_inv = HaloDataSpMV(env, cfg, data, 3)
            self.halo_dram_Y_inv = DRAM(env, "halo_y_inv", cfg["Mem"]["Halo_DRAM_BW"], self.halo_data_Y_inv)

            self.boundary_x_inv = [BoundaryPorts(env) for _ in range(self.tile_Y)]
            self.boundary_y_inv = [BoundaryPorts(env) for _ in range(self.tile_X)]

            self.HEU_X = HEU(env, cfg, self.buffers, self.halo_data_X, self.boundary_x, 0)  # X:0, Y:1
            self.HEU_Y = HEU(env, cfg, self.buffers, self.halo_data_Y, self.boundary_y, 1)
            self.HEU_X_INV = HEU(env, cfg, self.buffers, self.halo_data_X_inv, self.boundary_x_inv, 2)
            self.HEU_Y_INV = HEU(env, cfg, self.buffers, self.halo_data_Y_inv, self.boundary_y_inv, 3)
            self.PE_Array = PEArray(env, cfg, self.buffers, self.domain_data, self.domain_spmat_data, [self.boundary_x, self.boundary_y, self.boundary_x_inv, self.boundary_y_inv])

        self.actions = [env.process(self.run())]

    def wait_for_finish(self):
        yield self.domain_dram.proc_write

    def run(self):
        bar = progressbar.ProgressBar(maxval=self.domain_data.iters * self.domain_data.Z_depth)
        if self.progressbar: bar.start()
        while True:
            yield self.env.timeout(1)  # Simulate processing time
            if self.progressbar: bar.update(self.domain_data.read_dim0_idx)

    def check_correctness(self):
        data = self.data["x"] if self.compute_type == "sptrsv" else self.data["result"]
        gt = np.ones(self.data["size"]) if self.compute_type == "sptrsv" else self.data["gt"]
        assert(data.shape == gt.shape)
        return np.array_equal(data, gt)

    def get_critical_delay(self):
        stencil_type = self.cfg["StencilType"]
        stencil_points = self.data['A'].shape[-1]
        vec_lanes = self.cfg['Arch']['VecLanes']
        dims = self.cfg['NumDims']

        if dims == 2:
            z_delay = 0
            mul_cycle = math.ceil(stencil_points / vec_lanes)
        else:
            mul_cycle = math.ceil(stencil_points / vec_lanes)
            if stencil_type == 1:
                z_delay = self.cfg['Delay']['Mul'] + self.cfg['Delay']['Add']
            else:
                z_delay = self.cfg['Delay']['Mul']

        scalar_delay = self.cfg['Delay']['Div'] + self.cfg['Delay']['Add']
        self_delay = scalar_delay + z_delay
        delay = max(self_delay, mul_cycle)
        return delay

    def compute_info(self):
        wall_clock_time = self.env.now / (self.cfg["Freq"] * 1e9)
        # add mul div
        delay = self.get_critical_delay()
        ideal_cycles = math.ceil(math.prod(self.data["size"]) / (self.tile_X * self.tile_Y)) * delay

        read_counter = self.domain_dram.read_counter + \
                        self.domain_spmat_dram.read_counter + \
                        self.halo_dram_X.read_counter + \
                        self.halo_dram_Y.read_counter
        write_counter = self.domain_dram.write_counter + \
                        self.domain_spmat_dram.write_counter + \
                        self.halo_dram_X.write_counter + \
                        self.halo_dram_Y.write_counter

        domain_dram_counter = self.domain_dram.read_counter + self.domain_dram.write_counter + \
                        self.domain_spmat_dram.read_counter + self.domain_spmat_dram.write_counter
        halo_dram_counter = self.halo_dram_X.read_counter + self.halo_dram_X.write_counter + \
                        self.halo_dram_Y.read_counter + self.halo_dram_Y.write_counter

        dram_energy = read_counter * self.cfg["Energy"]["DRAM_read_per_data"] + \
                      write_counter * self.cfg["Energy"]["DRAM_write_per_data"]

        domain_sram_counter = self.buffers.domain_vec_in.counter + self.buffers.domain_vec_out.counter
        halo_sram_counter = self.buffers.halo_vec_in.counter + self.buffers.halo_vec_out.counter
        diag_sram_counter = self.buffers.domain_diag_mtx.counter
        spmat_sram_counter = self.buffers.domain_mtx.counter

        # additional energy cost by DRAM
        halo_sram_dram_counter = halo_dram_counter
        domain_sram_dram_counter = self.domain_dram.read_counter // 2
        diag_sram_dram_counter = self.domain_dram.read_counter // 2
        spmat_sram_dram_counter = self.domain_spmat_dram.read_counter

        sram_energy = (domain_sram_counter + domain_sram_dram_counter + diag_sram_counter + diag_sram_dram_counter) * self.cfg["Energy"]["DomainVectorBuffer_per_data"] + \
                    (halo_sram_counter + halo_sram_dram_counter) * self.cfg["Energy"]["HaloVectorBuffer_per_data"] + \
                    (spmat_sram_counter + spmat_sram_dram_counter) * self.cfg["Energy"]["DomainSpMatBuffer_per_data"]
        sram_energy_wo_spmat = (domain_sram_counter + domain_sram_dram_counter + diag_sram_counter + diag_sram_dram_counter) * self.cfg["Energy"]["DomainVectorBuffer_per_data"] + \
                    (halo_sram_counter + halo_sram_dram_counter) * self.cfg["Energy"]["HaloVectorBuffer_per_data"]

        PE_mul, PE_div, PE_add = self.PE_Array.stat()
        HEU_add = self.HEU_X.add_counter + self.HEU_Y.add_counter
        add_energy = (PE_add + HEU_add)*self.cfg["Energy"]["Add"]
        mul_energy = PE_mul*self.cfg["Energy"]["Mul"]
        div_energy = PE_div*self.cfg["Energy"]["Div"]
        compute_energy = add_energy + mul_energy + div_energy
        total_energy = dram_energy + sram_energy + compute_energy

        domain_dram_bw = (domain_dram_counter*8/1e9)/wall_clock_time
        halo_dram_bw = (halo_dram_counter*8/1e9)/wall_clock_time
        total_dram_bw = domain_dram_bw + halo_dram_bw

        pe_util = ideal_cycles / self.env.now * 100

        summ_info = SummaryInfo(self.env.now, wall_clock_time, pe_util, delay, ideal_cycles, total_energy)
        dram_info = DRAMInfo(domain_dram_counter, halo_dram_counter, domain_dram_bw, halo_dram_bw, total_dram_bw, dram_energy)
        sram_info = SRAMInfo(domain_sram_counter, spmat_sram_counter, halo_sram_counter, diag_sram_counter, sram_energy, sram_energy_wo_spmat)
        compute_info = ComputeInfo(PE_add, PE_mul, PE_div, HEU_add, compute_energy)
        total_info = Info(summ_info, compute_info, dram_info, sram_info)
        return total_info

    def print(self, info: Info):
        print("="*80)
        print(" "*30, "SIMULATION REPORT", " "*30)
        print("="*80)
        print(f"Total Cycles: {self.env.now}")
        print(f"Wallclock Time: {info.summ.wall_clock_time * 1000:.2f} ms")
        print(f"Ideal delay: {info.summ.ideal_delay}, Ideal Cycles: {info.summ.ideal_cycles}")
        print(f"PE Utilization: {info.summ.pe_util:.2f}%")
        print(f"Total Energy: {info.summ.energy:.2f} pJ")
        print('-'*35, "DRAM", '-'*35)
        print(f"Domain Access Volume: {info.dram.domain_access},\t\t\tAverage BW: {info.dram.domain_bw:.2f} GB/s")
        print(f"Halo Access Volume: {info.dram.halo_access},\t\t\tAverage BW: {info.dram.halo_bw:.2f} GB/s")
        print(f"Total DRAM bandwidth: {info.dram.total_bw:.2f} GB/s")
        print(f"Energy: {info.dram.energy:.2f} pJ ({info.dram.energy / info.summ.energy * 100:.2f}%)")
        print('-'*35, "SRAM", '-'*35)
        print(f"Domain Vector Access Volume: {info.sram.domain_access},\t\tAverage BW: {(info.sram.domain_access*8/1e9)/info.summ.wall_clock_time:.2f} GB/s")
        print(f"Halo Vector Access Volume: {info.sram.halo_access},\t\tAverage BW: {(info.sram.halo_access*8/1e9)/info.summ.wall_clock_time:.2f} GB/s")
        print(f"Domain Matrix Access Volume: {info.sram.spmat_access},\t\tAverage BW: {(info.sram.spmat_access*8/1e9)/info.summ.wall_clock_time:.2f} GB/s")
        print(f"Domain Diagonal Access Volume: {info.sram.diag_access},\t\tAverage BW: {(info.sram.diag_access*8/1e9)/info.summ.wall_clock_time:.2f} GB/s")
        print(f"Energy: {info.sram.energy:.2f} pJ ({info.sram.energy / info.summ.energy * 100:.2f}%)")
        print('-'*34, "PE/HEU", '-'*34)
        print(f"PE Add: {info.compute.pe_add}, Mul: {info.compute.pe_mul}, Div: {info.compute.pe_div}")
        print(f"HEU Add: {info.compute.heu_add}")
        print(f"Energy: {info.compute.energy:.2f} pJ ({info.compute.energy / info.summ.energy * 100:.2f}%)")
        print("="*80 + "\n")