import simpy
import math
import progressbar
from Memory import *
from PEArray import *
from HEU import *
from loguru import logger
from info import *
from DataSpMV import *
import numpy as np

class DomainData:
    def __init__(self, env, cfg, data):
        self.env = env
        self.cfg = cfg
        self.data = data
        self.readonly = False
        # size
        self.size = self.data["size"]
        self.tile_X = cfg["Arch"]["NumPEs"][0]
        self.tile_Y = cfg["Arch"]["NumPEs"][1]
        self.Z_depth = self.cfg["Mem"]["Depth"]
        self.dim0_extent = self.data['diag_A'].shape[0]
        self.iters = math.ceil(self.dim0_extent / self.Z_depth)
        # double buffering with the size of 2*depth
        self.domain_vec_in = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        self.domain_vec_out = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        self.domain_diag_mtx = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        self.domain_index = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        # index
        self.put_dim0_idx = 0
        self.put_dim0_idx_b = 0
        self.read_dim0_idx = 0

    def get_read_size(self):
        depth = min((self.dim0_extent - self.put_dim0_idx), self.Z_depth)
        access_size = (self.tile_X * self.tile_Y) * depth
        return 2 * access_size

    def get_write_size(self):
        depth = min((self.dim0_extent - self.read_dim0_idx), self.Z_depth)
        access_size = (self.tile_X * self.tile_Y) * depth
        return access_size

    def get_previous(self):
        # Get x from the current execution
        for _ in range(int(min(self.Z_depth, self.dim0_extent - self.read_dim0_idx))):
            for i in range(self.tile_X):
                for j in range(self.tile_Y):
                    value, ijk_index = yield self.domain_vec_out[i][j].get()
                    if ijk_index[0] < self.size[0] and ijk_index[1] < self.size[1] and \
                        ijk_index[2] < self.size[2]:
                        self.data["x"][ijk_index] = value
                        logger.trace(f"(Cycle {self.env.now}) DomainData: get x{ijk_index}={value} from PE({i}, {j})")
                    else:
                        logger.trace(f"(Cycle {self.env.now}) DomainData: get out-of-bound x{ijk_index}={value} from PE({i}, {j}), ignored")
            self.read_dim0_idx += 1

    def put_next(self):
        proc_put_b = self.env.process(self.put_next_b())
        proc_put_diagA = self.env.process(self.put_next_diagA())
        yield proc_put_b & proc_put_diagA

    def put_next_b(self):
        for _ in range(int(min(self.Z_depth, self.dim0_extent - self.put_dim0_idx_b))):
            for i in range(self.tile_X):
                for j in range(self.tile_Y):
                    # wait until b is valid (on halo and wholly updated by HEU)
                    while self.data["b_valid"][self.put_dim0_idx_b][i][j] > 0:
                        # ijk = self.data["ijk"][self.put_dim0_idx_b][i][j]
                        # print(f"waiting for b{ijk} to be valid, current valid state={bin(self.data["b_valid"][self.put_dim0_idx_b][i][j])}")
                        yield self.env.timeout(1)
                    yield self.domain_vec_in[i][j].put(self.data["b"][self.put_dim0_idx_b][i][j])
                    logger.trace(f"(Cycle {self.env.now}) DomainData: put b of row {self.data['ijk'][self.put_dim0_idx_b][i][j]} to PE({i}, {j})")
            self.put_dim0_idx_b += 1

    def put_next_diagA(self):
        # Put data for the next execution
        for _ in range(int(min(self.Z_depth, self.dim0_extent - self.put_dim0_idx))):
            for i in range(self.tile_X):
                for j in range(self.tile_Y):
                    yield self.domain_diag_mtx[i][j].put(self.data["diag_A"][self.put_dim0_idx][i][j])
                    yield self.domain_index[i][j].put(self.data["ijk"][self.put_dim0_idx][i][j])
                    logger.trace(f"(Cycle {self.env.now}) DomainData: put Ajj & ijk of row {self.data['ijk'][self.put_dim0_idx][i][j]} to PE({i}, {j})")
            self.put_dim0_idx += 1


class DomainSpMatData:
    def __init__(self, env, cfg, data):
        self.env = env
        self.cfg = cfg
        self.data = data
        self.readonly = True
        # size
        self.size = self.data["size"]
        self.tile_X = cfg["Arch"]["NumPEs"][0]
        self.tile_Y = cfg["Arch"]["NumPEs"][1]
        self.Z_depth = self.cfg["Mem"]["Depth"]
        self.dim0_extent = self.data['A'].shape[0]
        self.iters = math.ceil(self.dim0_extent / self.Z_depth)
        # double buffering with the size of 2*depth
        self.domain_mtx = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        # index
        self.put_dim0_idx = 0

    def get_read_size(self):
        depth = min((self.dim0_extent - self.put_dim0_idx), self.Z_depth)
        access_size = (self.tile_X * self.tile_Y) * depth
        num_points = self.data["A"].shape[-1]
        return num_points * access_size

    def put_next(self):
        # Put data for the next execution
        for _ in range(int(min(self.Z_depth, self.dim0_extent - self.put_dim0_idx))):
            for i in range(self.tile_X):
                for j in range(self.tile_Y):
                    yield self.domain_mtx[i][j].put((self.data["A"][self.put_dim0_idx][i][j], self.data['A_valid'][self.put_dim0_idx]))
                    logger.trace(f"(Cycle {self.env.now}) DomainSpMatData: put spmat data of row to PE({i}, {j}) with index={self.put_dim0_idx}")
            self.put_dim0_idx += 1



class HaloData:
    def __init__(self, env, cfg, data, position): # X:0, Y:1
        self.env = env
        self.cfg = cfg
        self.data = data
        self.position = position
        self.readonly = False
        # size
        self.size = self.data["size"]
        self.tile_X = cfg["Arch"]["NumPEs"][0]
        self.tile_Y = cfg["Arch"]["NumPEs"][1]
        self.Z_depth = self.cfg["Mem"]["Depth"]

        self.dim0_extent = math.ceil(self.size[0] / self.tile_X) * math.ceil(self.size[1] / self.tile_Y) * int(self.size[2])
        self.iters = math.ceil(self.dim0_extent / self.Z_depth)
        self.halo_points = get_num_halo_points(cfg["StencilType"], self.tile_X, self.tile_Y, cfg["ComputeType"])[self.position]
        # double buffering with the size of 2*depth
        self.halo_vec_in = [simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.halo_points)]
        self.halo_idx_in = [simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.halo_points)]
        self.halo_vec_out = [simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.halo_points)]
        self.halo_idx_out = [simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.halo_points)]
        # index
        self.put_dim0_idx = 0
        self.read_dim0_idx = 0
        self.stencil_type = cfg["StencilType"]
        self.actions = [self.env.process(self.get_previous_proc(i)) for i in range(self.halo_points)]

    def get_read_size(self):
        depth = min((self.dim0_extent - self.put_dim0_idx), self.Z_depth)
        return self.halo_points * depth

    def get_write_size(self):
        depth = min((self.dim0_extent - self.read_dim0_idx), self.Z_depth)
        return self.halo_points * depth

    def get_previous_proc(self, i):
        for _ in range(self.dim0_extent):
            logger.trace(f"(Cycle {self.env.now}) HaloData: HEU ({self.position}, {i}) waiting to get b from halo_vec")
            value, index, b_valid = yield self.halo_vec_out[i].get()
            ijk_index = yield self.halo_idx_out[i].get()
            self.data["b"][index] = value
            self.data["b_valid"][index] = b_valid
            logger.trace(f"(Cycle {self.env.now}) HaloData: get b{ijk_index}={value} from HEU ({self.position}, {i}) with valid state={bin(self.data["b_valid"][index])}")

    def get_previous(self):
        yield self.env.timeout(1)

    def put_next(self):
        for _ in range(int(min(self.Z_depth, self.dim0_extent - self.put_dim0_idx))):
            for i in range(self.halo_points):
                halo = "halo_x" if self.position == 0 else "halo_y"
                index = self.data[halo][self.put_dim0_idx][i]
                ijk_index = self.data["ijk"][index]

                # b_valid共4bit，bit[0]表示需要通过out_j更新，bit[1]表示需要通过out_i更新
                # bit[2]表示需要通过agg_i更新，bit[3]表示需要通过agg_j更新
                # 优先级从高bit到低bit，这是按列优先的调度顺序决定的
                # agg_j agg_i out_i out_j
                if index[0] >= 0:
                    if self.stencil_type == 0: # out_i -> out_j
                        mask_list = gen_mask([OUT_I_M])
                        if self.position == 1: # out_j
                            while self.data["b_valid"][index] & mask_list[0] != 0: # wait out_i
                                # print(f"Waiting b{ijk_index}, b_valid={b_valid}")
                                yield self.env.timeout(1)
                                # print("after waiting")

                    elif self.stencil_type == 1: # agg_i -> agg_j -> out_i -> out_j
                        if self.position == 1: # (agg_j | out_j)
                            while self.data["b_valid"][index] & (AGG_I_M | OUT_I_M) != 0: # wait agg_i & out_i
                                yield self.env.timeout(1)
                        # mask_list = gen_mask([AGG_I_M, AGG_J_M, OUT_I_M])
                        # if self.position == 1 and i % 2 != 0: # agg_j
                        #     while self.data["b_valid"][index] & mask_list[0] != 0: # wait agg_i
                        #         yield self.env.timeout(1)
                        # elif self.position == 0 and i % 2 == 0: # out_i
                        #     while self.data["b_valid"][index] & mask_list[1] != 0:
                        #         yield self.env.timeout(1)
                        # elif self.position == 1 and i % 2 == 0: # out_j
                        #     while self.data["b_valid"][index] & mask_list[2] != 0:
                        #         yield self.env.timeout(1)

                    elif self.stencil_type == 2: # agg_i -> out_i -> out_j
                        mask_list = gen_mask([AGG_I_M, OUT_I_M])
                        if self.position == 0 and i == 0: # only out_i
                            while self.data["b_valid"][index] & mask_list[0] != 0: # wait agg_i
                                yield self.env.timeout(1)
                        elif self.position == 1: # out_j
                            while self.data["b_valid"][index] & mask_list[1] != 0: # wait agg_i & out_i
                                yield self.env.timeout(1)

                    elif self.stencil_type == 3: # agg_j -> agg_i -> out_i -> out_j
                        ijk = self.data["ijk"][index]
                        mask_list = gen_mask([AGG_J_M, AGG_I_M, OUT_I_M])
                        if self.position == 0 and i != 0: # agg_i
                            while self.data["b_valid"][index] & mask_list[0] != 0: # wait agg_j
                                yield self.env.timeout(1)
                        elif self.position == 0 and i == 0: # out_i
                            while self.data["b_valid"][index] & mask_list[1] != 0: # wait agg_i & agg_j
                                yield self.env.timeout(1)
                        elif self.position == 1 and i % 2 == 0: # out_j
                            while self.data["b_valid"][index] & mask_list[2] != 0: # wait agg_ij & out_i
                                yield self.env.timeout(1)

                value = self.data["b"][index]
                ijk = self.data["ijk"][index]
                yield self.halo_vec_in[i].put((value, index, self.data["b_valid"][index]))
                yield self.halo_idx_in[i].put(ijk)
                if index[0] >= 0:
                    logger.trace(f"(Cycle {self.env.now}) HaloData: put b{ijk}={value} to HEU ({self.position}, {i}) with valid state={bin(self.data["b_valid"][index])}")
                else:
                    logger.trace(f"(Cycle {self.env.now}) HaloData: put invalid b to HEU ({self.position}, {i}) with valid state={bin(self.data["b_valid"][index])}")

            self.put_dim0_idx += 1


class Accelerator:
    def __init__(self, env, cfg, data, progressbar=False):
        self.env = env
        self.cfg = cfg
        self.data = data
        self.progressbar = progressbar
        self.compute_type = cfg["ComputeType"]

        # Memory
        # 由于数据深度不同所以为A单独设置DRAM
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
        # if self.compute_type == "spmv":
        #     for i in range(data.shape[0]):
        #         for j in range(data.shape[1]):
        #             for k in range(data.shape[2]):
        #                 assert(data[i][j][k] != 0)
        #                 if data[i][j][k] != gt[i][j][k]:
        #                     print(f"The value {data[i][j][k]} of index({i}, {j}, {k}) != {gt[i][j][k]}")
        #                     return False
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