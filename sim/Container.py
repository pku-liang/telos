import simpy
import math
from loguru import logger
from .Memory import *
from .PEArray import *
from .HEU import *
from .mask_const import *

class DomainDataSpMV:
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
        self.dim0_extent = self.data['x'].shape[0]
        self.iters = math.ceil(self.dim0_extent / self.Z_depth)
        # double buffering with the size of 2*depth
        self.domain_x_in = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        self.domain_b_in = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        self.domain_b_out = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        self.domain_index = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        # index
        self.put_dim0_idx = 0
        self.put_dim0_idx_b = 0
        self.read_dim0_idx = 0
        self.stencil_type = cfg["StencilType"]

    def get_read_size(self):
        depth_x = min((self.dim0_extent - self.put_dim0_idx), self.Z_depth)
        depth_b = min((self.dim0_extent - self.put_dim0_idx_b), self.Z_depth)
        access_size = (self.tile_X * self.tile_Y) * (depth_x + depth_b)
        return access_size

    def get_write_size(self):
        depth = min((self.dim0_extent - self.read_dim0_idx), self.Z_depth)
        access_size = (self.tile_X * self.tile_Y) * depth
        return access_size

    def get_previous(self):
        # Get updated b from the current execution
        for _ in range(int(min(self.Z_depth, self.dim0_extent - self.read_dim0_idx))):
            for i in range(self.tile_X):
                for j in range(self.tile_Y):
                    value, ijk_index, index = yield self.domain_b_out[i][j].get()
                    self.data["b"][index] = value
                    self.data["b_valid"][index] -= CUR_M

                    # compute finished, write back to result
                    if self.data["b_valid"][index] == 0:
                        if ijk_index[0] < self.size[0] and ijk_index[1] < self.size[1] and \
                            ijk_index[2] < self.size[2]:
                            self.data["result"][ijk_index] = value
                            logger.trace(f"(Cycle {self.env.now}) DomainData: write back result{ijk_index}={value} from PE({i}, {j})")
                        else:
                            logger.trace(f"(Cycle {self.env.now}) DomainData: Discarded out of bound result{ijk_index}={value} from PE({i}, {j})")
                    else:
                        logger.trace(f"(Cycle {self.env.now}) DomainData: get updated b{index}={value}, ijk={ijk_index} from PE({i}, {j}), b_valid={bin(self.data['b_valid'][index])}")

            self.read_dim0_idx += 1

    def put_next(self):
        proc_put_b = self.env.process(self.put_next_b())
        proc_put_x = self.env.process(self.put_next_x())
        yield proc_put_b & proc_put_x

    def put_next_b(self):
        for _ in range(int(min(self.Z_depth, self.dim0_extent - self.put_dim0_idx_b))):
            for i in range(self.tile_X):
                for j in range(self.tile_Y):
                    if self.stencil_type == 0: # out_i out_j cur out_j_inv out_i_inv
                        b_mask = OUT_I_M | OUT_J_M
                    elif self.stencil_type == 1: # agg_i out_i agg_j out_j cur out_j_inv agg_j_inv out_i_inv agg_i_inv
                        b_mask = AGG_I_M | OUT_I_M | AGG_J_M | OUT_J_M
                    elif self.stencil_type == 2: # out_i agg_i out_j cur out_j_inv agg_i_inv out_i_inv
                        b_mask = OUT_I_M | AGG_I_M | OUT_J_M
                    elif self.stencil_type == 3: # agg_j out_i agg_i out_j cur out_j_inv agg_i_inv out_i_inv agg_j_inv
                        b_mask = AGG_J_M | OUT_I_M | AGG_I_M | OUT_J_M

                    # wait until all the previous data is updated
                    while self.data["b_valid"][self.put_dim0_idx_b][i][j] & b_mask != 0:
                        yield self.env.timeout(1)

                    ijk_index = self.data["ijk"][self.put_dim0_idx_b][i][j]
                    index = (self.put_dim0_idx_b, i, j)
                    yield self.domain_index[i][j].put((ijk_index, index))
                    yield self.domain_b_in[i][j].put(self.data["b"][self.put_dim0_idx_b][i][j])

                    logger.trace(f"(Cycle {self.env.now}) DomainData: put b of row {self.data['ijk'][self.put_dim0_idx_b][i][j]} to PE({i}, {j})")
            self.put_dim0_idx_b += 1

    def put_next_x(self):
        # Put data for the next execution
        for _ in range(int(min(self.Z_depth, self.dim0_extent - self.put_dim0_idx))):
            for i in range(self.tile_X):
                for j in range(self.tile_Y):
                    ijk_index = self.data["ijk"][self.put_dim0_idx][i][j]
                    yield self.domain_x_in[i][j].put((self.data["x"][self.put_dim0_idx][i][j], ijk_index))
                    logger.trace(f"(Cycle {self.env.now}) DomainData: put x & ijk of row {self.data['ijk'][self.put_dim0_idx][i][j]} to PE({i}, {j})")
            self.put_dim0_idx += 1


class HaloDataSpMV:
    def __init__(self, env, cfg, data, position):
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

            # compute finished, write back to result
            if self.data["b_valid"][index] == 0:
                if ijk_index[0] < self.size[0] and ijk_index[1] < self.size[1] and \
                    ijk_index[2] < self.size[2]:
                    self.data["result"][ijk_index] = value
                    logger.trace(f"(Cycle {self.env.now}) HaloData: write back result{ijk_index}={value} from HEU ({self.position}, {i})")
                else:
                    logger.trace(f"(Cycle {self.env.now}) HaloData: Discarded out of bound result{ijk_index}={value} from HEU ({self.position}, {i})")
            else:
                logger.trace(f"(Cycle {self.env.now}) HaloData: get updated b{index}={value}, ijk={ijk_index} from HEU ({self.position}, {i}) with valid state={bin(self.data['b_valid'][index])}")

    def get_previous(self):
        yield self.env.timeout(1)

    def put_next(self):
        for _ in range(int(min(self.Z_depth, self.dim0_extent - self.put_dim0_idx))):
            for i in range(self.halo_points):
                # HEU position
                # [out_i, out_j, out_i_inv, out_j_inv]
                halo_list = ["halo_x", "halo_y", "halo_x_inv", "halo_y_inv"]
                halo = halo_list[self.position]
                index = self.data[halo][self.put_dim0_idx][i]

                # b_valid共8bit 从低bit到高bit
                # agg_i out_i agg_j out_j cur
                # agg_i_inv out_i_inv agg_j_inv out_j_inv

                # The scheduling method of spmv tiles adopts a column-priority approach
                # The scheduling order of various spmv stencils
                # 0: out_i out_j out_j_inv out_i_inv
                # 1: agg_i out_i agg_j out_j out_j_inv agg_j_inv out_i_inv agg_i_inv
                # 2: out_i agg_i out_j out_j_inv agg_i_inv out_i_inv
                # 3: agg_j out_i agg_i out_j out_j_inv agg_i_inv out_i_inv agg_j_inv

                if index[0] >= 0:
                    if self.stencil_type == 0: # out_i out_j cur out_j_inv out_i_inv
                        mask_list = gen_mask([OUT_I_M, OUT_J_M, CUR_M, OUT_J_INV_M])
                        if self.position == 1: # out_j
                            while self.data["b_valid"][index] & mask_list[0] != 0:
                                yield self.env.timeout(1)
                        elif self.position == 3: # out_j_inv
                            while self.data["b_valid"][index] & mask_list[2] != 0:
                                yield self.env.timeout(1)
                        elif self.position == 2: # out_i_inv
                            while self.data["b_valid"][index] & mask_list[3] != 0:
                                yield self.env.timeout(1)

                    elif self.stencil_type == 1: # agg_i out_i agg_j out_j cur out_j_inv agg_j_inv out_i_inv agg_i_inv
                        mask_list = gen_mask([AGG_I_M, OUT_I_M, AGG_J_M, OUT_J_M, CUR_M, OUT_J_INV_M, AGG_J_INV_M, OUT_I_INV_M])
                        if self.position == 0 and i % 2 == 0: # out_i
                            while self.data["b_valid"][index] & mask_list[0] != 0:
                                yield self.env.timeout(1)
                        elif self.position == 1 and i % 2 != 0: # agg_j
                            while self.data["b_valid"][index] & mask_list[1] != 0:
                                yield self.env.timeout(1)
                        elif self.position == 1 and i % 2 == 0: # out_j
                            while self.data["b_valid"][index] & mask_list[2] != 0:
                                yield self.env.timeout(1)
                        elif self.position == 3 and i % 2 == 0: # out_j_inv
                            while self.data["b_valid"][index] & mask_list[4] != 0:
                                yield self.env.timeout(1)
                        elif self.position == 3 and i % 2 != 0: # agg_j_inv
                            while self.data["b_valid"][index] & mask_list[5] != 0:
                                yield self.env.timeout(1)
                        elif self.position == 2 and i % 2 == 0: # out_i_inv
                            while self.data["b_valid"][index] & mask_list[6] != 0:
                                yield self.env.timeout(1)
                        elif self.position == 2 and i % 2 != 0: # agg_i_inv
                            while self.data["b_valid"][index] & mask_list[7] != 0:
                                yield self.env.timeout(1)

                    elif self.stencil_type == 2: # out_i agg_i out_j cur out_j_inv agg_i_inv out_i_inv
                        mask_list = gen_mask([OUT_I_M, AGG_I_M, OUT_J_M, CUR_M, OUT_J_INV_M, AGG_I_INV_M])
                        if self.position == 0 and i == 0: # agg_i
                            while self.data["b_valid"][index] & mask_list[0] != 0:
                                yield self.env.timeout(1)
                        elif self.position == 1: # out_j
                            while self.data["b_valid"][index] & mask_list[1] != 0:
                                yield self.env.timeout(1)
                        elif self.position == 3: # out_j_inv
                            while self.data["b_valid"][index] & mask_list[3] != 0:
                                yield self.env.timeout(1)
                        elif self.position == 2 and i != 0: # agg_i_inv
                            while self.data["b_valid"][index] & mask_list[4] != 0:
                                yield self.env.timeout(1)
                        elif self.position == 2 and i == 0: # out_i_inv
                            while self.data["b_valid"][index] & mask_list[5] != 0:
                                yield self.env.timeout(1)

                    elif self.stencil_type == 3: # agg_j out_i agg_i out_j cur out_j_inv agg_i_inv out_i_inv agg_j_inv
                        mask_list = gen_mask([AGG_J_M, OUT_I_M, AGG_I_M, OUT_J_M, CUR_M, OUT_J_INV_M, AGG_I_INV_M, OUT_I_INV_M])
                        if self.position == 0 and i != 0: # out_i
                            while self.data["b_valid"][index] & mask_list[0] != 0:
                                yield self.env.timeout(1)
                        elif self.position == 0 and i == 0: # agg_i
                            while self.data["b_valid"][index] & mask_list[1] != 0:
                                yield self.env.timeout(1)
                        elif self.position == 1 and i == 0: # out_j
                            while self.data["b_valid"][index] & mask_list[2] != 0:
                                yield self.env.timeout(1)
                        elif self.position == 3 and i != 0: # out_j_inv
                            while self.data["b_valid"][index] & mask_list[4] != 0:
                                yield self.env.timeout(1)
                        elif self.position == 2 and i != 0: # agg_i_inv
                            while self.data["b_valid"][index] & mask_list[5] != 0:
                                yield self.env.timeout(1)
                        elif self.position == 2 and i == 0: # out_i_inv
                            while self.data["b_valid"][index] & mask_list[6] != 0:
                                yield self.env.timeout(1)
                        elif self.position == 3 and i == 0: # agg_j_inv
                            while self.data["b_valid"][index] & mask_list[7] != 0:
                                yield self.env.timeout(1)

                value = self.data["b"][index]
                ijk = self.data["ijk"][index]
                yield self.halo_vec_in[i].put((value, index, self.data["b_valid"][index]))
                yield self.halo_idx_in[i].put(ijk)
                if index[0] >= 0:
                    logger.trace(f"(Cycle {self.env.now}) HaloData: put b{ijk}={value} to HEU ({self.position}, {i}) with valid state={bin(self.data['b_valid'][index])}")
                else:
                    logger.trace(f"(Cycle {self.env.now}) HaloData: put invalid b to HEU ({self.position}, {i}) with valid state={bin(self.data['b_valid'][index])}")

            self.put_dim0_idx += 1

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
            logger.trace(f"(Cycle {self.env.now}) HaloData: get b{ijk_index}={value} from HEU ({self.position}, {i}) with valid state={bin(self.data['b_valid'][index])}")

    def get_previous(self):
        yield self.env.timeout(1)

    def put_next(self):
        for _ in range(int(min(self.Z_depth, self.dim0_extent - self.put_dim0_idx))):
            for i in range(self.halo_points):
                halo = "halo_x" if self.position == 0 else "halo_y"
                index = self.data[halo][self.put_dim0_idx][i]
                ijk_index = self.data["ijk"][index]

                # b_valid consists of 4 bits:
                # bit[0] indicates the need for an update via out_j,
                # bit[1] indicates the need for an update via out_i,
                # bit[2] indicates the need for an update via agg_i,
                # bit[3] indicates the need for an update via agg_j.
                # The priority is from higher bits to lower bits, determined by the column-priority scheduling order.
                # agg_j agg_i out_i out_j
                if index[0] >= 0:
                    if self.stencil_type == 0: # out_i -> out_j
                        mask_list = gen_mask([OUT_I_M])
                        if self.position == 1: # out_j
                            while self.data["b_valid"][index] & mask_list[0] != 0: # wait out_i
                                yield self.env.timeout(1)

                    elif self.stencil_type == 1: # agg_i -> agg_j -> out_i -> out_j
                        if self.position == 1: # (agg_j | out_j)
                            while self.data["b_valid"][index] & (AGG_I_M | OUT_I_M) != 0: # wait agg_i & out_i
                                yield self.env.timeout(1)

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
                    logger.trace(f"(Cycle {self.env.now}) HaloData: put b{ijk}={value} to HEU ({self.position}, {i}) with valid state={bin(self.data['b_valid'][index])}")
                else:
                    logger.trace(f"(Cycle {self.env.now}) HaloData: put invalid b to HEU ({self.position}, {i}) with valid state={bin(self.data['b_valid'][index])}")

            self.put_dim0_idx += 1
