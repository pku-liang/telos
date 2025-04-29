import math
from loguru import logger
from .PreprocessSpMV import get_num_halo_points
from .mask_const import *

class HEU:
    def __init__(self, env, cfg, bufs, data, bd_ports, position):
        self.env = env
        self.cfg = cfg
        self.bufs = bufs
        self.data = data
        self.boundary_ports = bd_ports
        self.position = position

        self.num_PEs = cfg["Arch"]["NumPEs"]
        self.num_lanes = cfg["Arch"]["HaloLanes"][position]
        self.stencil_type = cfg["StencilType"]
        self.dims = cfg["NumDims"]
        self.compute_type = cfg["ComputeType"]
        self.halo_points = get_num_halo_points(self.stencil_type, self.num_PEs[0], self.num_PEs[1], self.compute_type)[self.position]
        self.add_counter = 0
        # the number of processes is determined by the number of halo points
        self.actions = [env.process(self.run(i)) for i in range(self.halo_points)]

    def run(self, i):
        while True:
            tick = self.env.now
            yield self.env.process(self.bufs.halo_vec_in.access(1))
            b_val, b_idx, b_valid = yield self.data.halo_vec_in[i].get()
            b_ijk = yield self.data.halo_idx_in[i].get()
            logger.trace(f"(Cycle {self.env.now}) HEU ({self.position}, {i}): b{b_ijk}={b_val} with index={b_idx}, b_valid={bin(b_valid)} is ready")

            out = 0; agg_out = 0

            out_mask = [OUT_I_M, OUT_J_M, OUT_I_INV_M, OUT_J_INV_M]
            agg_mask = [AGG_I_M, AGG_J_M, AGG_I_INV_M, AGG_J_INV_M]

            if self.compute_type == "sptrsv":
                if self.stencil_type == 0: # Star
                    out = yield self.boundary_ports[i].out.get()
                    self.add_counter += 1
                    b_valid -= out_mask[self.position]

                elif self.stencil_type == 1: # Star
                    if i % 2 == 0:
                        out = yield self.boundary_ports[i // 2].out.get()
                        b_valid -= out_mask[self.position]
                    else:
                        out = yield self.boundary_ports[i // 2].agg_out.get()
                        b_valid -= agg_mask[self.position]
                    self.add_counter += 1

                elif self.stencil_type == 2: # Diamond
                    if self.position == 0: # HEU_X
                        if i != self.num_PEs[1]:
                            out = (yield self.boundary_ports[i].out.get())
                            self.add_counter += 1
                            b_valid -= out_mask[self.position]
                        if i != 0:
                            agg_out = (yield self.boundary_ports[i - 1].agg_out.get())
                            self.add_counter += 1
                            b_valid -= agg_mask[self.position]

                    else: # HEU_Y
                        out = yield self.boundary_ports[i].out.get()
                        self.add_counter += 1
                        b_valid -= out_mask[self.position]

                elif self.stencil_type == 3: # Box
                    if self.position == 0: # HEU_X
                        if i != self.num_PEs[1]:
                            out = (yield self.boundary_ports[i].out.get())
                            self.add_counter += 1
                            b_valid -= out_mask[self.position]
                        if i != 0:
                            agg_out = (yield self.boundary_ports[i - 1].agg_out.get())
                            self.add_counter += 1
                            b_valid -= agg_mask[self.position]
                    else: # HEU_Y
                        if i % 2 == 0:
                            out = yield self.boundary_ports[i // 2].out.get()
                            b_valid -= out_mask[self.position]
                        else:
                            out = yield self.boundary_ports[i // 2].agg_out.get()
                            b_valid -= agg_mask[self.position]
                        self.add_counter += 1
            else:
                if self.stencil_type == 0: # Star7p
                    out = yield self.boundary_ports[i].out.get()
                    b_valid -= out_mask[self.position]
                    self.add_counter += 1

                elif self.stencil_type == 1: # Star13p
                    if i % 2 == 0:
                        out = yield self.boundary_ports[i // 2].out.get()
                        b_valid -= out_mask[self.position]
                    else:
                        out = yield self.boundary_ports[i // 2].agg_out.get()
                        b_valid -= agg_mask[self.position]
                    self.add_counter += 1

                elif self.stencil_type == 2: # Diamond
                    if self.position == 0: # HEU_X
                        if i != 0:
                            out = (yield self.boundary_ports[i - 1].out.get())
                            b_valid -= out_mask[self.position]
                            self.add_counter += 1
                        if i != self.num_PEs[1]:
                            agg_out = (yield self.boundary_ports[i].agg_out.get())
                            b_valid -= agg_mask[self.position]
                            self.add_counter += 1

                    elif self.position == 2: # HEU_X_INV
                        if i != self.num_PEs[1]:
                            out = (yield self.boundary_ports[i].out.get())
                            b_valid -= out_mask[self.position]
                            self.add_counter += 1
                        if i != 0:
                            agg_out = (yield self.boundary_ports[i - 1].agg_out.get())
                            b_valid -= agg_mask[self.position]
                            self.add_counter += 1

                    else: # HEU_Y | HEU_Y_INV
                        out = yield self.boundary_ports[i].out.get()
                        self.add_counter += 1
                        b_valid -= out_mask[self.position]

                elif self.stencil_type == 3: # Box
                    if self.position == 0: # HEU_X
                        if i != 0:
                            out = (yield self.boundary_ports[i - 1].out.get())
                            b_valid -= out_mask[self.position]
                            self.add_counter += 1
                        if i != self.num_PEs[1]:
                            agg_out = (yield self.boundary_ports[i].agg_out.get())
                            b_valid -= agg_mask[self.position]
                            self.add_counter += 1

                    elif self.position == 2: # HEU_X_INV
                        if i != self.num_PEs[1]:
                            out = (yield self.boundary_ports[i].out.get())
                            b_valid -= out_mask[self.position]
                            self.add_counter += 1
                        if i != 0:
                            agg_out = (yield self.boundary_ports[i - 1].agg_out.get())
                            b_valid -= agg_mask[self.position]
                            self.add_counter += 1

                    elif self.position == 1: # HEU_Y
                        if i != self.num_PEs[0]:
                            out = (yield self.boundary_ports[i].out.get())
                            b_valid -= out_mask[self.position]
                            self.add_counter += 1
                        if i != 0:
                            agg_out = (yield self.boundary_ports[i - 1].agg_out.get())
                            b_valid -= agg_mask[self.position]
                            self.add_counter += 1

                    else: # HEU_Y_INV
                        if i != 0:
                            out = (yield self.boundary_ports[i - 1].out.get())
                            b_valid -= out_mask[self.position]
                            self.add_counter += 1
                        if i != self.num_PEs[0]:
                            agg_out = (yield self.boundary_ports[i].agg_out.get())
                            b_valid -= agg_mask[self.position]
                            self.add_counter += 1

            if self.compute_type == "sptrsv":
                new_b = b_val - (out + agg_out)
            else:
                new_b = b_val + (out + agg_out)

            # For invalid values where index[0] < 0, agg & in port values need to be fetched first to avoid blocking, and the output can be discarded directly.
            if b_idx[0] < 0:
                logger.trace(f"(Cycle {self.env.now}) HEU ({self.position}, {i}) ignore the invalid b_idx={b_idx}")
                continue

            logger.trace(f"(Cycle {self.env.now}) HEU ({self.position}, {i}): get data ready takes {self.env.now - tick} cycles")
            logger.info(f"(Cycle {self.env.now}) HEU ({self.position}, {i}): update b{b_ijk}={new_b} with b={b_val}, out={out}, agg_out={agg_out}, b_idx={b_idx}, b_ijk={b_ijk}")

            # process time is determined by the number of lanes
            times = math.ceil(self.halo_points / self.num_lanes)
            delay = self.cfg["Delay"]["Add"] * times
            yield self.env.timeout(delay)
            yield self.env.process(self.bufs.halo_vec_out.access(1))
            yield self.data.halo_vec_out[i].put((new_b, b_idx, b_valid))
            yield self.data.halo_idx_out[i].put(b_ijk)
            logger.trace(f"(Cycle {self.env.now}) HEU ({self.position}, {i}): update b{b_ijk}={new_b} one iteration takes {self.env.now - tick} cycles")
