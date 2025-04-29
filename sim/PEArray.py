import simpy
from loguru import logger
from .stencil import get_id2stage, get_stencil_points, get_stage_lanes, get_num_stencil_points, get_affine_stencil_points

class PEPorts:
    def __init__(self, env, compute_type):
        # output ports
        self.out_i = simpy.Store(env, capacity=1)
        self.out_j = simpy.Store(env, capacity=1)
        self.agg_out_i = simpy.Store(env, capacity=1)
        self.agg_out_j = simpy.Store(env, capacity=1)
        if compute_type == "spmv":
            self.out_i_inv = simpy.Store(env, capacity=1)
            self.out_j_inv = simpy.Store(env, capacity=1)
            self.agg_out_i_inv = simpy.Store(env, capacity=1)
            self.agg_out_j_inv = simpy.Store(env, capacity=1)
        # input ports
        self.in_i = simpy.Store(env, capacity=1)
        self.in_j = simpy.Store(env, capacity=1)
        self.agg_in_i = simpy.Store(env, capacity=1)
        self.agg_in_j = simpy.Store(env, capacity=1)
        if compute_type == "spmv":
            self.in_i_inv = simpy.Store(env, capacity=1)
            self.in_j_inv = simpy.Store(env, capacity=1)
            self.agg_in_i_inv = simpy.Store(env, capacity=1)
            self.agg_in_j_inv = simpy.Store(env, capacity=1)

        # debug ports
        self.out_i_ijk = simpy.Store(env, capacity=1)
        self.out_j_ijk = simpy.Store(env, capacity=1)
        self.in_i_ijk = simpy.Store(env, capacity=1)
        self.in_j_ijk = simpy.Store(env, capacity=1)

class BoundaryPorts:
    def __init__(self, env):
        self.out = simpy.Store(env, capacity=1)
        self.agg_out = simpy.Store(env, capacity=1)

class PE:
    def __init__(self, env, cfg, bufs, data, spmat_data, ports, i, j, compute_type):
        self.env = env
        self.cfg = cfg
        self.bufs = bufs
        self.data = data
        self.spmat_data = spmat_data
        self.ports = ports
        self.compute_type = compute_type
        self.stencil_type = cfg["StencilType"]
        self.dims = cfg["NumDims"]
        self.num_pes = cfg["Arch"]["NumPEs"]
        # PE index
        self.i = i
        self.j = j
        # computes
        self.mul_counter = 0
        self.div_counter = 0
        self.add_counter = 0
        # important, to prevent deadlock
        self.vec_fifo_depth = 4

        # internal control
        self.R_k = simpy.Store(env, capacity=1)
        self.ijk_index = simpy.Store(env, capacity=1)
        self.index = simpy.Store(env, capacity=1)

        if self.compute_type == "sptrsv":
            self.new_x = simpy.Store(env, capacity=1)
            self.actions = [env.process(self.ScalarUnit()), env.process(self.VectorUnit())]
        else:
            self.tmp_port = simpy.Store(env, capacity=1)
            self.actions = [env.process(self.ScalarUnit_SpMV()), env.process(self.VectorUnit_SpMV())]

        self.id2stage = get_id2stage(self.dims, self.stencil_type, self.compute_type)
        self.stage_lanes = get_stage_lanes(self.dims, self.stencil_type, self.compute_type)
        self.num_stencil_points = get_num_stencil_points(self.dims, self.stencil_type, self.compute_type)
        self.vec_results = [simpy.Store(env, capacity=self.vec_fifo_depth) for _ in range(self.num_stencil_points)]

        # init shift_x
        if self.dims == 3:
            self.shift_x = [(0, -(i + 1), True, (self.i, self.j, -(i + 1))) for i in range(len(self.stage_lanes))]
        else:
            self.shift_x = [(0, -(i + 1), True, (-1, -1)) for i in range(len(self.stage_lanes))]

        self.term_id_dict = {f"term[{i}]": i for i in range(self.num_stencil_points)}

        if self.compute_type == "sptrsv":
            self.index_offset = get_affine_stencil_points(self.dims, self.stencil_type)
            self.z_offset = [-i[2] for i in self.index_offset] if self.dims == 3 else [0 for _ in self.index_offset]
        else:
            self.index_offset = get_stencil_points(self.dims, self.stencil_type)
            self.z_offset = [i[2] for i in self.index_offset] if self.dims == 3 else [0 for _ in self.index_offset]

        self.Aggregator_init_processes()

    def ScalarUnit(self):
        while True:
            # Get data from the buffer
            tick = self.env.now
            yield self.env.process(self.bufs.domain_vec_in.access(1))
            yield self.env.process(self.bufs.domain_diag_mtx.access(1))

            aii_start = self.env.now
            aii = yield self.data.domain_diag_mtx[self.i][self.j].get()
            ijk_index = yield self.data.domain_index[self.i][self.j].get()
            aii_end = self.env.now
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get aii, ijk_index ready takes {self.env.now - tick} cycles ijk_index={ijk_index}")

            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: waiting for b")
            b_start = self.env.now
            b = yield self.data.domain_vec_in[self.i][self.j].get()
            b_end = self.env.now
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get b ready takes {self.env.now - tick} cycles")

            in_start = self.env.now
            in_j, target_j_ijk = (yield self.ports.in_j.get()) if self.j != 0 else (0, 0)
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get in_j ready takes {self.env.now - tick} cycles")

            in_i, target_i_ijk = (yield self.ports.in_i.get()) if self.i != 0 else (0, 0)
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get in_i ready takes {self.env.now - tick} cycles")

            R_k, target_k_ijk = (yield self.R_k.get()) if ijk_index[2] != 0 else (0, 0)
            in_end = self.env.now

            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get all data ready takes {self.env.now - tick} cycles, aii ready takes {aii_end - aii_start} cycles, b ready takes {b_end - b_start} cycles, in ready takes {in_end - in_start} cycles")

            sum = in_i + in_j + R_k
            x = (b - sum) / aii

            logger.info(f"""(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: compute variable x{ijk_index}={x} with
                        aii={aii}, b={b}, in_i={in_i}, target_ijk={target_i_ijk}, in_j={in_j}, target_ijk={target_j_ijk}, R_k={R_k}, target_ijk={target_k_ijk}""")

            self.add_counter += self.dims
            self.div_counter += 1
            delay = self.cfg["Delay"]["Add"] + self.cfg["Delay"]["Div"]
            yield self.env.timeout(delay)
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: waiting new_x...")
            yield self.new_x.put((x, ijk_index))

            yield self.env.process(self.bufs.domain_vec_out.access(1))
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: waiting domain_vec...")
            yield self.data.domain_vec_out[self.i][self.j].put((x, ijk_index))
            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: one iteration takes {self.env.now - tick} cycles")

    def ScalarUnit_SpMV(self):
        while True:
            # Get data from the buffer
            tick = self.env.now

            ijk_start = self.env.now
            ijk_index, index = yield self.data.domain_index[self.i][self.j].get()
            ijk_end = self.env.now
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get ijk_index ready takes {self.env.now - tick} cycles ijk_index={ijk_index}")

            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: waiting for b")
            b_start = self.env.now
            yield self.env.process(self.bufs.domain_vec_in.access(1))
            b = yield self.data.domain_b_in[self.i][self.j].get()
            b_end = self.env.now
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get b ready takes {self.env.now - tick} cycles")

            in_start = self.env.now
            in_i, target_i_ijk = (yield self.ports.in_i.get()) if self.i != 0 else (0, 0)
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get in_i ready takes {self.env.now - tick} cycles")

            in_j, target_j_ijk = (yield self.ports.in_j.get()) if self.j != 0 else (0, 0)
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get in_j ready takes {self.env.now - tick} cycles")

            in_i_inv, target_i_inv_ijk = (yield self.ports.in_i_inv.get()) if self.i != self.num_pes[0] - 1 else (0, 0)
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get in_i_inv ready takes {self.env.now - tick} cycles")

            in_j_inv, target_j_inv_ijk = (yield self.ports.in_j_inv.get()) if self.j != self.num_pes[1] - 1 else (0, 0)
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get in_j_inv ready takes {self.env.now - tick} cycles")

            R_k, target_k_ijk = (yield self.R_k.get())
            in_end = self.env.now

            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get all data ready takes {self.env.now - tick} cycles, \
                        b ready takes {b_end - b_start} cycles, \
                        all inputs ready takes {in_end - in_start} cycles")

            b_new = b + in_i + in_j + R_k + in_i_inv + in_j_inv

            logger.info(f"""(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: compute variable b_new{ijk_index}={b_new} with
                        b={b}, in_i={in_i}, target_ijk={target_i_ijk}, in_j={in_j}, target_ijk={target_j_ijk}, R_k={R_k}, target_ijk={target_k_ijk}
                        in_i_inv={in_i_inv}, target_ijk={target_i_inv_ijk}, in_j_inv={in_j_inv}, target_ijk={target_j_inv_ijk}
                        """)

            self.add_counter += 5
            delay = 2 * self.cfg["Delay"]["Add"]
            yield self.env.timeout(delay)
            yield self.data.domain_b_out[self.i][self.j].put((b_new, ijk_index, index))

            yield self.env.process(self.bufs.domain_vec_out.access(1))
            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: one iteration takes {self.env.now - tick} cycles")

    def VectorUnit(self):
        # schedule multiply index
        # Sorted in descending order of critical path length
        sche_seq_3d = [
            [0, 1, 2],
            [0, 3, 1, 4, 2, 5],
            # [0, 3, 4, 5, 1, 2],
            [0, 4, 5, 2, 3, 1],
            [0, 10, 11, 12, 6, 8, 9, 1, 3, 5, 7, 4, 2]
        ]
        sche_seq_2d = [
            [0, 1],
            [2, 3, 0, 1],
            [2, 1, 0],
            [3, 2, 0, 1]
        ]
        sche_seq = sche_seq_3d if self.dims == 3 else sche_seq_2d
        dim0_index = -1
        while True:
            tick = self.env.now
            yield self.env.process(self.bufs.domain_mtx.access(self.num_stencil_points))
            vec_A, valid = yield self.spmat_data.domain_mtx[self.i][self.j].get()

            if valid:
                logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: waiting for new_x")
                new_x, ijk_index = yield self.new_x.get()
                dim0_index += 1
            else:
                new_x, ijk_index = 0, (0, 0, 0)

            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: get data ready takes {self.env.now - tick} cycles")

            # shift x
            self.shift_x.insert(0, (new_x, dim0_index, valid, ijk_index))
            self.shift_x.pop()

            vec_x = []
            for shift_x, lane_n in zip(self.shift_x, self.stage_lanes):
                vec_x += [shift_x[0]] * lane_n

            # yield self.ijk_index.put(ijk_index)
            # schedule
            lanes = self.cfg["Arch"]["VecLanes"]
            sche_cycles = (self.num_stencil_points + lanes - 1) // lanes
            mul_points = 0
            for i in range(sche_cycles):
                # pipeline
                yield self.env.timeout(self.cfg["Delay"]["Mul"] if i == 0 else 1)
                sche_ids = []
                for j in range(lanes):
                    id = i * lanes + j
                    if id >= self.num_stencil_points:
                        break
                    sche_id = sche_seq[self.stencil_type][id]
                    stage_id = self.id2stage[sche_id]

                    z = self.data.size[2]
                    z_offset = self.z_offset[sche_id] if self.dims == 3 else 0

                    cur_z = self.shift_x[stage_id][1]
                    x_valid = self.shift_x[stage_id][2]
                    cur_ijk_index = self.shift_x[stage_id][3]
                    z_target = z_offset + cur_z

                    # Prerequisite: The target z-coordinate range does not exceed the overall boundary
                    if x_valid and z_target >= 0 and z_target < self.data.dim0_extent and \
                        (z_offset - stage_id <= 0 or z_target % z != 0):
                        # Special case for output to the R_k port (satisfying z_offset > stage_id). If z_target % z == 0,
                        # it indicates that the target and the current are not in the same tile, so do not output to avoid
                        # blocking the R_k port. This is because R_k does not fetch numbers from the port when a new tile starts.
                        # For other ports, if the target ijk exceeds the intermediate tile boundary, the corresponding A element
                        # will be 0, and the output result will be 0. This will not affect the computed value, so no check is needed.
                        target_index = tuple(a - b for a, b in zip(cur_ijk_index, self.index_offset[sche_id]))
                        logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: waiting for port[{sche_id}] to be empty")
                        yield self.vec_results[sche_id].put((vec_x[sche_id] * vec_A[sche_id], target_index))
                        sche_ids.append(sche_id)
                        mul_points += 1
                        logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: pass terms[{sche_id}], cur_index={cur_ijk_index}, target_index={target_index}")
                    else:
                        sche_ids.append(-1)

            self.mul_counter += mul_points
            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: one iteration takes {self.env.now - tick} cycles")

    def VectorUnit_SpMV(self):
        # schedule multiply index
        # Sorted in descending order of critical path length
        sche_seq_3d = [
            [0, 3, 6, 1, 2, 4, 5],
            [0, 1, 11, 12, 6, 2, 3, 4, 5, 7, 8, 9, 10],
            [5, 8, 7, 11, 3, 0, 1, 4, 9, 12, 2, 6, 10],
            [0, 2, 6, 8, 9, 11, 15, 17, 18,
             20, 24, 26, 1, 3, 5, 7, 10, 12,
             14, 16, 19, 21, 23, 25, 4, 13, 22],
        ]
        sche_seq_2d = [
            [0, 1, 2, 3, 4],
            [0, 2, 6, 8, 1, 3, 5, 7, 4],
            [1, 5, 0, 2, 4, 6, 3],
            [0, 2, 6, 8, 1, 3, 5, 7, 4],
        ]
        sche_seq = sche_seq_3d if self.dims == 3 else sche_seq_2d
        dim0_index = -1
        while True:
            tick = self.env.now
            yield self.env.process(self.bufs.domain_mtx.access(self.num_stencil_points))
            vec_A, valid = yield self.spmat_data.domain_mtx[self.i][self.j].get()

            if valid:
                logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: waiting for new_x")
                new_x, ijk_index = yield self.data.domain_x_in[self.i][self.j].get()
                dim0_index += 1
            else:
                new_x, ijk_index = 0, (0, 0, 0)

            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: get data ready takes {self.env.now - tick} cycles")

            # shift x
            self.shift_x.insert(0, (new_x, dim0_index, valid, ijk_index))
            self.shift_x.pop()

            # schedule
            lanes = self.cfg["Arch"]["VecLanes"]
            sche_cycles = (self.num_stencil_points + lanes - 1) // lanes
            mult_n = 0
            for i in range(sche_cycles):
                # pipeline
                yield self.env.timeout(self.cfg["Delay"]["Mul"] if i == 0 else 1)
                sche_ids = []
                for j in range(lanes):
                    id = i * lanes + j
                    if id >= self.num_stencil_points:
                        break
                    sche_id = sche_seq[self.stencil_type][id]
                    stage_id = self.id2stage[sche_id]
                    z_offset = self.z_offset[sche_id] if self.dims == 3 else 0

                    cur_z = self.shift_x[stage_id][1]
                    # x_valid = self.shift_x[stage_id][2]
                    cur_ijk_index = self.shift_x[stage_id][3]
                    z_target = z_offset + cur_z

                    # Prerequisite: The target z-coordinate range does not exceed the overall boundary
                    if z_target >= 0 and z_target < self.data.dim0_extent:
                        target_index = tuple(a + b for a, b in zip(cur_ijk_index, self.index_offset[sche_id]))
                        logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: waiting for port[{sche_id}] to be empty")
                        mult_val = self.shift_x[stage_id][0] * vec_A[sche_id]
                        yield self.vec_results[sche_id].put((mult_val, target_index))
                        sche_ids.append(sche_id)
                        mult_n += 1
                        logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: pass terms[{sche_id}], cur_index={cur_ijk_index}, target_index={target_index}, val={mult_val}, A={vec_A[sche_id]}, x={self.shift_x[stage_id][0]}")
                    else:
                        sche_ids.append(-1)

            self.mul_counter += mult_n
            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: one iteration takes {self.env.now - tick} cycles")

    def Aggregator_init_processes(self):
        # bind name to port for debugging
        agg_in_i = (self.ports.agg_in_i, "agg_in_i")
        agg_in_j = (self.ports.agg_in_j, "agg_in_j")
        agg_out_i = (self.ports.agg_out_i, "agg_out_i")
        agg_out_j = (self.ports.agg_out_j, "agg_out_j")
        out_i = (self.ports.out_i, "out_i")
        out_j = (self.ports.out_j, "out_j")
        rk = (self.R_k, "R_k")
        vec_results = [(self.vec_results[i], f"term[{i}]") for i in range(self.num_stencil_points)]

        if self.compute_type == "spmv":
            agg_in_i_inv = (self.ports.agg_in_i_inv, "agg_in_i_inv")
            agg_in_j_inv = (self.ports.agg_in_j_inv, "agg_in_j_inv")
            agg_out_i_inv = (self.ports.agg_out_i_inv, "agg_out_i_inv")
            agg_out_j_inv = (self.ports.agg_out_j_inv, "agg_out_j_inv")
            out_i_inv = (self.ports.out_i_inv, "out_i_inv")
            out_j_inv = (self.ports.out_j_inv, "out_j_inv")
            tmp_port = (self.tmp_port, "tmp_port")

            if self.dims == 3:
                if self.stencil_type == 0:
                    self.actions += [
                        self.env.process(self.buf(vec_results[4], out_i)),
                        self.env.process(self.buf(vec_results[5], out_j)),
                        self.env.process(self.buf(vec_results[2], out_i_inv)),
                        self.env.process(self.buf(vec_results[1], out_j_inv)),
                        self.env.process(self.adder([vec_results[0], vec_results[3], vec_results[6]], rk))
                    ]
                elif self.stencil_type == 1:
                    self.actions += [
                        self.env.process(self.adder([agg_in_i, vec_results[7]], out_i)),
                        self.env.process(self.adder([agg_in_j, vec_results[9]], out_j)),
                        self.env.process(self.adder([agg_in_i_inv, vec_results[2]], out_i_inv)),
                        self.env.process(self.adder([agg_in_j_inv, vec_results[3]], out_j_inv)),
                        self.env.process(self.adder([vec_results[0], vec_results[1], vec_results[11], vec_results[12]], tmp_port)),
                        self.env.process(self.adder([tmp_port, vec_results[6]], rk)),
                        self.env.process(self.buf(vec_results[8], agg_out_i)),
                        self.env.process(self.buf(vec_results[10], agg_out_j)),
                        self.env.process(self.buf(vec_results[4], agg_out_i_inv)),
                        self.env.process(self.buf(vec_results[5], agg_out_j_inv))
                    ]
                elif self.stencil_type == 2:
                    self.actions += [
                        self.env.process(self.adder([vec_results[7], vec_results[11]], out_i)),
                        self.env.process(self.adder([agg_in_i_inv, vec_results[9], vec_results[12]], out_j)),
                        self.env.process(self.adder([vec_results[0], vec_results[3]], out_i_inv)),
                        self.env.process(self.adder([agg_in_i, vec_results[1], vec_results[4]], out_j_inv)),
                        self.env.process(self.adder([vec_results[2], vec_results[6], vec_results[10]], rk)),
                        self.env.process(self.buf(vec_results[8], agg_out_i)),
                        self.env.process(self.buf(vec_results[5], agg_out_i_inv)),
                    ]
                elif self.stencil_type == 3:
                    self.actions += [
                        self.env.process(self.adder([agg_in_j, vec_results[5], vec_results[14], vec_results[23]], out_i)),
                        self.env.process(self.adder([agg_in_i_inv, vec_results[7], vec_results[16], vec_results[25]], out_j)),
                        self.env.process(self.adder([agg_in_j_inv, vec_results[3], vec_results[12], vec_results[21]], out_i_inv)),
                        self.env.process(self.adder([agg_in_i, vec_results[1], vec_results[10], vec_results[19]], out_j_inv)),
                        self.env.process(self.adder([vec_results[4], vec_results[13], vec_results[22]], rk)),
                        self.env.process(self.adder([vec_results[2], vec_results[11], vec_results[20]], agg_out_i)),
                        self.env.process(self.adder([vec_results[8], vec_results[17], vec_results[26]], agg_out_j)),
                        self.env.process(self.adder([vec_results[6], vec_results[15], vec_results[24]], agg_out_i_inv)),
                        self.env.process(self.adder([vec_results[0], vec_results[9], vec_results[18]], agg_out_j_inv)),
                    ]
            else:
                if self.stencil_type == 0: # star5p
                    self.actions += [
                        self.env.process(self.buf(vec_results[3], out_i)),
                        self.env.process(self.buf(vec_results[4], out_j)),
                        self.env.process(self.buf(vec_results[1], out_i_inv)),
                        self.env.process(self.buf(vec_results[0], out_j_inv)),
                        self.env.process(self.buf(vec_results[2], rk)),
                    ]
                elif self.stencil_type == 1: # star7p
                    self.actions += [
                        self.env.process(self.adder([agg_in_i, vec_results[5]], out_i)),
                        self.env.process(self.adder([agg_in_j, vec_results[7]], out_j)),
                        self.env.process(self.adder([agg_in_i_inv, vec_results[3]], out_i_inv)),
                        self.env.process(self.adder([agg_in_j_inv, vec_results[1]], out_j_inv)),
                        self.env.process(self.buf(vec_results[4], rk)),
                        self.env.process(self.buf(vec_results[6], agg_out_i)),
                        self.env.process(self.buf(vec_results[8], agg_out_j)),
                        self.env.process(self.buf(vec_results[2], agg_out_i_inv)),
                        self.env.process(self.buf(vec_results[0], agg_out_j_inv)),
                    ]
                elif self.stencil_type == 2: # diamond7p
                    self.actions += [
                        self.env.process(self.buf(vec_results[4], out_i)),
                        self.env.process(self.adder([agg_in_i_inv, vec_results[6]], out_j)),
                        self.env.process(self.buf(vec_results[2], out_i_inv)),
                        self.env.process(self.adder([agg_in_i, vec_results[0]], out_j_inv)),
                        self.env.process(self.buf(vec_results[3], rk)),
                        self.env.process(self.buf(vec_results[1], agg_out_i)),
                        self.env.process(self.buf(vec_results[5], agg_out_i_inv)),
                    ]
                elif self.stencil_type == 3: # box9p
                    self.actions += [
                        self.env.process(self.adder([agg_in_j, vec_results[5]], out_i)),
                        self.env.process(self.adder([agg_in_i_inv, vec_results[7]], out_j)),
                        self.env.process(self.adder([agg_in_j_inv, vec_results[3]], out_i_inv)),
                        self.env.process(self.adder([agg_in_i, vec_results[1]], out_j_inv)),
                        self.env.process(self.buf(vec_results[4], rk)),
                        self.env.process(self.buf(vec_results[2], agg_out_i)),
                        self.env.process(self.buf(vec_results[8], agg_out_j)),
                        self.env.process(self.buf(vec_results[6], agg_out_i_inv)),
                        self.env.process(self.buf(vec_results[0], agg_out_j_inv)),
                    ]
        else:
            # generate adder & buf processes
            if self.dims == 3:
                if self.stencil_type == 0:
                    self.actions += [
                        self.env.process(self.buf(vec_results[2], out_i)),
                        self.env.process(self.buf(vec_results[1], out_j)),
                        self.env.process(self.buf(vec_results[0], rk))
                    ]
                elif self.stencil_type == 1:
                    self.actions += [
                        self.env.process(self.adder([agg_in_i, vec_results[1]], out_i)),
                        self.env.process(self.adder([agg_in_j, vec_results[2]], out_j)),
                        self.env.process(self.adder([vec_results[0], vec_results[3]], rk)),
                        self.env.process(self.buf(vec_results[4], agg_out_i)),
                        self.env.process(self.buf(vec_results[5], agg_out_j))
                    ]
                elif self.stencil_type == 2:
                    self.actions += [
                        self.env.process(self.buf(vec_results[1], out_i)),
                        self.env.process(self.adder([agg_in_i, vec_results[2], vec_results[3]], out_j)),
                        self.env.process(self.buf(vec_results[0], rk)),
                        self.env.process(self.adder([vec_results[4], vec_results[5]], agg_out_i))
                    ]
                elif self.stencil_type == 3:
                    self.actions += [
                        self.env.process(self.adder([vec_results[1], vec_results[4], vec_results[7]], out_i)),
                        self.env.process(self.adder([agg_in_i, vec_results[2], vec_results[3], vec_results[5]], out_j)),
                        self.env.process(self.adder([agg_in_j, vec_results[6], vec_results[8], vec_results[9]], agg_out_i)),
                        self.env.process(self.adder([vec_results[10], vec_results[11], vec_results[12]], agg_out_j)),
                        self.env.process(self.buf(vec_results[0], rk))
                    ]
            else:
                if self.stencil_type == 0: # star5p
                    self.actions += [
                        self.env.process(self.buf(vec_results[0], out_i)),
                        self.env.process(self.buf(vec_results[1], out_j))
                    ]
                # [(1, 0), (0, 1), (2, 0), (0, 2)]
                elif self.stencil_type == 1: # star7p
                    self.actions += [
                        self.env.process(self.adder([agg_in_i, vec_results[0]], out_i)),
                        self.env.process(self.adder([agg_in_j, vec_results[1]], out_j)),
                        self.env.process(self.buf(vec_results[2], agg_out_i)),
                        self.env.process(self.buf(vec_results[3], agg_out_j))
                    ]
                # [(1, 0), (0, 1), (1, 1)]
                elif self.stencil_type == 2: # diamond7p
                    self.actions += [
                        self.env.process(self.buf(vec_results[0], out_i)),
                        self.env.process(self.adder([agg_in_i, vec_results[1]], out_j)),
                        self.env.process(self.buf(vec_results[2], agg_out_i))
                    ]
                # [(1, 0), (0, 1), (1, 1), (1, 2)]
                elif self.stencil_type == 3: # box9p
                    self.actions += [
                        self.env.process(self.buf(vec_results[0], out_i)),
                        self.env.process(self.adder([agg_in_i, vec_results[1]], out_j)),
                        self.env.process(self.adder([agg_in_j, vec_results[2]], agg_out_i)),
                        self.env.process(self.buf(vec_results[3], agg_out_j))
                    ]

    def adder(self, inputs, output):
        # Maximum of 4-input adder
        assert(len(inputs) <= 4)
        output_port, output_name = output
        while True:
            sum = 0
            input_names = []
            target_indexes = []
            for in_port, in_name in inputs:
                if (in_name == "agg_in_i" and self.i == 0) or \
                    (in_name == "agg_in_j" and self.j == 0) or \
                    (in_name == "agg_in_i_inv" and self.i == self.num_pes[0] - 1) or \
                    (in_name == "agg_in_j_inv" and self.j == self.num_pes[1] - 1):
                    continue
                if output_name == "out_j":
                    logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: waiting for in port={in_name} to be available")
                input_data, target_index = yield in_port.get()
                if output_name == "out_j":
                    logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: get input from in port={in_name}")

                target_indexes.append(target_index)
                sum += input_data
                input_names.append(in_name)

            # print(target_indexes)
            # assert(len(set(target_indexes)) == 1)
            input_str = "+".join(input_names)
            yield self.env.timeout(self.cfg['Delay']['Add'])
            logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: waiting for output port={output_name} to be available")
            yield output_port.put((sum, target_indexes[0]))

            self.add_counter += len(inputs) - 1
            logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: pass {input_str}={sum} through {output_name}, target_index={target_indexes[0]}")

    def buf(self, input, output):
        input_port, input_name = input
        output_port, output_name = output
        while True:
            input_data, target_index = yield input_port.get()
            logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: waiting for output port={output_name} to be available")
            yield output_port.put((input_data, target_index))
            logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: pass {input_name}={input_data} through {output_name} target_index={target_index}")


class PEArray:
    def __init__(self, env, cfg, bufs, data, spmat_data, boundaries):
        self.env = env
        self.cfg = cfg
        self.data = data
        self.spmat_data = spmat_data
        self.boundaries = boundaries
        self.num_PEs = cfg["Arch"]["NumPEs"]
        self.compute_type = cfg["ComputeType"]

        self.ports = [[PEPorts(env, self.compute_type) for _ in range(self.num_PEs[1])] for _ in range(self.num_PEs[0])]
        self.PEs = [[PE(env, cfg, bufs, data, spmat_data, self.ports[i][j], i, j, self.compute_type) for j in range(self.num_PEs[1])] for i in range(self.num_PEs[0])]

        self.actions = []
        for i in range(self.num_PEs[0]):
            for j in range(self.num_PEs[1]):
                self.actions += [env.process(self.trans_out(i, j)), env.process(self.trans_aggr(i, j))]
                # self.actions += [env.process(self.trans_out_i(i, j)), env.process(self.trans_out_j(i, j)),
                #                   env.process(self.trans_aggr_i(i, j)), env.process(self.trans_aggr_j(i, j))]
                if self.compute_type == "spmv":
                    self.actions += [env.process(self.trans_out_inv(i, j)), env.process(self.trans_aggr_inv(i, j)),]


    def stat(self):
        mul_counter = sum([pe.mul_counter for row in self.PEs for pe in row])
        div_counter = sum([pe.div_counter for row in self.PEs for pe in row])
        add_counter = sum([pe.add_counter for row in self.PEs for pe in row])
        return mul_counter, div_counter, add_counter

    def trans_out_inv(self, i, j):
        while True:
            # out_i_inv
            out_i_inv, target_index = yield self.ports[i][j].out_i_inv.get()
            if i != 0:
                yield self.ports[i - 1][j].in_i_inv.put((out_i_inv, target_index))
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i - 1}, {j}) through (out_i_inv, in_i_inv), target_ijk={target_index}")
            else:
                yield self.boundaries[2][j].out.put(out_i_inv)
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (2, {j}) out through out_i_inv, target_ijk={target_index}")

            # out_j_inv
            out_j_inv, target_index = yield self.ports[i][j].out_j_inv.get()
            if j != 0:
                yield self.ports[i][j - 1].in_j_inv.put((out_j_inv, target_index))
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i}, {j - 1}) through (out_j_inv, in_j_inv), target_ijk={target_index}")
            else:
                yield self.boundaries[3][i].out.put(out_j_inv)
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (3, {i}) out through out_j_inv, target_ijk={target_index}")

    def trans_out(self, i, j):
        while True:
            # out_i
            out_i, target_index = yield self.ports[i][j].out_i.get()
            if i != self.num_PEs[0] - 1:
                yield self.ports[i + 1][j].in_i.put((out_i, target_index))
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i + 1}, {j}) through (out_i, in_i), target_ijk={target_index}")
            else:
                yield self.boundaries[0][j].out.put(out_i)
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (0, {j}) out through out_i, target_ijk={target_index}")

            # out_j
            out_j, target_index = yield self.ports[i][j].out_j.get()
            if j != self.num_PEs[1] - 1:
                yield self.ports[i][j + 1].in_j.put((out_j, target_index))
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i}, {j + 1}) through (out_j, in_j), target_ijk={target_index}")
            else:
                yield self.boundaries[1][i].out.put(out_j)
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (1, {i}) out through out_j, target_ijk={target_index}")



    def trans_aggr_inv(self, i, j):
        stencil_type = self.cfg["StencilType"]
        use_agg_i = False if stencil_type == 0 else True    # Star13p & Diamond & Box
        use_agg_i_inv = use_agg_i if self.compute_type == "spmv" else False
        use_agg_j = True if stencil_type == 1 or stencil_type == 3 else False    # Star13p & Box
        use_agg_j_inv = use_agg_j if self.compute_type == "spmv" else False

        if not (use_agg_i_inv or use_agg_j_inv):
            return
        while True:
            if use_agg_i_inv:
                agg_out_i_inv, target_index = yield self.ports[i][j].agg_out_i_inv.get()
                if i != 0:
                    yield self.ports[i - 1][j].agg_in_i_inv.put((agg_out_i_inv, target_index))
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i - 1}, {j}) through (agg_out_i_inv, agg_in_i_inv), target_ijk={target_index}")
                else:
                    yield self.boundaries[2][j].agg_out.put(agg_out_i_inv)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (2, {j}) agg_out through agg_out_i_inv, target_ijk={target_index}")

            if use_agg_j_inv:
                agg_out_j_inv, target_index = yield self.ports[i][j].agg_out_j_inv.get()
                if j != 0:
                    yield self.ports[i][j - 1].agg_in_j_inv.put((agg_out_j_inv, target_index))
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i}, {j - 1}) through (agg_out_j_inv, agg_in_j_inv), target_ijk={target_index}")
                else:
                    yield self.boundaries[3][i].agg_out.put(agg_out_j_inv)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (3, {i}) agg_out through agg_out_j_inv, target_ijk={target_index}")


    def trans_aggr(self, i, j):
        stencil_type = self.cfg["StencilType"]
        use_agg_i = False if stencil_type == 0 else True    # Star13p & Diamond & Box
        use_agg_j = True if stencil_type == 1 or stencil_type == 3 else False    # Star13p & Box
        if not (use_agg_i or use_agg_j):
            return

        while True:
            if use_agg_i:
                agg_out_i, target_index = yield self.ports[i][j].agg_out_i.get()
                if i != self.num_PEs[0] - 1:
                    yield self.ports[i + 1][j].agg_in_i.put((agg_out_i, target_index))
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i + 1}, {j}) through (agg_out_i, agg_in_i), target_ijk={target_index}")
                else:
                    yield self.boundaries[0][j].agg_out.put(agg_out_i)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (0, {j}) agg_out through agg_out_i, target_ijk={target_index}")

            if use_agg_j:
                agg_out_j, target_index = yield self.ports[i][j].agg_out_j.get()
                if j != self.num_PEs[1] - 1:
                    yield self.ports[i][j + 1].agg_in_j.put((agg_out_j, target_index))
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i}, {j + 1}) through (agg_out_j, agg_in_j), target_ijk={target_index}")
                else:
                    yield self.boundaries[1][i].agg_out.put(agg_out_j)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (1, {i}) agg_out through agg_out_j, target_ijk={target_index}")
