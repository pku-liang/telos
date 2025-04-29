import simpy
from math import ceil
from loguru import logger

class SRAM:
    def __init__(self, env, name, bw):
        self.env = env
        self.name = name
        self.SRAM_BW = bw
        self.container = simpy.Container(env, init=bw, capacity=bw)
        self.actions = env.process(self.run())
        self.counter = 0

    def run(self):
        while True:
            amount = self.SRAM_BW - self.container.level
            if amount > 0:
                yield self.container.put(amount)
            yield self.env.timeout(1)

    def access(self, size):
        self.counter += size
        yield self.container.get(size)

class DRAM:
    def __init__(self, env, name, bw, data):
        self.env = env
        self.name = name
        self.DRAM_BW = bw
        self.data = data
        self.read_counter = 0
        self.write_counter = 0
        self.proc_read = env.process(self.run_read())
        self.proc_write = env.process(self.run_write())
        self.actions = [self.proc_read, self.proc_write]

    def run_read(self):
        for i in range(self.data.iters):
            tick = self.env.now
            size = self.data.get_read_size()
            self.read_counter += size
            delay = ceil(size / self.DRAM_BW)
            yield self.env.timeout(delay) & self.env.process(self.data.put_next())
            logger.trace(f"(Cycle {self.env.now}) DRAM: read {size} {self.name} values (iter {i}) takes {delay} cycles")
            logger.trace(f"(Cycle {self.env.now}) DRAM: put the read value to the port {size} {self.name} (iter {i}) takes {self.env.now - tick} cycles")
        logger.info(f"(Cycle {self.env.now}) DRAM: read {self.name} finished")

    def run_write(self):
        if self.data.readonly:
            return
        for i in range(self.data.iters):
            size = self.data.get_write_size()
            self.write_counter += size
            delay = ceil(size / self.DRAM_BW)
            yield self.env.timeout(delay) & self.env.process(self.data.get_previous())
            logger.trace(f"(Cycle {self.env.now}) DRAM: write {size} {self.name} values (iter {i}) needs {delay} cycles")
        logger.info(f"(Cycle {self.env.now}) DRAM: write {self.name} finished")

class Buffers:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.domain_vec_in = SRAM(env, "domain_vec_in", cfg["SRAM"]["DomainVec_BW"])
        self.domain_vec_out = SRAM(env, "domain_vec_out", cfg["SRAM"]["DomainVec_BW"])
        self.domain_diag_mtx = SRAM(env, "domain_diag_mtx", cfg["SRAM"]["DomainVec_BW"])
        self.domain_mtx = SRAM(env, "domain_mtx", cfg["SRAM"]["DomainMtx_BW"])
        self.halo_vec_in = SRAM(env, "halo_vec_in", cfg["SRAM"]["HaloVec_BW"])
        self.halo_vec_out = SRAM(env, "halo_vec_out", cfg["SRAM"]["HaloVec_BW"])
