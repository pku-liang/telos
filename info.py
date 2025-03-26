class SummaryInfo:
    def __init__(self, cycles, wall_clock_time, pe_util, ideal_delay, ideal_cycles, energy):
        self.cycles = cycles
        self.wall_clock_time = wall_clock_time
        self.pe_util = pe_util
        self.ideal_delay = ideal_delay
        self.ideal_cycles = ideal_cycles
        self.energy = energy

class ComputeInfo:
    def __init__(self, pe_add, pe_mul, pe_div, heu_add, energy):
        self.pe_add = pe_add
        self.pe_mul = pe_mul
        self.pe_div = pe_div
        self.heu_add = heu_add
        self.energy = energy

class DRAMInfo:
    def __init__(self, domain_access, halo_access, domain_bw, halo_bw, total_bw, energy):
        self.domain_access = domain_access
        self.halo_access = halo_access
        self.domain_bw = domain_bw
        self.halo_bw = halo_bw
        self.total_bw = total_bw
        self.energy = energy

class SRAMInfo:
    def __init__(self, domain_access, spmat_access, halo_access, diag_access, energy, energy_wo_spmat):
        self.domain_access = domain_access
        self.halo_access = halo_access
        self.spmat_access = spmat_access
        self.diag_access = diag_access
        self.energy = energy
        self.energy_wo_spmat = energy_wo_spmat

class Info:
    def __init__(self, summ_info: SummaryInfo, compute_info: ComputeInfo,
                 dram_info: DRAMInfo, sram_info: SRAMInfo):
        self.summ = summ_info
        self.compute = compute_info
        self.dram = dram_info
        self.sram = sram_info