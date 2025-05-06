import sys, json, time, os
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed
from sim.matrix_gen import get_linear_system_sptrsv, get_linear_system_spmv
from sim.PreprocessSpTRSV import preprocess_sptrsv
from sim.PreprocessSpMV import preprocess_spmv
from sim.Accelerator import *
from main import read_config, set_allocate_DRAM_BW

stencil_str_2d = ["2D-Star-5P", "2D-Star-9P", "2D-Diamond-7P", "2D-Box-9P"]
stencil_str_3d= ["3D-Star-7P", "3D-Star-13P", "3D-Diamond-13P", "3D-Box-27P"]

def get_stencil_str(dims, stencil_type):
    if dims == 3:
        return stencil_str_3d[stencil_type]
    else:
        return stencil_str_2d[stencil_type]

def sim_task(size, config):
    dims = len(size)
    stencil_type = config["StencilType"]
    print(f"current size: {size}, stencil type: {stencil_type}, PE size: {config['Arch']['NumPEs']}, DRAM BW: {config['Mem']['DRAM_BW']}GB/s")

    start_time = time.time()
    if config["ComputeType"] == "spmv":
        data = get_linear_system_spmv(dims, stencil_type, size)
        data = preprocess_spmv(data, config["Arch"]["NumPEs"][0], config["Arch"]["NumPEs"][1], stencil_type, dims)
    else:
        data = get_linear_system_sptrsv(dims, stencil_type, size)
        data = preprocess_sptrsv(data, config["Arch"]["NumPEs"][0], config["Arch"]["NumPEs"][1], stencil_type, dims, config["ScheScheme"])
    print(f"Preprocessing finished, time usage: {time.time() - start_time:.2f}s")

    env = simpy.Environment()
    acc = Accelerator(env, config, data, progressbar=True)
    proc = env.process(acc.wait_for_finish())
    start_time = time.time()
    env.run(until=proc)
    print(f"\nSimulation Time: {time.time() - start_time:.2f}s")
    info = acc.compute_info()
    assert(acc.check_correctness())
    return info


def run_batch_sim(size_list, stencil_list, csv_path, max_threads):
    with ProcessPoolExecutor(max_workers=max_threads) as executor:
        tasks = {}
        for size in size_list:
            for stencil_type in stencil_list:
                dims = len(size)
                config = read_config("cfg/config_base.json")
                config["NumDims"] = dims
                config["StencilType"] = stencil_type
                set_allocate_DRAM_BW(config)
                tasks[executor.submit(sim_task, size, config)] = (size, config)

        with open(csv_path, "w") as csv_file:
            csv_file.write("Grid Size, Stencil Type, Cycles, Latency(ms), \
                        Compute Energy(pJ), SRAM Energy(pJ), DRAM Energy(pJ), Total Energy(pJ), \
                        Domain DRAM BW(GB/s), Halo DRAM BW(GB/s), Total BW(GB/s), PE utilization\n")
            for task in as_completed(tasks):
                size, config = tasks[task]
                info: Info = task.result()
                csv_file.write(
                    f"\"{size}\", {stencil_type}, {info.summ.cycles}, {info.summ.wall_clock_time * 1000}\
                    {info.compute.energy}, {info.sram.energy}, {info.dram.energy}, {info.summ.energy},\
                    {info.dram.domain_bw}, {info.dram.halo_bw},\
                    {info.dram.domain_bw + info.dram.halo_bw:.2f}, {info.summ.pe_util:.2f}%\n"
                )

def run_scalability_DRAM_BW_Stencil(size_list, dram_bw_list, max_threads):
    with ProcessPoolExecutor(max_workers=max_threads) as executor:
        tasks = {}
        for size in size_list:
            dims = len(size)
            for dram_bw in dram_bw_list:
                for stencil_type in range(3, 4):
                    config = read_config("cfg/config_base.json")
                    config["NumDims"] = dims
                    config["StencilType"] = stencil_type
                    config["Mem"]["DRAM_BW"] = dram_bw
                    set_allocate_DRAM_BW(config)
                    tasks[executor.submit(sim_task, size, config)] = (size, config)

        with open(csv_path, "w") as csv_file:
            csv_file.write("Size, Stencil Type, DRAM BW(GB/s), Cycles, Real DRAM BW(GB/s), PE utilization\n")
            for task in as_completed(tasks):
                info: Info = task.result()
                size, config = tasks[task]
                stencil_type = config["StencilType"]
                dram_bw = config["Mem"]["DRAM_BW"]
                csv_file.write(
                    f"\"{size}\", {stencil_type}, {dram_bw}, {info.summ.cycles},\
                    {info.dram.domain_access}, {info.dram.halo_access},\
                    {info.dram.total_bw:.2f}, {info.summ.pe_util:.2f}%\n"
                )
                csv_file.flush()

def run_scalability_DRAM_BW_PE(size_list, pe_size_list, dram_bw_list, max_threads):
    with ProcessPoolExecutor(max_workers=max_threads) as executor:
        tasks = {}
        stencil_type = 2
        for size in size_list:
            dims = len(size)
            for dram_bw in dram_bw_list:
                for pe_size in pe_size_list:
                    config = read_config("cfg/config_base.json")
                    config["NumDims"] = dims
                    config["StencilType"] = stencil_type
                    config["Mem"]["DRAM_BW"] = dram_bw
                    config["Arch"]["NumPEs"] = pe_size
                    set_allocate_DRAM_BW(config)
                    tasks[executor.submit(sim_task, size, config)] = (size, config)

        with open(csv_path, "w") as csv_file:
            csv_file.write("DRAM BW(GB/s), PE size, Cycles, Real DRAM BW(GB/s), PE utilization\n")
            for task in as_completed(tasks):
                info: Info = task.result()
                size, config = tasks[task]
                stencil_type = config["StencilType"]
                dram_bw = config["Mem"]["DRAM_BW"]
                csv_file.write(
                    f"{dram_bw}, {config['Arch']['NumPEs'][0]}, {info.summ.cycles},\
                    {info.dram.total_bw:.2f}, {info.summ.pe_util:.2f}%\n"
                )
                csv_file.flush()

def run_scalability_Veclane(size_list, vec_lane_list, max_threads):
    with ProcessPoolExecutor(max_workers=max_threads) as executor:
        tasks = {}
        for size in size_list:
            for vec_lane in vec_lane_list:
                dims = len(size)
                for stencil_type in range(4):
                    config = read_config("cfg/config_base.json")
                    config["NumDims"] = dims
                    config["StencilType"] = stencil_type
                    config["Arch"]["VecLanes"] = vec_lane
                    config["Mem"]["DRAM_BW"] = 1024 # make sure DRAM is not bottlenecked
                    set_allocate_DRAM_BW(config)
                    tasks[executor.submit(sim_task, size, config)] = (size, config)

        with open(csv_path, "w") as csv_file:
            csv_file.write("Size, Stencil Type, VecLanes, Cycles, PE utilization\n")
            for task in as_completed(tasks):
                info: Info = task.result()
                size, config = tasks[task]
                dims = len(size)
                stencil_type = config["StencilType"]
                csv_file.write(
                    f"{size[0]}, {get_stencil_str(dims, stencil_type)},\
                    {config['Arch']['VecLanes']}, {info.summ.cycles},\
                    {info.summ.pe_util:.2f}%\n"
                )
                csv_file.flush()

def run_tile_pipeline(size_list, stencil_list, csv_path, max_threads):
    with ProcessPoolExecutor(max_workers=max_threads) as executor:
        tasks = {}
        for size in size_list:
            for stencil_type in stencil_list:
                for i in range(2):
                    dims = len(size)
                    config = read_config("cfg/config_base.json")
                    config["NumDims"] = dims
                    config["StencilType"] = stencil_type
                    set_allocate_DRAM_BW(config)
                    if i == 1:
                        if dims == 3:
                            input_size = (config["Arch"]["NumPEs"][0], config["Arch"]["NumPEs"][1], size[2])
                        else:
                            input_size = (config["Arch"]["NumPEs"][0], config["Arch"]["NumPEs"][1])

                        tasks[executor.submit(sim_task, input_size, config)] = (size, config, i)
                    else:
                        tasks[executor.submit(sim_task, size, config)] = (size, config, i)

        with open(csv_path, "w") as csv_file:
            csv_file.write("Size, Stencil Type, w/o Pipelining, Latency(ms)\n")
            for task in as_completed(tasks):
                size, config, pipe_flag = tasks[task]
                dims = len(size)
                stencil_type = config["StencilType"]
                info: Info = task.result()
                if pipe_flag == 1:
                    factor_x = math.ceil(size[0] / config["Arch"]["NumPEs"][0])
                    factor_y = math.ceil(size[1] / config["Arch"]["NumPEs"][1])
                    wall_clock_time = factor_x * factor_y * info.summ.wall_clock_time
                else:
                    wall_clock_time = info.summ.wall_clock_time

                pipe_str = "without" if pipe_flag else "with"
                csv_file.write(
                    f"{size[0]}, {get_stencil_str(dims, stencil_type)}, {pipe_str}, {wall_clock_time * 1000}\n"
                )

def run_energy_breakdown(size_list, csv_path, max_threads):
    with ProcessPoolExecutor(max_workers=max_threads) as executor:
        tasks = {}
        for size in size_list:
            for stencil_type in range(4):
                dims = len(size)
                config = read_config("cfg/config_base.json")
                config["NumDims"] = dims
                config["StencilType"] = stencil_type
                set_allocate_DRAM_BW(config)
                tasks[executor.submit(sim_task, size, config)] = (size, config)

        with open(csv_path, "w") as csv_file:
            csv_file.write("Size, Stencil Type, Adiag, b & x, HaloVector,\
                            DRAM Energy(pJ), SRAM Energy(pJ), Computation Energy(pJ)\n")
            for task in as_completed(tasks):
                size, config = tasks[task]
                dims = len(size)
                stencil_type = config["StencilType"]
                info: Info = task.result()
                # sram_access = info.sram.diag_access + info.sram.domain_access + info.sram.halo_access
                csv_file.write(
                    f"{size[0]}, {get_stencil_str(dims, stencil_type)}, \
                    {info.sram.diag_access}, {info.sram.domain_access}, {info.sram.halo_access}, \
                    {info.dram.energy}, {info.sram.energy_wo_spmat}, {info.compute.energy}\n"
                )

def run_solver_comp(size_list, stencil_list, csv_path, max_threads):
    with ProcessPoolExecutor(max_workers=max_threads) as executor:
        tasks = {}
        for size in size_list:
            for stencil_type in stencil_list:
                dims = len(size)
                config = read_config("cfg/config_solver.json")
                config["NumDims"] = dims
                config["StencilType"] = stencil_type
                set_allocate_DRAM_BW(config)
                tasks[executor.submit(sim_task, size, config)] = (size, config)

        with open(csv_path, "w") as csv_file:
            csv_file.write("Grid Size, Stencil Type, Cycles, Latency(ms), Total Energy(pJ), PE utilization\n")
            for task in as_completed(tasks):
                size, config = tasks[task]
                info: Info = task.result()
                csv_file.write(
                    f"\"{size}\", {stencil_type}, {info.summ.cycles}, {info.summ.wall_clock_time * 1000}\
                    {info.summ.energy}, {info.summ.pe_util:.2f}%\n"
                )

def run_sche_comp(size_list, stencil_list, csv_path, max_threads):
    with ProcessPoolExecutor(max_workers=max_threads) as executor:
        tasks = {}
        for size in size_list:
            for stencil_type in stencil_list:
                for sche_scheme in ["default", "wavefront"]:
                    dims = len(size)
                    config = read_config("cfg/config_base.json")
                    config["NumDims"] = dims
                    config["StencilType"] = stencil_type
                    config["ScheScheme"] = sche_scheme
                    set_allocate_DRAM_BW(config)
                    tasks[executor.submit(sim_task, size, config)] = (size, config)


        with open(csv_path, "w") as csv_file:
            csv_file.write("Grid Size, Stencil Type, Schedule Scheme, Cycles, PE utilization\n")
            for task in as_completed(tasks):
                size, config = tasks[task]
                info: Info = task.result()
                stencil_type = config["StencilType"]
                csv_file.write(
                    f"\"{size}\", {stencil_type}, {config['ScheScheme']}, {info.summ.cycles}, {info.summ.pe_util:.2f}%\n"
                )

def run_spmv_sim(size_list, stencil_list, csv_path, max_threads):
    with ProcessPoolExecutor(max_workers=max_threads) as executor:
        tasks = {}
        for size in size_list:
            for stencil_type in stencil_list:
                dims = len(size)
                config = read_config("cfg/config_base.json")
                config["ComputeType"] = "spmv"
                config["ScheScheme"] = "default"
                config["NumDims"] = dims
                config["StencilType"] = stencil_type
                set_allocate_DRAM_BW(config)
                tasks[executor.submit(sim_task, size, config)] = (size, config)

        with open(csv_path, "w") as csv_file:
            csv_file.write("Grid Size, Stencil Type, Cycles, Latency(ms), \
                        Total Energy(pJ), Compute Energy(pJ), SRAM Energy(pJ), DRAM Energy(pJ), \
                        Domain DRAM BW(GB/s), Halo DRAM BW(GB/s), Total BW(GB/s), PE utilization\n")
            for task in as_completed(tasks):
                size, config = tasks[task]
                info: Info = task.result()
                stencil_type = config["StencilType"]
                dims = config["NumDims"]
                csv_file.write(
                    f"\"{size[0]}\",{get_stencil_str(dims, stencil_type)},{info.summ.cycles},{info.summ.wall_clock_time * 1000},\
                    {info.summ.energy},{info.compute.energy},{info.sram.energy},{info.dram.energy},\
                    {info.dram.domain_bw},{info.dram.halo_bw},\
                    {info.dram.domain_bw + info.dram.halo_bw:.2f},{info.summ.pe_util:.2f}%\n"
                )

if __name__ == "__main__":
    logger.remove()
    if len(sys.argv) != 4:
        print("Usage: python exp.py [csv_path] [exp_type] [max_procs]")
    csv_path = sys.argv[1]
    exp_type = sys.argv[2]
    max_procs = int(sys.argv[3])

    if exp_type == "sptrsv":
        size_lists = [
            (64, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096)
        ]
        run_batch_sim(size_lists, [0], csv_path, max_procs)

    elif exp_type == "spmv":
        size_list = [
            (64, 64, 64),
            (96, 96, 96),
            (128, 128, 128),
            # (256, 256, 256),
            (256, 256),
            (512, 512),
            (768, 768),
            (1024, 1024),
        ]
        run_spmv_sim(size_list, range(4), csv_path, max_procs)

    elif exp_type == "scale_stencil":
        DRAM_BW_list = range(896, 1024 + 1, 64)
        size_list = [(64, 64, 64)]
        run_scalability_DRAM_BW_PE(size_list, DRAM_BW_list, max_procs)
    elif exp_type == "scale_pe":
        DRAM_BW_list = range(64, 1024 + 1, 64)
        size_list = [(64, 64, 64)]
        pe_size_list = [(i, i) for i in range(4, 12 + 1)]
        run_scalability_DRAM_BW_PE(size_list, pe_size_list, DRAM_BW_list, max_procs)
    elif exp_type == "scale_veclane":
        size_list = [(64, 64, 64), (1024, 1024)]
        # PE_size_list = [(i, i) for i in range(4, 8 + 1)]
        vec_lane_list = range(5, 5 + 1)
        run_scalability_Veclane(size_list, vec_lane_list, max_procs)
    elif exp_type == "pipeline":
        size_list = [(i, i, i) for i in range(32, 256 + 1, 32)]
        run_tile_pipeline(size_list, [2], csv_path, max_procs)
    elif exp_type == "breakdown":
        size_list = [
            [64, 64, 64],
            [256, 256, 256],
            [1024, 1024],
            [4096, 4096]
        ]
        run_energy_breakdown(size_list, csv_path, max_procs)
    elif exp_type == "solver":
        size_list = [
            (64, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096)
        ]
        run_solver_comp(size_list, [0], csv_path, max_procs)
    elif exp_type == "sche_comp":
        size_list = [
            (64, 64), (128, 128), (256, 256), (512, 512),
            (1024, 1024), (2048, 2048), (3072, 3072), (4096, 4096)
        ]
        run_sche_comp(size_list, [3], csv_path, max_procs)
