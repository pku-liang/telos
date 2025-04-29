import json
import simpy
import sys
import os
from loguru import logger
from sim.matrix_gen import get_linear_system_sptrsv, get_linear_system_spmv
from sim.PreprocessSpTRSV import preprocess_sptrsv
from sim.PreprocessSpMV import preprocess_spmv
from sim.Accelerator import *
import time

def set_allocate_DRAM_BW(cfg):
    clock_freq = cfg["Freq"]
    dram_bw = cfg["Mem"]["DRAM_BW"] / (clock_freq * cfg["DataWidth"])
    stencil_type = cfg["StencilType"]
    dims = cfg["NumDims"]
    pe_x = cfg["Arch"]["NumPEs"][0]
    pe_y = cfg["Arch"]["NumPEs"][1]
    halo_points_tp = get_num_halo_points(stencil_type, pe_x, pe_y, cfg["ComputeType"])
    pe_n = pe_x * pe_y

    if cfg["ComputeType"] == "sptrsv":
        stencil_points = len(get_stencil_points(dims, stencil_type)) // 2
        spmat_ratio = stencil_points * pe_n
        domain_ratio = 3 * pe_n
        halo_ratio = 2 * sum(halo_points_tp)
    else:
        stencil_points = len(get_stencil_points(dims, stencil_type))
        spmat_ratio = stencil_points * pe_n
        domain_ratio = 2 * pe_n
        halo_ratio = 4 * sum(halo_points_tp)

    total_ratio = spmat_ratio + domain_ratio + halo_ratio
    spmat_bw = round(dram_bw * spmat_ratio / total_ratio)
    domain_bw = round(dram_bw * domain_ratio / total_ratio)
    halo_bw = round(dram_bw * halo_ratio / total_ratio)
    # print(f"Allocated Spmat BW: {spmat_bw}, Domain BW: {domain_bw}, Halo BW: {halo_bw}")
    cfg["Mem"]["SpMat_DRAM_BW"] = spmat_bw
    cfg["Mem"]["Domain_DRAM_BW"] = domain_bw
    cfg["Mem"]["Halo_DRAM_BW"] = halo_bw

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def dump_debug_log():
    all_log_path = "log/all.log"
    aggr_log_path = "log/aggr.log"
    scalar_log_path = "log/scalar.log"
    vec_log_path = "log/vec.log"
    heu_log_path = "log/heu.log"
    compute_log_path = "log/compute.log"
    domain_data_log_path = "log/domain_data.log"

    if os.path.exists(all_log_path):
        os.remove(all_log_path)
    if os.path.exists(aggr_log_path):
        os.remove(aggr_log_path)
    if os.path.exists(vec_log_path):
        os.remove(vec_log_path)
    if os.path.exists(scalar_log_path):
        os.remove(scalar_log_path)
    if os.path.exists(heu_log_path):
        os.remove(heu_log_path)
    if os.path.exists(compute_log_path):
        os.remove(compute_log_path)
    if os.path.exists(domain_data_log_path):
        os.remove(domain_data_log_path)

    logger.add(all_log_path, format="<level>{message}</level>",
            level="TRACE", filter=lambda r: "" in r["message"])
    logger.add(aggr_log_path, format="<level>{message}</level>",
            level="TRACE", filter=lambda r: "Aggregator" in r["message"])
    logger.add(scalar_log_path, format="<level>{message}</level>",
            level="TRACE", filter=lambda r: "ScalarUnit" in r["message"])
    logger.add(vec_log_path, format="<level>{message}</level>",
            level="TRACE", filter=lambda r: "VectorUnit" in r["message"])
    logger.add(heu_log_path, format="<level>{message}</level>",
            level="TRACE", filter=lambda r: "PEArray" in r["message"])
    logger.add(compute_log_path, format="<level>{message}</level>",
            level="TRACE", filter=lambda r: "PE(0, 0) ScalarUnit: compute variable" in r["message"])
    logger.add(domain_data_log_path, format="<level>{message}</level>",
            level="TRACE", filter=lambda r: "DomainData" in r["message"])


def run_sim(config, size, test_flag):
    stencil = config["StencilType"]
    dims = config["NumDims"]
    print(f"Current Test: Compute Type={config['ComputeType']}, Size={size}, Stencil Type={stencil}")
    set_allocate_DRAM_BW(config)

    if config["ComputeType"] == "sptrsv": # sptrsv
        start_time = time.time()
        # gen testbench
        data = get_linear_system_sptrsv(dims, stencil, size)
        # preprocess data
        data = preprocess_sptrsv(data, config["Arch"]["NumPEs"][0], config["Arch"]["NumPEs"][1], stencil, dims, config["ScheScheme"])
        print(f"Data preprocessing finished, used time: {time.time() - start_time:.2f}s")

    elif config["ComputeType"] == "spmv": # spmv
        if config["ScheScheme"] != "default":
            raise RuntimeError("SpMV only support default schedule scheme")
        start_time = time.time()
        # gen testbench
        data = get_linear_system_spmv(dims, stencil, size)
        # preprocess data
        data = preprocess_spmv(data, config["Arch"]["NumPEs"][0], config["Arch"]["NumPEs"][1], stencil, dims)
        print(f"Data preprocessing finished, used time: {time.time() - start_time:.2f}s")
    else:
        raise RuntimeError(f"Unsupported compute type {config['ComputeType']}\nSupported compute type: sptrsv, spmv")

    env = simpy.Environment()
    acc = Accelerator(env, config, data, progressbar=True)
    proc = env.process(acc.wait_for_finish())
    start_time = time.time()
    env.run(until=proc)
    correct_flag = acc.check_correctness()
    if not test_flag:
        print(f"\nSimulation Time: {time.time() - start_time:.2f}s")
        print(f"Correctness: {correct_flag}")
        info = acc.compute_info()
        acc.print(info)
    else:
        str = "PASSED" if correct_flag else "FAILED"
        print(f"TEST {str}, Config=[Size={size}, Stencil Type={stencil}]")
    return correct_flag


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <config_file> x y z [debug_flag]")
        sys.exit(1)

    config_file = sys.argv[1]
    config = read_config(config_file)
    dims = config["NumDims"]
    x = int(sys.argv[2])
    y = int(sys.argv[3])
    if dims == 3:
        z = int(sys.argv[4])
        debug_flag = False if len(sys.argv) < 6 else True
        size = (x, y, z)
    else:
        debug_flag = False if len(sys.argv) < 5 else True
        size = (x, y)

    logger.remove()
    print(f"Debug Mode: {debug_flag}")
    if debug_flag:
        dump_debug_log()

    run_sim(config, size, False)
