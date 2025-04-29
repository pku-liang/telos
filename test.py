import sys
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed
from sim.Accelerator import *
from main import run_sim, read_config

def test_multithread(compute_type, size_list, stencil_list, max_threads):
    passed_n = 0
    total_n = len(size_list) * len(stencil_list)
    with ProcessPoolExecutor(max_workers=max_threads) as executor:
        tasks = {}
        for size in size_list:
            for stencil_type in stencil_list:
                dims = len(size)
                config = read_config("cfg/config_base.json")
                config["ComputeType"] = compute_type
                config["NumDims"] = dims
                config["StencilType"] = stencil_type
                config["ScheScheme"] = "default" if compute_type == "spmv" else "wavefront"
                tasks[executor.submit(run_sim, config, size, True)] = (size, config)

        for task in as_completed(tasks):
            correct_flag = task.result()
            if correct_flag:
                passed_n += 1

    print(f"TEST {compute_type}: TOTAL {total_n}, PASSED {passed_n}, FAILED {total_n - passed_n}")

def test_seq(compute_type, size_list, stencil_list):
    passed_n = 0
    total_n = len(size_list) * len(stencil_list)

    for size in size_list:
        for stencil_type in stencil_list:
            dims = len(size)
            config = read_config("cfg/config_base.json")
            config["ComputeType"] = compute_type
            config["NumDims"] = dims
            config["StencilType"] = stencil_type
            config["ScheScheme"] = "default" if compute_type == "spmv" else "wavefront"
            correct_flag = run_sim(config, size, True)
            if correct_flag:
                passed_n += 1

    print(f"TEST {compute_type}: TOTAL {total_n}, PASSED {passed_n}, FAILED {total_n - passed_n}")

if __name__ == "__main__":
    logger.remove()
    assert(len(sys.argv) == 2)
    compute_type = sys.argv[1]

    size_list = [
        (1, 64),
        (2, 32),
        (4, 16),
        (8, 8),
        (8, 14),
        (14, 15),
        (64, 64),
        (64, 128),
        (256, 301),
        # (512, 512),
        # (495, 1024),

        (1, 6, 100),
        (101, 2, 10),
        (2, 101, 10),
        (10, 10, 10),
        (64, 64, 64),
        (8, 8, 300),
        (300, 8, 10),
        # (256, 256, 256)
    ]
    stencil_list = range(4)
    # test_seq("spmv", size_list, stencil_list)
    test_multithread(compute_type, size_list, stencil_list, 32)