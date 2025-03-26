from tqdm import tqdm
from stencil import get_stencil_points
from math import ceil

# fully assocative cache
class LRUCache:
    def __init__(self, cache_lines, line_size):
        self.cache_lines = cache_lines
        self.line_size = line_size
        self.content = []
        self.miss_cnt = 0
        self.hit_cnt = 0

    def access(self, addr, wr):
        for i, id in enumerate(self.content):
            if addr >= id and addr < id + self.line_size: # hit
                self.hit_cnt += 1
                self.content.pop(i)
                self.content.append(id)
                # print(f"Cache hit addr: {addr}")
                return

        if not wr:
            self.miss_cnt += 1
            # print(f"Cache miss addr: {addr}")
        else:
            # print(f"Cache write addr: {addr}")
            pass

        if len(self.content) == self.cache_lines:
            self.content.pop(0)
        self.content.append((addr // self.line_size) * self.line_size)



def compute_SymGS_tiles(stencil_type, size_list, tile_size=8):
    dim = len(size_list)
    assert(dim == 2 or dim == 3)
    grid_n = size_list[0] * size_list[1] * (size_list[2] if dim == 3 else 1)
    stencil_points = get_stencil_points(stencil_type, dim, True)
    row_tiles = ceil(grid_n / tile_size)
    print(row_tiles)
    total_tiles = 0

    cache = LRUCache(cache_lines=16, line_size=8)
    for tile_id in tqdm(range(row_tiles)):
        col_ids = []
        # col_ids = set()
        for tile_offset in range(min(tile_size, grid_n - tile_id * tile_size)):
            row_id = tile_id * tile_size + tile_offset
            assert(row_id < grid_n)
            if dim == 2:
                i = row_id // size_list[1]
                j = row_id % size_list[1]
                for dp in stencil_points:
                    col_id = (i + dp[0]) * size_list[1] + j + dp[1]
                    if col_id >= 0 and col_id < tile_id * tile_size:
                        col_ids.append(col_id)
                        # col_ids.add(col_id)

            else:
                i = row_id // (size_list[1] * size_list[2])
                tmp = row_id % (size_list[1] * size_list[2])
                j = tmp // size_list[2]
                k = tmp % size_list[2]
                for dp in stencil_points:
                    col_id = (i + dp[0]) * size_list[0] * size_list[1] + (j + dp[1]) * size_list[1] + (k + dp[2])
                    if col_id >= 0 and col_id < tile_id * tile_size:
                        col_ids.append(col_id)
                        # col_ids.add(col_id)

        gemv_tiles = compute_gemv_tiles(col_ids, tile_size, cache)
        # gemv_tiles = ceil(len(col_ids) / tile_size)
        total_tiles += (gemv_tiles + 1)
        cache.access(tile_id * tile_size, True)

    print(f"Hit: {cache.hit_cnt}, Miss: {cache.miss_cnt}")
    # assert(cache.hit_cnt + cache.miss_cnt == total_tiles - row_tiles)
    return total_tiles - row_tiles, row_tiles, cache.miss_cnt, cache.hit_cnt


def compute_gemv_tiles(col_ids, tile_size, cache):
    col_ids.sort()
    tot_len = len(col_ids)
    cur_ptr = 0
    tiles = 0
    col_id = 0
    while cur_ptr < tot_len:
        col_id = col_ids[cur_ptr]
        cache.access(col_id, False)
        end_id = (col_id // tile_size + 1) * tile_size
        while cur_ptr < tot_len and col_ids[cur_ptr] < end_id:
            cur_ptr += 1
        tiles += 1
    return tiles

def compute_SpMV_tiles(stencil_type, size_list, tile_size=8):
    dim = len(size_list)
    assert(dim == 2 or dim == 3)
    grid_n = size_list[0] * size_list[1] * (size_list[2] if dim == 3 else 1)
    stencil_points = get_stencil_points(stencil_type, dim, False)
    row_tiles = ceil(grid_n / tile_size)

    total_tiles = 0
    for tile_id in tqdm(range(row_tiles)):
        col_ids = []
        # col_ids = set()
        for tile_offset in range(min(tile_size, grid_n - tile_id * tile_size)):
            row_id = tile_id * tile_size + tile_offset
            assert(row_id < grid_n)
            if dim == 2:
                i = row_id // size_list[1]
                j = row_id % size_list[1]
                for dp in stencil_points:
                    col_id = (i + dp[0]) * size_list[1] + j + dp[1]
                    if col_id >= 0:
                        col_ids.append(col_id)
                        # col_ids.add(col_id)
            else:
                i = row_id // (size_list[1] * size_list[2])
                tmp = row_id % (size_list[1] * size_list[2])
                j = tmp // size_list[2]
                k = tmp % size_list[2]
                for dp in stencil_points:
                    col_id = (i + dp[0]) * size_list[0] * size_list[1] + (j + dp[1]) * size_list[1] + (k + dp[2])
                    if col_id >= 0:
                        col_ids.append(col_id)
                        # col_ids.add(col_id)

        gemv_tiles = compute_gemv_tiles(col_ids, tile_size)
        # gemv_tiles = ceil(len(col_ids) / tile_size)
        total_tiles += gemv_tiles

    return total_tiles

if __name__ == "__main__":
    grid_size = [1024, 1024]
    tile_size = 12
    stencil_type = 3
    gemv_tiles, symgs_tiles = compute_SymGS_tiles(stencil_type, grid_size, tile_size)
    spmv_gemv_tiles = compute_SpMV_tiles(stencil_type, grid_size, tile_size)
    print(f"SpTRSV: D_SymGS tiles: {symgs_tiles} GEMV tiles(elements): {gemv_tiles}")
    print(f"SpMV: GEMV tiles: {spmv_gemv_tiles}")