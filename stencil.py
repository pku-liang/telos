# 返回每个stage包含的通道数
def get_stage_lanes(dims, stencil_type, compute_type):
    if compute_type == "sptrsv":
        stage_lanes_3d = [[3], [3, 3], [3, 2, 1], [3, 2, 3, 1, 2, 1, 1]]
        stage_lanes_2d = [[2], [2, 2], [2, 1], [2, 1, 1]]
    else:
        stage_lanes_3d = [[1, 5, 1], [1, 1, 9, 1, 1], [3, 7, 3], [9, 9, 9]]
        stage_lanes_2d = [[5], [9], [7], [9]]
    return stage_lanes_3d[stencil_type] if dims == 3 else stage_lanes_2d[stencil_type]


# 返回通道ID对应的stage ID
def get_id2stage(dims, stencil_type, compute_type):
    if compute_type == "sptrsv":
        id2stage = []
        stage_lanes = get_stage_lanes(dims, stencil_type, compute_type)
        for i, lane_n in enumerate(stage_lanes):
            id2stage += [i] * lane_n
        return id2stage
    else:
        stencil_length = len(get_stencil_points(dims, stencil_type))
        stencil_id2stage_3d = [
            [2, 1, 1, 1, 1, 1, 0],
            [4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0],
            [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [2, 2, 2, 2, 2, 2, 2, 2, 2,
            1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            ]
        ]
        return stencil_id2stage_3d[stencil_type] if dims == 3 else [0 for _ in range(stencil_length)]


def get_stages(dims, stencil_type, compute_type):
    return len(get_stage_lanes(dims, stencil_type, compute_type))

# 返回position索引的halo_points个数的list
def get_num_halo_points(stencil_type, tile_x, tile_y, compute_type):
    if compute_type == "sptrsv":
        if stencil_type == 0: 
            return tile_y, tile_x
        if stencil_type == 1:
            return 2 * tile_y, 2 * tile_x
        if stencil_type == 2:
            return tile_y + 1, tile_x
        if stencil_type == 3:
            return tile_y + 1, 2 * tile_x
    else:
        if stencil_type == 0: 
            return tile_y, tile_x, tile_y, tile_x
        if stencil_type == 1: 
            return 2 * tile_y, 2 * tile_x, 2 * tile_y, 2 * tile_x
        if stencil_type == 2:
            return tile_y + 1, tile_x, tile_y + 1, tile_x
        if stencil_type == 3:
            return tile_y + 1, tile_x + 1, tile_y + 1, tile_x + 1
    
def get_num_stencil_points(dim, stencil_type, compute_type):
    if compute_type == "sptrsv":
        return len(get_affine_stencil_points(dim, stencil_type))
    else:
        return len(get_stencil_points(dim, stencil_type))

def get_stencil_points(dim, stencil_type):
    if stencil_type == 0:
        # 3D-Star-7P/2D-Star-5P
        if dim == 2:
            stencil_points = [
                (0, -1),
                (-1, 0),
                (0, 0),
                (1, 0),
                (0, 1),
            ]
        elif dim == 3:
            stencil_points = [
                (0, 0, 1),
                
                (0, -1, 0),
                (-1, 0, 0),
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),

                (0, 0, -1),
            ]
    elif stencil_type == 1:
        # 3D-Star-13P/2D-Star-9P
        if dim == 2:
            stencil_points = [
                (0, -2),
                (0, -1),
                (-2, 0),
                (-1, 0),
                (0, 0),
                (1, 0),
                (2, 0),
                (0, 1),
                (0, 2),
            ]
        elif dim == 3:
            stencil_points = [
                (0, 0, 2),
                (0, 0, 1),

                (-1, 0, 0),
                (0, -1, 0),
                (-2, 0, 0),
                (0, -2, 0),
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (0, 1, 0),
                (0, 2, 0),

                (0, 0, -1),
                (0, 0, -2),
            ]
    elif stencil_type == 2:
        # 3D-Diamond-13P / 2D-Diamond-7P
        if dim == 2:
            stencil_points = [
                (0, -1),
                (1, -1),
                (-1, 0),
                (0, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
            ]
        elif dim == 3:
            stencil_points = [
                (-1, 0, 1),
                (0, -1, 1),
                (0, 0, 1),
                
                (-1, 0, 0),
                (0, -1, 0),
                (-1, 1, 0),
                (0, 0, 0),
                (1, 0, 0),
                (1, -1, 0),
                (0, 1, 0),

                (0, 0, -1),
                (1, 0, -1),
                (0, 1, -1),
            ]
    elif stencil_type == 3:
        # 3D-Box-27P / 2D-Box-9P
        if dim == 2:
            stencil_points = [
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (0, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ]
        elif dim == 3:
            stencil_points = [
                (-1, -1, 1),
                (0, -1, 1),
                (1, -1, 1),
                (-1, 0, 1),
                (0, 0, 1),
                (1, 0, 1),
                (-1, 1, 1),
                (0, 1, 1),
                (1, 1, 1),

                (-1, -1, 0),
                (0, -1, 0),
                (1, -1, 0),
                (-1, 0, 0),
                (0, 0, 0),
                (1, 0, 0),
                (-1, 1, 0),
                (0, 1, 0),
                (1, 1, 0),

                (-1, -1, -1),
                (0, -1, -1),
                (1, -1, -1),
                (-1, 0, -1),
                (0, 0, -1),
                (1, 0, -1),
                (-1, 1, -1),
                (0, 1, -1),
                (1, 1, -1),
            ]
    return stencil_points

# 经过仿射变换后的stencil相对偏移
def get_affine_stencil_points(dim, stencil_type):
    if stencil_type == 0:
        # 3D-Star-7P/2D-Star-5P
        if dim == 2:
            stencil_points = [
                (-1, 0), (0, -1),
            ]
        elif dim == 3:
            stencil_points = [
                (0, 0, -1), (0, -1, 0), (-1, 0, 0),
            ]
    elif stencil_type == 1:
        # 3D-Star-13P/2D-Star-9P
        if dim == 2:
            stencil_points = [
                (-1, 0), (0, -1),
                (-2, 0), (0, -2),
            ]
        elif dim == 3:
            stencil_points = [
                (0, 0, -1), (-1, 0, 0), (0, -1, 0),
                (0, 0, -2), (-2, 0, 0), (0, -2, 0),
            ]
    elif stencil_type == 2:
        # 3D-Diamond-13P / 2D-Diamond-7P
        if dim == 2:
            stencil_points = [
                (-1, 0), (0, -1),
                (-1, -1),
            ]
        elif dim == 3:
            stencil_points = [
                (0, 0, -1), (-1, 0, 0), (0, -1, 0),
                (0, -1, -1), (-1, -1, 0),
                (-1, -1, -1),
            ]
    elif stencil_type == 3:
        # 3D-Box-27P / 2D-Box-9P
        if dim == 2:
            stencil_points = [
                (-1, 0), (0, -1),
                (-1, -1),
                (-1, -2),
            ]
        elif dim == 3:
            stencil_points = [
                (0, 0, -1), (-1, 0, 0), (0, -1, 0),
                (0, -1, -1), (-1, 0, -1),
                (0, -1, -2), (-1, -1, -1), (-1, 0, -2),
                (-1, -1, -2),
                (-1, -1, -3), (-1, -2, -2),
                (-1, -2, -3),
                (-1, -2, -4),
            ]
    return stencil_points