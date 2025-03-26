def get_stencil_points(stencil_type, dim, lower_flag):
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
                (0, 0, -1),
                (0, -1, 0),
                (-1, 0, 0),

                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
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
                (0, 0, -1),
                (-1, 0, 0),
                (0, -1, 0),

                (0, 0, -2),
                (-2, 0, 0),
                (0, -2, 0),

                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (0, 1, 0),
                (0, 2, 0),
                (0, 0, 1),
                (0, 0, 2),
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
                (0, 0, -1),
                (-1, 0, 0),
                (0, -1, 0),
                (-1, 1, 0),
                (-1, 0, 1),
                (0, -1, 1),


                (0, 0, 0),
                (1, 0, 0),
                (1, -1, 0),
                (0, 1, 0),
                (0, 0, 1),
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
                (-1, -1, -1),
                (0, -1, -1),
                (1, -1, -1),
                (-1, 0, -1),
                (0, 0, -1),
                (1, 0, -1),
                (-1, 1, -1),
                (0, 1, -1),
                (1, 1, -1),
                (-1, -1, 0),
                (0, -1, 0),
                (1, -1, 0),
                (-1, 0, 0),
                (0, 0, 0),
                (1, 0, 0),
                (-1, 1, 0),
                (0, 1, 0),
                (1, 1, 0),
                (-1, -1, 1),
                (0, -1, 1),
                (1, -1, 1),
                (-1, 0, 1),
                (0, 0, 1),
                (1, 0, 1),
                (-1, 1, 1),
                (0, 1, 1),
                (1, 1, 1),
            ]
    if lower_flag:
        if dim == 2:
            stencil_points = [dp for dp in stencil_points if 2 * dp[0] + dp[1] <= 0]
        else:
            stencil_points = [dp for dp in stencil_points if 4 * dp[0] + 2 * dp[1] + dp[2] <= 0]
    
    return stencil_points

if __name__ == "__main__":
    get_stencil_points(3, 3)