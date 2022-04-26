import torch

def corner_to_surfaces_3d_torch(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    Args:
        corners (float array, [N, 8, 3]): 3d box corners.
    Returns:
        surfaces (float array, [N, 6, 4, 3]):
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = torch.stack(
        [
            torch.stack([corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]]),
            torch.stack([corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]]),
            torch.stack([corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]]),
            torch.stack([corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]]),
            torch.stack([corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]]),
            torch.stack([corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]]),
        ]
    ).permute([2, 0, 1, 3])
    return surfaces

def rotation_3d_in_axis_torch(points, angles, axis=2):
    # points: [N, point_size, 3]
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos).to(points.device)
    zeros = torch.zeros_like(rot_cos).to(points.device)
    if axis == 2:
        rot_mat_T = torch.stack(
            [
                torch.stack([rot_cos, -rot_sin, zeros]),
                torch.stack([rot_sin, rot_cos, zeros]),
                torch.stack([zeros, zeros, ones]),
            ]
        )
    else:
        raise ValueError("axis should in range")

    return torch.einsum("aij,jka->aik", points, rot_mat_T)

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

def corners_nd_torch(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = torch.stack(
        unravel_index(torch.arange(2 ** ndim), [2] * ndim), axis=1
    ).to(dims.dtype).to(dims.device)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - torch.Tensor(origin).to(dims.dtype).to(dims.device)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])
    return corners

def center_to_corner_box3d_torch(centers, dims, angles=None, origin=(0.5, 0.5, 0.5), axis=2):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd_torch(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis_torch(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners

def surface_equ_3d_jitv2_torch(surfaces):
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    num_polygon = surfaces.shape[0]
    max_num_surfaces = surfaces.shape[1]
    normal_vec = torch.zeros((num_polygon, max_num_surfaces, 3), dtype=surfaces.dtype).to(surfaces.device)
    d = torch.zeros((num_polygon, max_num_surfaces), dtype=surfaces.dtype).to(surfaces.device)
    sv0 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    sv1 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    for i in range(num_polygon):
        for j in range(max_num_surfaces):
            sv0[0] = surfaces[i, j, 0, 0] - surfaces[i, j, 1, 0]
            sv0[1] = surfaces[i, j, 0, 1] - surfaces[i, j, 1, 1]
            sv0[2] = surfaces[i, j, 0, 2] - surfaces[i, j, 1, 2]
            sv1[0] = surfaces[i, j, 1, 0] - surfaces[i, j, 2, 0]
            sv1[1] = surfaces[i, j, 1, 1] - surfaces[i, j, 2, 1]
            sv1[2] = surfaces[i, j, 1, 2] - surfaces[i, j, 2, 2]
            normal_vec[i, j, 0] = sv0[1] * sv1[2] - sv0[2] * sv1[1]
            normal_vec[i, j, 1] = sv0[2] * sv1[0] - sv0[0] * sv1[2]
            normal_vec[i, j, 2] = sv0[0] * sv1[1] - sv0[1] * sv1[0]

            d[i, j] = (
                    -surfaces[i, j, 0, 0] * normal_vec[i, j, 0]
                    - surfaces[i, j, 0, 1] * normal_vec[i, j, 1]
                    - surfaces[i, j, 0, 2] * normal_vec[i, j, 2]
            )
    return normal_vec, d

#@torch.jit.script
def _points_in_convex_polygon_3d_jit_torch(
        points, polygon_surfaces, normal_vec, d, num_surfaces
):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = torch.ones((num_points, num_polygons), dtype=torch.bool).to(points.device)
    sign = 0.0

    points_multi_normal_vec_surfaces = []
    for k in range(max_num_surfaces):
        points_multi_normal_vec = points @ normal_vec[:, k, :].T + d[:, k].unsqueeze(0)
        points_multi_normal_vec_surfaces.append(points_multi_normal_vec.unsqueeze(-1)>=0)
    points_multi_normal_vec_surfaces = torch.cat(points_multi_normal_vec_surfaces, dim=-1)
    points_multi_normal_vec_surfaces = points_multi_normal_vec_surfaces.sum(-1) >0
    ret[points_multi_normal_vec_surfaces] = False

    return ret

    from IPython import embed; embed()
    raise ValueError('Stop.')
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                        points[i, 0] * normal_vec[j, k, 0]
                        + points[i, 1] * normal_vec[j, k, 1]
                        + points[i, 2] * normal_vec[j, k, 2]
                        + d[j, k]
                )
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


def points_in_convex_polygon_3d_jit_torch(points, polygon_surfaces, num_surfaces=None):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = torch.full((num_polygons,), 9999).to(torch.int64).to(points.device)
    normal_vec, d = surface_equ_3d_jitv2_torch(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_in_convex_polygon_3d_jit_torch(
        points, polygon_surfaces, normal_vec, d, num_surfaces
    )

def points_in_rbbox_torch(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0.5)):
    rbbox_corners = center_to_corner_box3d_torch(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, -1], origin=origin, axis=z_axis
    )
    surfaces = corner_to_surfaces_3d_torch(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit_torch(points[:, :3], surfaces)
    return indices