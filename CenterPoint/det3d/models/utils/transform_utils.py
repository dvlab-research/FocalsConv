import math
import torch

try:
    from kornia.geometry.conversions import (
        convert_points_to_homogeneous,
        convert_points_from_homogeneous,
    )
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')


def project_to_image(project, points):
    """
    Project points to image
    Args:
        project [torch.tensor(..., 3, 4)]: Projection matrix
        points [torch.Tensor(..., 3)]: 3D points
    Returns:
        points_img [torch.Tensor(..., 2)]: Points in image
        points_depth [torch.Tensor(...)]: Depth of each point
    """
    # Reshape tensors to expected shape
    points = convert_points_to_homogeneous(points)
    points = points.unsqueeze(dim=-1)
    project = project.unsqueeze(dim=1)

    # Transform points to image and get depths
    points_t = project @ points
    points_t = points_t.squeeze(dim=-1)
    points_img = convert_points_from_homogeneous(points_t)
    points_depth = points_t[..., -1] - project[..., 2, 3]

    return points_img, points_depth

def camera_to_image(project, points, normalize=True):
    """
    Camera points to image
    Args:
        project [torch.tensor(..., 3, 3)]: Intrinsic matrix
        points [torch.Tensor(..., 3)]: 3D points
    Returns:
        points_img [torch.Tensor(..., 2)]: Points in image
    """
    # Reshape tensors to expected shape
    points = convert_points_to_homogeneous(points)
    project_pad = torch.eye(4).unsqueeze(0)
    project_pad = project_pad.repeat(project.shape[0],1,1).to(project.device)
    project_pad[:,:project.shape[1],:project.shape[2]] = project

    # Transform points to image and get depths
    points_t = torch.bmm(points, project_pad.permute(0,2,1))
    points_img = convert_points_from_homogeneous(points_t)
    if normalize:
        points_img = points_img / points_img[...,-1].unsqueeze(-1)

    return points_img[...,:2]


def normalize_coords(coords, shape):
    """
    Normalize coordinates of a grid between [-1, 1]
    Args:
        coords: (..., 3), Coordinates in grid
        shape: (3), Grid shape
    Returns:
        norm_coords: (.., 3), Normalized coordinates in grid
    """
    min_n = -1
    max_n = 1
    shape = torch.flip(shape, dims=[0])  # Reverse ordering of shape

    # Subtract 1 since pixel indexing from [0, shape - 1]
    norm_coords = coords / (shape - 1) * (max_n - min_n) + min_n
    return norm_coords