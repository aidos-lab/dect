"""
This the functional (in the programming sense) implementation of the Euler
Characteristic Transform. Only contains core functions, without the torch or
torch geometric modules.

Current implementations, except the calculation of the ECT for point clouds, 
are naive implementations and have rather large memory requirements since the 
ECT of each simplex is calculated individually. 

For an fast implementation of the ECT please look at the `fect.py`, which for 
graphs handles large graphs at high resolution. 

Accelerating the implementation for the differentiable case is an active part of 
research. The non-differentiable case is easily scalable to medium large simplicial 
complexes.
"""

from typing import Callable, TypeAlias

import torch
from dect.ect_fn import indicator

Tensor: TypeAlias = torch.Tensor
"""@private"""


def normalize(ect):
    """Returns the normalized ect, scaled to lie in the interval 0,1"""
    return ect / torch.amax(ect, dim=(2, 3)).unsqueeze(2).unsqueeze(2)


def compute_ect(
    x: Tensor,
    *simplices,
    v: Tensor,
    radius: float,
    resolution: int,
    scale: float,
    index: Tensor | None = None,
    ect_fn: Callable[..., Tensor] = indicator,
) -> Tensor:
    """
    NOTE: Under Active development. Not fully tested yet.

    Computes the Euler Characteristic Transform of an arbitrary Simplicial
    Complex. This is the most general, but least optimized which is great for
    small problems and a good start. If performance is a requirement, one of the
    other implemetations is most likely faster at the cost of less flexibility.

    Parameters
    ----------
    x : Tensor
        The point cloud of shape [BxN,D] where B is the number of point clouds,
        N is the number of points and D is the ambient dimension.
    simplices: Iterable
        Contains, as _ordered_ set of arguments, the index tensors for the
        simplicial complex in ascending order. See examples.
    v : Tensor
        The tensor of directions of shape [D,N], where D is the ambient
        dimension and N is the number of directions.
    radius : float
        Radius of the interval to discretize the ECT into. (Is irrelevant for
        this experiment.)
    resolution : int
        Number of steps to divide the lin interval into.
    scale : Tensor
        The multipicative factor for the sigmoid function.
    index: Tensor
        Tensor of integers batching the points in their respective batch.
        The index tensor is assumed to start at 0, otherwise fails.
    Returns
    -------
    Tensor
        The ECT of the point cloud of shape [B,N,R] where B is the number of
        point clouds (thus ECT's), N is the number of direction and R is the
        resolution.
    """

    # ecc.shape[0], index.max().item() + 1, ecc.shape[2],
    if index is not None:
        batch_len = int(index.max() + 1)
    else:
        batch_len = 1
        index = torch.zeros(size=(len(x),), dtype=torch.int32)

    # v is of shape [d, num_thetas]
    num_thetas = v.shape[1]

    out_shape = (resolution, batch_len, num_thetas)

    # Node heights have shape [num_points, num_directions]
    nh = x @ v
    lin = torch.linspace(-radius, radius, resolution).view(-1, 1, 1)
    ecc = ect_fn(scale * torch.sub(lin, nh))

    output = torch.zeros(
        size=out_shape,
        device=nh.device,
    )

    output.index_add_(1, index, ecc)

    # For the calculation of the edges, loop over the simplex tensors.
    # Each index tensor is assumed to be of shape [d,num_simplices],
    # where d is the dimension of the simplex.
    for i, simplex in enumerate(simplices):
        # Simplex heights.
        sh, _ = nh[simplex].max(dim=0)

        # Compute which batch an edge belongs to. We take the first index of the
        # edge (or faces) and do a lookup on the batch index of that node in the
        # batch indices of the nodes.
        index_simplex = index[simplex[0]]

        # Calculate the ECC of the simplices.
        secc = (-1) ** (i + 1) * ect_fn(scale * torch.sub(lin, sh))

        # Add the ECC of the simplices to the running total.
        output.index_add_(1, index_simplex, secc)

    # Returns the ect as [batch_len, num_thetas, resolution]
    return output.movedim(0, 1).movedim(-1, -2)


def compute_ect_point_cloud(
    x: Tensor,
    v: Tensor,
    radius: float,
    resolution: int,
    scale: float,
) -> Tensor:
    """
    Computes the ECT of a point cloud. Assumes the point clouds are batched and
    of the same cardinality. The shape is assumed to be of the form [B,N,D], the
    first dimension forms the index vector.

    Parameters
    ----------
    x : Tensor
        The point cloud of shape [B,N,D] where B is the number of point clouds,
        N is the number of points and D is the ambient dimension.
    v : Tensor
        The tensor of directions of shape [D,N], where D is the ambient
        dimension and N is the number of directions.
    radius : float
        Radius of the interval to discretize the ECT into. (Is irrelevant for
        this experiment.)
    resolution : int
        Number of steps to divide the lin interval into.
    scale : Tensor
        The multipicative factor for the sigmoid function.

    Returns
    -------
    Tensor
        The ECT of the point cloud of shape [B,N,R] where B is the number of
        point clouds (thus ECT's), N is the number of direction and R is the
        resolution.
    """
    lin = torch.linspace(
        start=-radius, end=radius, steps=resolution, device=x.device
    ).view(-1, 1, 1)
    nh = (x @ v).unsqueeze(1)
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    ect = torch.sum(ecc, dim=2)
    return ect

def compute_ect_edges(
    x: Tensor,
    edge_index: Tensor,
    v: Tensor,
    radius: float,
    resolution: int,
    scale: float,
    index: Tensor | None = None,
):
    """Computes the Euler Characteristic Transform of a batch of graphs.

    Parameters
    ----------
    batch : Batch
        A batch of data containing the node coordinates, the edges and batch
        index.
    v: torch.FloatTensor
        The direction vector that contains the directions.
    lin: torch.FloatTensor
        The discretization of the interval [-1,1] each node height falls in this
        range due to rescaling in normalizing the data.
    """

    # ecc.shape[0], index.max().item() + 1, ecc.shape[2],
    if index is not None:
        batch_len = int(index.max() + 1)
    else:
        batch_len = 1
        index = torch.zeros(size=(len(x),), dtype=torch.int32)

    # v is of shape [d, num_thetas]
    num_thetas = v.shape[1]

    out_shape = (resolution, batch_len, num_thetas)

    # Node heights have shape [num_points, num_directions]
    nh = x @ v
    lin = torch.linspace(-radius, radius, resolution).view(-1, 1, 1)
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    output = torch.zeros(
        size=out_shape,
        device=nh.device,
    )

    output.index_add_(1, index, ecc)

    # For the calculation of the edges, loop over the simplex tensors.
    # Each index tensor is assumed to be of shape [d,num_simplices],
    # where d is the dimension of the simplex.

    # Edges heights.
    sh, _ = nh[edge_index].max(dim=0)

    # Compute which batch an edge belongs to. We take the first index of the
    # edge (or faces) and do a lookup on the batch index of that node in the
    # batch indices of the nodes.
    index_simplex = index[edge_index[0]]

    # Calculate the ECC of the simplices.
    secc = (-1) * torch.nn.functional.sigmoid(scale * torch.sub(lin, sh))

    # Add the ECC of the simplices to the running total.
    output.index_add_(1, index_simplex, secc)

    # Returns the ect as [batch_len, num_thetas, resolution]
    return output.movedim(0, 1).movedim(-1, -2)


def compute_ect_mesh(
    x: Tensor,
    edge_index: Tensor,
    face_index: Tensor,
    v: Tensor,
    radius: float,
    resolution: int,
    scale: float,
    index: Tensor | None = None,
):
    """Computes the Euler Characteristic Transform of a batch of graphs.

    Parameters
    ----------
    batch : Batch
        A batch of data containing the node coordinates, the edges and batch
        index.
    v: torch.FloatTensor
        The direction vector that contains the directions.
    lin: torch.FloatTensor
        The discretization of the interval [-1,1] each node height falls in this
        range due to rescaling in normalizing the data.
    """

    # ecc.shape[0], index.max().item() + 1, ecc.shape[2],
    if index is not None:
        batch_len = int(index.max() + 1)
    else:
        batch_len = 1
        index = torch.zeros(size=(len(x),), dtype=torch.int32)

    # v is of shape [d, num_thetas]
    num_thetas = v.shape[1]

    out_shape = (resolution, batch_len, num_thetas)

    # Node heights have shape [num_points, num_directions]
    nh = x @ v
    lin = torch.linspace(-radius, radius, resolution).view(-1, 1, 1)
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    output = torch.zeros(
        size=out_shape,
        device=nh.device,
    )

    output.index_add_(1, index, ecc)

    # For the calculation of the edges, loop over the simplex tensors.
    # Each index tensor is assumed to be of shape [d,num_simplices],
    # where d is the dimension of the simplex.

    # Edges heights.
    eh, _ = nh[edge_index].max(dim=0)

    # Compute which batch an edge belongs to. We take the first index of the
    # edge (or faces) and do a lookup on the batch index of that node in the
    # batch indices of the nodes.
    index_simplex = index[edge_index[0]]

    # Calculate the ECC of the simplices.
    edges_ecc = (-1) * torch.nn.functional.sigmoid(scale * torch.sub(lin, eh))

    # Add the ECC of the simplices to the running total.
    output.index_add_(1, index_simplex, edges_ecc)

    # Faces heights.
    fh, _ = nh[face_index].max(dim=0)

    # Compute which batch an edge belongs to. We take the first index of the
    # edge (or faces) and do a lookup on the batch index of that node in the
    # batch indices of the nodes.
    index_simplex = index[face_index[0]]

    # Calculate the ECC of the simplices.
    faces_ecc = (-1) * torch.nn.functional.sigmoid(scale * torch.sub(lin, fh))

    # Add the ECC of the simplices to the running total.
    output.index_add_(1, index_simplex, faces_ecc)

    # Returns the ect as [batch_len, num_thetas, resolution]
    return output.movedim(0, 1).movedim(-1, -2)
