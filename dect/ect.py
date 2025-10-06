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

import os
import warnings
from typing import Callable, List, Optional, TypeAlias

import torch

from dect.ect_fn import indicator

Tensor: TypeAlias = torch.Tensor
"""@private"""


def normalize_ect(ect):
    """Returns the normalized ect, scaled to lie in the interval 0,1"""
    return ect / torch.amax(ect, dim=(-2, -3)).clamp_min(1e-12)


def compute_ect(
    x: Tensor,
    *simplices: Tensor,
    v: Tensor,
    radius: float,
    resolution: int,
    scale: float,
    index: Tensor | None = None,
    ect_fn: Callable[..., Tensor] = indicator,
) -> Tensor:
    """
    NOTE: Under Active development.

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

    # ensure that the scale is in the right device
    scale_tensor = torch.tensor([scale], device=x.device)

    if index is not None:
        batch_len = int(index.max() + 1)
    else:
        batch_len = 1
        index = torch.zeros(
            size=(len(x),),
            dtype=torch.long,
            device=x.device,
        )

    # v is of shape [d, num_thetas]
    num_thetas = v.shape[1]

    out_shape = (resolution, batch_len, num_thetas)

    # Node heights have shape [num_points, num_directions]
    nh = x @ v
    lin = torch.linspace(-radius, radius, resolution, device=x.device).view(-1, 1, 1)
    ecc = ect_fn(scale_tensor * torch.sub(lin, nh))

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
        secc = (-1) ** (i + 1) * ect_fn(scale_tensor * torch.sub(lin, sh))

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
    normalize: bool = False,
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
    normalize : bool
        Rescale the pixel values to the interval [0,1]. Default is False.

    Returns
    -------
    Tensor
        The ECT of the point cloud of shape [B,N,R] where B is the number of
        point clouds (thus ECT's), N is the number of direction and R is the
        resolution.
    """

    # ensure that the scale is in the right device
    scale_tensor = torch.tensor([scale], device=x.device)

    lin = torch.linspace(
        start=-radius, end=radius, steps=resolution, device=x.device
    ).view(-1, 1, 1)
    nh = (x @ v).unsqueeze(1)
    nh[nh.isnan()] = torch.inf
    nh[nh.isinf()] = torch.inf
    ecc = torch.nn.functional.sigmoid(scale_tensor * torch.sub(lin, nh))
    ect = torch.sum(ecc, dim=2)
    if normalize:
        ect = normalize_ect(ect)

    return ect


def compute_ect_points(
    x: Tensor,
    v: Tensor,
    radius: float,
    resolution: int,
    scale: float,
    index: Tensor | None = None,
):
    """
    Computes the Euler Characteristic Transform of a batch of point
    clouds in `torch-geometric` format.

    Parameters
    ----------
    x : Tensor
        The point cloud of shape [B,N,D] where B is the number of point clouds,
        N is the number of points and D is the ambient dimension.
    v : Tensor
        The tensor of directions of shape [D,N], where D is the ambient
        dimension and N is the number of directions.
    radius : float
        Radius of the interval to discretize the ECT into.
    resolution : int
        Number of steps to divide the lin interval into.
    scale : Tensor
        The multiplicative factor for the sigmoid function.
    index: Tensor
        Tensor of integers batching the points in their respective batch.
        The index tensor is assumed to start at 0.
    """

    # ensure that the scale is in the right device
    scale_tensor = torch.tensor([scale], device=x.device)

    if index is not None:
        batch_len = int(index.max() + 1)
    else:
        batch_len = 1
        index = torch.zeros(
            size=(len(x),),
            dtype=torch.int32,
            device=x.device,
        )

    # v is of shape [ambient_dimension, num_thetas]
    num_thetas = v.shape[1]

    out_shape = (resolution, batch_len, num_thetas)

    # Node heights have shape [num_points, num_directions]
    nh = x @ v
    lin = torch.linspace(-radius, radius, resolution, device=x.device).view(-1, 1, 1)
    ecc = torch.nn.functional.sigmoid(scale_tensor * torch.sub(lin, nh))
    output = torch.zeros(
        size=out_shape,
        device=nh.device,
    )

    output.index_add_(1, index, ecc)

    # Returns the ect as [batch_len, num_thetas, resolution]
    return output.movedim(0, 1)


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
    x : Tensor
        The point cloud of shape [B,N,D] where B is the number of point clouds,
        N is the number of points and D is the ambient dimension.
    edge_index : Tensor
        The edge index tensor in torch geometric format, has to have shape
        [2,num_edges]. Be careful when using undirected graphs, since torch
        geometric views undirected graphs as 2 directed edges, leading to
        double counts.
    v : Tensor
        The tensor of directions of shape [D,N], where D is the ambient
        dimension and N is the number of directions.
    radius : float
        Radius of the interval to discretize the ECT into.
    resolution : int
        Number of steps to divide the lin interval into.
    scale : Tensor
        The multiplicative factor for the sigmoid function.
    index: Tensor
        Tensor of integers batching the points in their respective batch.
        The index tensor is assumed to start at 0.
    """

    # ensure that the scale is in the right device
    scale_tensor = torch.tensor([scale], device=x.device)

    if index is not None:
        batch_len = int(index.max() + 1)
    else:
        batch_len = 1
        index = torch.zeros(
            size=(len(x),),
            dtype=torch.int32,
            device=x.device,
        )

    # v is of shape [ambient_dimension, num_thetas]
    num_thetas = v.shape[1]

    out_shape = (resolution, batch_len, num_thetas)

    # Node heights have shape [num_points, num_directions]
    nh = x @ v
    lin = torch.linspace(-radius, radius, resolution, device=x.device).view(-1, 1, 1)
    ecc = torch.nn.functional.sigmoid(scale_tensor * torch.sub(lin, nh))
    output = torch.zeros(
        size=out_shape,
        device=x.device,
    )

    output.index_add_(1, index, ecc)

    # Edges heights.
    sh, _ = nh[edge_index].max(dim=0)

    # Compute which batch an edge belongs to. We take the first index of the
    # edge (or faces) and do a lookup on the batch index of that node in the
    # batch indices of the nodes.
    index_simplex = index[edge_index[0]]

    # Calculate the ECC of the edges.
    secc = (-1) * torch.nn.functional.sigmoid(scale_tensor * torch.sub(lin, sh))

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
    x : Tensor
        The point cloud of shape [B,N,D] where B is the number of point clouds,
        N is the number of points and D is the ambient dimension.
    edge_index : Tensor
        The edge index tensor in torch geometric format, has to have shape
        [2,num_edges]. Be careful when using undirected graphs, since torch
        geometric views undirected graphs as 2 directed edges, leading to
        double counts.
    face_index : Tensor
        The face index tensor of shape [3,num_faces]. Each column is a face
        where a face is a triple of indices referencing to the rows of the
        x tensor with coordinates.
    v : Tensor
        The tensor of directions of shape [D,N], where D is the ambient
        dimension and N is the number of directions.
    radius : float
        Radius of the interval to discretize the ECT into.
    resolution : int
        Number of steps to divide the lin interval into.
    scale : Tensor
        The multipicative factor for the sigmoid function.
    index: Tensor
        Tensor of integers batching the points in their respective batch.
        The index tensor is assumed to start at 0.
    """

    # ensure that the scale is in the right device
    scale_tensor = torch.tensor([scale], device=x.device)

    if index is not None:
        batch_len = int(index.max() + 1)
    else:
        batch_len = 1
        index = torch.zeros(size=(len(x),), dtype=torch.long, device=x.device)

    # v is of shape [d, num_thetas]
    num_thetas = v.shape[1]

    out_shape = (resolution, batch_len, num_thetas)

    # Node heights have shape [num_points, num_directions]
    nh = x @ v
    lin = torch.linspace(-radius, radius, resolution, device=x.device).view(-1, 1, 1)
    ecc = torch.nn.functional.sigmoid(scale_tensor * torch.sub(lin, nh))

    output = torch.zeros(
        size=out_shape,
        device=nh.device,
    )

    _ = output.index_add_(1, index, ecc)

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
    edges_ecc = (-1) * torch.nn.functional.sigmoid(scale_tensor * torch.sub(lin, eh))

    # Add the ECC of the simplices to the running total.
    _ = output.index_add_(1, index_simplex, edges_ecc)

    # Faces heights.
    fh, _ = nh[face_index].max(dim=0)

    # Compute which batch an edge belongs to. We take the first index of the
    # edge (or faces) and do a lookup on the batch index of that node in the
    # batch indices of the nodes.
    index_simplex = index[face_index[0]]

    # Calculate the ECC of the simplices.
    faces_ecc = torch.nn.functional.sigmoid(scale_tensor * torch.sub(lin, fh))

    # Add the ECC of the simplices to the running total.
    _ = output.index_add_(1, index_simplex, faces_ecc)

    # Returns the ect as [batch_len, num_thetas, resolution]
    return output.movedim(0, 1).movedim(-1, -2)


def compute_ect_channels(
    x: Tensor,
    v: Tensor,
    radius: float,
    resolution: int,
    scale: float,
    channels: Tensor,
    index: Tensor | None = None,
    max_channels: int | None = None,
    normalize: bool = False,
):
    """
    Allows for channels within the point cloud to separated in different
    ECT's.

    Input is a ragged-batch point set of size (N, 3) with an additional vector
    `channels` containing a channel id per point and a vector `index` with the
    batch id per point. Supports either per-batch directions `v` of shape
    (B, k, 3) or shared directions of shape (3, k).

    Returns a tensor of shape (B, k, resolution, C) where C is the number of
    channels (== max_channels).
    """

    # Normalize shared direction shape: accept (3,k) or (k,3); keep as (3,k)
    if v.dim() == 2:
        if v.shape[-1] == 3 and v.shape[0] != 3:
            # provided as (k,3) -> transpose to (3,k)
            v = v.transpose(0, 1).contiguous()
        elif v.shape[0] == 3:
            # already (3,k)
            v = v.contiguous()
        else:
            raise ValueError(f"v must be (3,k) or (k,3) when 2D; got {tuple(v.shape)}")

    # Ensure types/devices
    x = x.to(dtype=torch.get_default_dtype())
    scale = torch.as_tensor(scale, device=x.device, dtype=x.dtype)

    # Infer B and set default index if needed
    if index is None or index.numel() == 0:
        index = torch.zeros((x.size(0),), dtype=torch.long, device=x.device)
    else:
        index = index.to(dtype=torch.long, device=x.device)
    B = int(index.max().item()) + 1

    # Channels and max_channels
    channels = channels.to(dtype=torch.long, device=x.device)
    if max_channels is None:
        max_channels = int(channels.max().item()) + 1

    # Determine number of directions k and compute per-point heights nh: (N, k)
    if v.dim() == 2:
        # shared directions: v is (3, k)
        assert v.size(0) == x.size(1), "v must be (D, k) with D == x.size(1)"
        k = v.size(1)
        nh = x @ v  # (N, k)
    elif v.dim() == 3:
        # per-batch directions: v is (B, k, 3)
        assert v.size(0) == B and v.size(-1) == x.size(
            1
        ), "v must be (B, k, D) with D == x.size(1) and B matching index"
        k = v.size(1)
        # gather directions for each point's batch id, then compute dot
        Vp = v[index]  # (N, k, 3)
        nh = (x.unsqueeze(1) * Vp).sum(dim=-1)  # (N, k)
    else:
        raise ValueError("v must have shape (D, k) or (B, k, D)")

    # Discretize thresholds and compute ECC per point & direction: (R, N, k)
    lin = torch.linspace(
        -radius, radius, resolution, device=x.device, dtype=x.dtype
    ).view(resolution, 1, 1)
    ecc = torch.sigmoid(scale * (lin - nh.unsqueeze(0)))  # (R, N, k)

    # Aggregate by (batch, channel) via a flattened index
    idx_bc = (index * max_channels + channels).to(dtype=torch.long)  # (N,)
    out = x.new_zeros((resolution, B * max_channels, k))  # (R, B*C, k)
    out.index_add_(1, idx_bc, ecc)  # sum over points

    # Reshape to (B, C, R, k) then permute to (B, k, R, C)
    ect = (
        out.view(resolution, B, max_channels, k).permute(1, 3, 0, 2).contiguous()
    )  # (B, k, R, C)

    if normalize:
        # normalize per (B,k) across (R,C)
        denom = torch.amax(ect, dim=(-1, -2), keepdim=True).clamp_min(1e-12)
        ect = ect / denom

    return ect


def compute_ect_hypergraph(
    x: Tensor,
    hyperedges: List[Tensor],
    v: Tensor,
    radius: float,
    resolution: int,
    scale: float,
    index: Tensor | None = None,
    normalize: bool = False,
) -> Tensor:
    """
    ECT for a hypergraph:
      - Nodes (0-simplices) contribute with +1
      - A hyperedge with m nodes is a (m-1)-simplex and contributes with (-1)^(m-1)

    Returns: Tensor of shape [B, T, R]
      B = number of batches
      T = number of directions
      R = resolution
    """
    # Normalize direction shape to (D, T)
    if v.dim() != 2:
        raise ValueError("v must be 2D with shape (D, T) or (T, D)")
    if v.shape[0] != x.shape[1] and v.shape[1] == x.shape[1]:
        v = v.transpose(0, 1).contiguous()
    if v.shape[0] != x.shape[1]:
        raise ValueError(
            f"v has incompatible shape {tuple(v.shape)} for x with D={x.shape[1]}"
        )

    # Types/devices
    x = x.to(dtype=torch.get_default_dtype())
    device = x.device
    scale_t = torch.as_tensor(scale, device=device, dtype=x.dtype)

    # Batch index handling (must be Long for index_add_)
    if index is None:
        index = torch.zeros((x.size(0),), dtype=torch.long, device=device)
        B = 1
    else:
        index = index.to(dtype=torch.long, device=device)
        B = int(index.max().item()) + 1

    T = v.shape[1]
    R = int(resolution)

    # Threshold grid and node heights
    lin = torch.linspace(-radius, radius, R, device=device, dtype=x.dtype).view(R, 1, 1)
    nh = x @ v  # (N, T)

    # Accumulator (R, B, T)
    out = x.new_zeros((R, B, T))

    # 0-simplices (nodes): +1
    ecc_nodes = torch.sigmoid(scale_t * (lin - nh))  # (R, N, T)
    out.index_add_(1, index, ecc_nodes)

    # Hyperedges: treat each as a simplex of dimension (m-1) with sign (-1)**(m-1)
    for he in hyperedges:
        if he.numel() == 0:
            continue
        he = he.to(device=device, dtype=torch.long)

        # Max-height across constituent nodes for each direction
        he_height, _ = nh[he].max(dim=0)  # (T,)
        # Batch id: take the first node’s batch
        b = index[he[0]]  # scalar Long tensor

        m = int(he.numel())
        sign = -1 if (m % 2 == 0) else 1  # (-1)**(m-1)

        ecc_he = sign * torch.sigmoid(
            scale_t * (lin - he_height.view(1, 1, -1))
        )  # (R, 1, T)
        out.index_add_(1, b.view(1), ecc_he)

    # (R, B, T) -> (B, T, R)
    ect = out.movedim(0, 2)

    if normalize:
        denom = torch.amax(ect, dim=-1, keepdim=True).clamp_min(1e-12)
        ect = ect / denom

    return ect


def compute_ect_hypergraph_channels(
    x: Tensor,
    hyperedges: List[Tensor],
    v: Tensor,
    radius: float,
    resolution: int,
    scale: float,
    channels: Tensor,
    hyperedge_channels: Optional[List[Tensor]] = None,
    index: Tensor | None = None,
    max_channels: int | None = None,
    normalize: bool = False,
) -> Tensor:
    """
    Channel-aware ECT for hypergraphs.

    Nodes contribute with +1. A hyperedge with m nodes is treated as a
    simplex of dimension (m-1) and contributes with sign (-1)**(m-1).

    Returns a tensor of shape (B, T, R, C), where
      B = batches, T = directions, R = resolution, C = channels.
    """
    # Dtypes/devices
    x = x.to(dtype=torch.get_default_dtype())
    device = x.device
    scale_t = torch.as_tensor(scale, device=device, dtype=x.dtype)

    # Batch index handling (must be Long for index_add_)
    if index is None or index.numel() == 0:
        index = torch.zeros((x.size(0),), dtype=torch.long, device=device)
    else:
        index = index.to(dtype=torch.long, device=device)
    B = int(index.max().item()) + 1

    # Channels
    channels = channels.to(dtype=torch.long, device=device)
    if max_channels is None:
        max_channels = int(channels.max().item()) + 1

    # Handle direction shapes
    D = x.size(1)
    if v.dim() == 2:
        # accept (D, T) or (T, D)
        if v.shape[0] != D and v.shape[1] == D:
            v = v.transpose(0, 1).contiguous()
        if v.shape[0] != D:
            raise ValueError(
                f"v has incompatible shape {tuple(v.shape)} for x with D={D}"
            )
        T = v.shape[1]
        nh = x @ v  # (N, T)
    elif v.dim() == 3:
        # accept (B, T, D) or (B, D, T)
        if v.shape[0] != B:
            raise ValueError(f"v batch dimension {v.shape[0]} must match B={B}")
        if v.shape[-1] == D:
            # (B, T, D)
            T = v.shape[1]
            Vp = v[index]  # (N, T, D)
        elif v.shape[1] == D:
            # (B, D, T) -> (B, T, D)
            v = v.transpose(1, 2).contiguous()
            T = v.shape[1]
            Vp = v[index]  # (N, T, D)
        else:
            raise ValueError(
                f"v has incompatible shape {tuple(v.shape)} for x with D={D}"
            )
        nh = (x.unsqueeze(1) * Vp).sum(dim=-1)  # (N, T)
    else:
        raise ValueError("v must have shape (D,T), (T,D), (B,T,D), or (B,D,T)")

    R = int(resolution)

    # Threshold grid
    lin = torch.linspace(-radius, radius, R, device=device, dtype=x.dtype).view(R, 1, 1)

    # NODE contributions (R, N, T)
    ecc_nodes = torch.sigmoid(scale_t * (lin - nh))

    # Aggregate nodes by flattened (batch, channel)
    idx_bc = (index * max_channels + channels).to(dtype=torch.long)
    out = x.new_zeros((R, B * max_channels, T))  # (R, B*C, T)
    out.index_add_(1, idx_bc, ecc_nodes)

    # HYPEREDGE contributions
    # Iterate per hyperedge; each is a 1D tensor of node indices
    for i, he in enumerate(hyperedges):
        he = he.to(device=device, dtype=torch.long)
        m = int(he.numel())
        if m == 0:
            continue
        # Max-height across member nodes per direction
        he_h, _ = nh[he].max(dim=0)  # (T,)
        # Batch from first node
        b = index[he[0]]  # scalar Long
        # Channel: provided or inherit from first node
        if hyperedge_channels is not None and i < len(hyperedge_channels):
            ch_i = hyperedge_channels[i]
            if isinstance(ch_i, torch.Tensor):
                ch = ch_i.to(device=device, dtype=torch.long).view(-1)[0]
            else:
                ch = torch.as_tensor(int(ch_i), device=device, dtype=torch.long)
        else:
            ch = channels[he[0]]  # scalar Long

        # Alternating sign: (-1)**(m-1)
        sign = -1 if (m % 2 == 0) else 1
        ecc_he = sign * torch.sigmoid(
            scale_t * (lin - he_h.view(1, 1, -1))
        )  # (R, 1, T)

        # Flattened (batch, channel) index
        idx = (b * max_channels + ch).view(1)  # (1,)
        out.index_add_(1, idx, ecc_he)

    # Reshape to (B, T, R, C)
    ect = out.view(R, B, max_channels, T).permute(1, 3, 0, 2).contiguous()

    if normalize:
        denom = torch.amax(ect, dim=(-1, -2), keepdim=True).clamp_min(1e-12)
        ect = ect / denom

    return ect
