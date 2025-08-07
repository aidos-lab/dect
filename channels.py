from typing import Callable, TypeAlias

import torch

from dect.ect_fn import indicator

Tensor: TypeAlias = torch.Tensor
"""@private"""


def compute_ect_channels(
    x: Tensor,
    v: Tensor,
    radius: float,
    resolution: int,
    scale: float,
    channels: Tensor,
    index: Tensor | None = None,
    max_channels: int | None = None,
):
    """
    Allows for channels within the point cloud to separated in different
    ECT's.

    Input is a point cloud of size (B*num_point_per_pc,num_features) with an addtional feature vector with the
    channel number for each point and the output is ECT for shape [B,num_channels,num_thetas,resolution]
    """

    # Ensure that the scale is in the right device
    scale = torch.tensor([scale], device=x.device)

    # Compute maximum channels.
    if max_channels is None:
        max_channels = int(channels.max())

    if index is not None:
        batch_len = int(index.max() + 1)
    else:
        batch_len = 1
        index = torch.zeros(
            size=(len(x),),
            dtype=torch.int32,
            device=x.device,
        )

    # Fix the index to interleave with the channel info.
    index = max_channels * index + channels

    # v is of shape [ambient_dimension, num_thetas]
    num_thetas = v.shape[1]

    out_shape = (resolution, batch_len, num_thetas)

    # Node heights have shape [num_points, num_directions]
    nh = x @ v
    lin = torch.linspace(-radius, radius, resolution, device=x.device).view(-1, 1, 1)
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    output = torch.zeros(
        size=out_shape,
        device=nh.device,
    )

    output.index_add_(1, index, ecc)

    # Returns the ect as [batch_len, num_thetas, resolution]
    return output.movedim(0, 1).reshape(-1, max_channels, num_thetas, resolution)
