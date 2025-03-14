from typing import Literal

import geotorch
import torch
from torch import nn

def compute_wecc(
    nh: torch.FloatTensor,
    index: torch.LongTensor,
    lin: torch.FloatTensor,
    weight: torch.FloatTensor,
    scale: float = 500,
):
    """Computes the Weighted Euler Characteristic Curve.

    Parameters
    ----------
    nh : torch.FloatTensor
        The node heights, computed as the inner product of the node coordinates
        x and the direction vector v.
    index: torch.LongTensor
        The index that indicates to which pointcloud a node height belongs. For
        the node heights it is the same as the batch index, for the higher order
        simplices it will have to be recomputed.
    lin: torch.FloatTensor
        The discretization of the interval [-1,1] each node height falls in this
        range due to rescaling in normalizing the data.
    weight: torch.FloatTensor
        The weight of the node, edge or face. It is the maximum of the node
        weights for the edges and faces.
    scale: torch.FloatTensor
        A single number that scales the sigmoid function by multiplying the
        sigmoid with the scale. With high (100>) values, the ect will resemble a
        discrete ECT and with lower values it will smooth the ECT.
    """
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh)) * weight.view(
        1, -1, 1
    )
    ecc = ecc.movedim(0, 2).movedim(0, 1)
    return segment_add_coo(ecc, index)


def compute_wect(
    batch: Batch,
    v: torch.FloatTensor,
    lin: torch.FloatTensor,
    wect_type: Literal["points"] | Literal["edges"] | Literal["faces"],
):
    """
    Computes the Weighted Euler Characteristic Transform of a batch of point
    clouds.

    Parameters
    ----------
    batch : Batch
        A batch of data containing the node coordinates, batch index,
        edge_index, face, and node weights.
    v: torch.FloatTensor
        The direction vector that contains the directions.
    lin: torch.FloatTensor
        The discretization of the interval [-1,1] each node height falls in this
        range due to rescaling in normalizing the data.
    wect_type: str
        The type of WECT to compute. Can be "points", "edges", or "faces".
    """

    nh = batch.x @ v
    if wect_type in ["edges", "faces"]:
        edge_weights, _ = batch.node_weights[batch.edge_index].max(axis=0)
        eh, _ = nh[batch.edge_index].min(dim=0)
    if wect_type == "faces":
        face_weights, _ = batch.node_weights[batch.face].max(axis=0)
        fh, _ = nh[batch.face].min(dim=0)

    if wect_type == "points":
        return compute_wecc(nh, batch.batch, lin, batch.node_weights)
    if wect_type == "edges":
        # noinspection PyUnboundLocalVariable
        return compute_wecc(
            nh, batch.batch, lin, batch.node_weights
        ) - compute_wecc(
            eh, batch.batch[batch.edge_index[0]], lin, edge_weights
        )
    if wect_type == "faces":
        # noinspection PyUnboundLocalVariable
        return (
            compute_wecc(nh, batch.batch, lin, batch.node_weights)
            - compute_wecc(
                eh, batch.batch[batch.edge_index[0]], lin, edge_weights
            )
            + compute_wecc(fh, batch.batch[batch.face[0]], lin, face_weights)
        )
    raise ValueError(f"Invalid wect_type: {wect_type}")



