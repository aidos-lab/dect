"""
Implementation of the ECT.
"""

from dataclasses import dataclass
import torch
from torch import nn
from torch_scatter import segment_add_coo


@dataclass(frozen=True)
class EctConfig:
    """
    Configuration of the ECT Layer.
    """

    num_thetas: int = 32
    bump_steps: int = 32
    radius: float = 1.1
    ect_type: str = "points"
    device: str = "cpu"
    num_features: int = 3
    normalized: bool = False


@dataclass()
class Batch:
    """Template of the required attributes for a data batch.

    Parameters
    ----------
    x : torch.FloatTensor
        The coordinates of the nodes in the simplical complex provided in the
        format [num_nodes,feature_size].
    batch: torch.LongTensor
        An index that indicates to which pointcloud a point belongs to, in
        principle automatically created by torch_geometric when initializing the
        batch.
    edge_index: torch.LongTensor
        The indices of the points that span an edge in the graph. Conforms to
        pytorch_geometric standards. Shape has to be of the form [2,num_edges].
    face:
        The indices of the points that span a face in the simplicial complex.
        Conforms to pytorch_geometric standards. Shape has to be of the form
        [3,num_faces].
    """

    x: torch.FloatTensor
    batch: torch.LongTensor
    edge_index: torch.LongTensor | None = None
    face: torch.LongTensor | None = None


def compute_ecc(
    nh: torch.FloatTensor,
    index: torch.LongTensor,
    lin: torch.FloatTensor,
    scale: float = 100,
) -> torch.FloatTensor:
    """Computes the Euler Characteristic curve.

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
    out: torch.FloatTensor
        The shape of the resulting tensor after summation. It has to be of the
        shape [num_discretization_steps, batch_size, num_thetas]
    scale: torch.FloatTensor
        A single number that scales the sigmoid function by multiplying the
        sigmoid with the scale. With high (100>) values, the ect will resemble a
        discrete ECT and with lower values it will smooth the ECT.
    """
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))

    # Due to (I believe) a bug in segment_add_coo, we have to first transpose
    # and then apply segment add. In the original code movedim was applied after
    # and that yields an bug in the backwards pass. Will have to be reported to
    # pytorch eventually.
    ecc = ecc.movedim(0, 2).movedim(0, 1)
    return segment_add_coo(ecc, index)


def compute_ect_points(
    batch: Batch, v: torch.FloatTensor, lin: torch.FloatTensor
):
    """Computes the Euler Characteristic Transform of a batch of point clouds.

    Parameters
    ----------
    batch : Batch
        A batch of data containing the node coordinates and batch index.
    v: torch.FloatTensor
        The direction vector that contains the directions.
    lin: torch.FloatTensor
        The discretization of the interval [-1,1] each node height falls in this
        range due to rescaling in normalizing the data.
    out: torch.FloatTensor
        The shape of the resulting tensor after summation. It has to be of the
        shape [num_discretization_steps, batch_size, num_thetas]
    """
    nh = batch.x @ v
    return compute_ecc(nh, batch.batch, lin)


def compute_ect_edges(
    data: Batch, v: torch.FloatTensor, lin: torch.FloatTensor
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
    out: torch.FloatTensor
        The shape of the resulting tensor after summation. It has to be of the
        shape [num_discretization_steps, batch_size, num_thetas]
    """
    # Compute the node heigths
    nh = data.x @ v

    # Perform a lookup with the edge indices on node heights, this replaces the
    # node index with its node height and then compute the maximum over the
    # columns to compute the edge height.
    eh, _ = nh[data.edge_index].max(dim=0)

    # Compute which batch an edge belongs to. We take the first index of the
    # edge (or faces) and do a lookup on the batch index of that node in the
    # batch indices of the nodes.
    batch_index_nodes = data.batch
    batch_index_edges = data.batch[data.edge_index[0]]

    return compute_ecc(nh, batch_index_nodes, lin) - compute_ecc(
        eh, batch_index_edges, lin
    )


def compute_ect_faces(
    data: Batch, v: torch.FloatTensor, lin: torch.FloatTensor
):
    """Computes the Euler Characteristic Transform of a batch of meshes.

    Parameters
    ----------
    batch : Batch
        A batch of data containing the node coordinates, edges, faces and batch
        index.
    v: torch.FloatTensor
        The direction vector that contains the directions.
    lin: torch.FloatTensor
        The discretization of the interval [-1,1] each node height falls in this
        range due to rescaling in normalizing the data.
    out: torch.FloatTensor
        The shape of the resulting tensor after summation. It has to be of the
        shape [num_discretization_steps, batch_size, num_thetas]
    """
    # Compute the node heigths
    nh = data.x @ v

    # Perform a lookup with the edge indices on node heights, this replaces the
    # node index with its node height and then compute the maximum over the
    # columns to compute the edge height.
    eh, _ = nh[data.edge_index].max(dim=0)

    # Do the same thing for the faces.
    fh, _ = nh[data.face].max(dim=0)

    # Compute which batch an edge belongs to. We take the first index of the
    # edge (or faces) and do a lookup on the batch index of that node in the
    # batch indices of the nodes.
    batch_index_nodes = data.batch
    batch_index_edges = data.batch[data.edge_index[0]]
    batch_index_faces = data.batch[data.face[0]]

    return (
        compute_ecc(nh, batch_index_nodes, lin)
        - compute_ecc(eh, batch_index_edges, lin)
        + compute_ecc(fh, batch_index_faces, lin)
    )


def normalize(ect):
    """Returns the normalized ect, scaled to lie in the interval 0,1"""
    return ect / torch.amax(ect, dim=(2, 3)).unsqueeze(2).unsqueeze(2)


class EctLayer(nn.Module):
    """Machine learning layer for computing the ECT."""

    def __init__(self, config: EctConfig, V=None):
        super().__init__()
        self.config = config
        self.lin = (
            torch.linspace(-config.radius, config.radius, config.bump_steps)
            .view(-1, 1, 1, 1)
            .to(config.device)
        )

        # If provided with one set of directions.
        # For backwards compatibility.
        if V.ndim == 2:
            V.unsqueeze(0)

        self.v = V

        if config.ect_type == "points":
            self.compute_ect = compute_ect_points
        elif config.ect_type == "edges":
            self.compute_ect = compute_ect_edges
        elif config.ect_type == "faces":
            self.compute_ect = compute_ect_faces

    def forward(self, batch: Batch):
        """Forward method for the ECT Layer."""
        ect = self.compute_ect(batch, self.v, self.lin)
        if self.config.normalized:
            return normalize(ect)
        return ect.squeeze()
