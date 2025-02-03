"""
Implementation of the ECT with learnable parameters.
TODO: Needs implementation and refactoring.

"""

from typing import TypeAlias, Literal
from dataclasses import dataclass

import torch
from torch import nn


Tensor: TypeAlias = torch.Tensor
"""@private"""


@dataclass
class EctBatch:
    x: Tensor | None = None
    ect: Tensor | None = None


class EctConfig:
    """
    Config for initializing an ect layer.
    """

    num_thetas: int
    resolution: int
    r: float
    scale: float
    ect_type: Literal["points"]
    ambient_dimension: int
    normalized: bool
    seed: int


# ---------------------------------------------------------------------------- #
#                               To be depreciated                              #
# ---------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ECTConfig:
    """
    Configuration of the ECT Layer.

    Parameters
    ----------
    bump_steps : int
        The number of steps to discretize the ECT into.
    radius : float
        The radius of the circle the directions lie on. Usually this is a bit
        larger than the objects we wish to compute the ECT for, which in most
        cases have radius 1. For now it defaults to 1 as well.
    ect_type : str
        The type of ECT we wish to compute. Can be "points" for point clouds,
        "edges" for graphs or "faces" for meshes.
    normalized: bool
        Whether or not to normalize the ECT. Only work with ect_type set to
        points and normalized the ECT to the interval [0,1].
    fixed: bool
        Option to keep the directions fixed or not. In case the directions are
        learnable, we can use backpropagation to optimize over a set of
        directions. See notebooks for examples.
    """

    bump_steps: int = 32
    radius: float = 1.1
    ect_type: str = "points"
    normalized: bool = False
    fixed: bool = True


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
        [3,num_faces] or [4, num_faces], depending on the type of complex
        (simplicial or cubical).
    node_weights: torch.FloatTensor
        Optional weights for the nodes in the complex. The shape has to be
        [num_nodes,].
    """

    x: torch.FloatTensor
    batch: torch.LongTensor
    edge_index: torch.LongTensor | None = None
    face: torch.LongTensor | None = None
    node_weights: torch.FloatTensor | None = None


def compute_ecc(
    nh: torch.FloatTensor,
    index: torch.LongTensor,
    lin: torch.FloatTensor,
    scale: float = 100,
) -> torch.FloatTensor:
    """Computes the Euler Characteristic Curve.

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


def compute_ect_points(batch: Batch, v: torch.FloatTensor, lin: torch.FloatTensor):
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
    """
    nh = batch.x @ v
    return compute_ecc(nh, batch.batch, lin)


def compute_ect_edges(batch: Batch, v: torch.FloatTensor, lin: torch.FloatTensor):
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
    # Compute the node heigths
    nh = batch.x @ v

    # Perform a lookup with the edge indices on node heights, this replaces the
    # node index with its node height and then compute the maximum over the
    # columns to compute the edge height.
    eh, _ = nh[batch.edge_index].max(dim=0)

    # Compute which batch an edge belongs to. We take the first index of the
    # edge (or faces) and do a lookup on the batch index of that node in the
    # batch indices of the nodes.
    batch_index_nodes = batch.batch
    batch_index_edges = batch.batch[batch.edge_index[0]]

    return compute_ecc(nh, batch_index_nodes, lin) - compute_ecc(
        eh, batch_index_edges, lin
    )


def compute_ect_faces(batch: Batch, v: torch.FloatTensor, lin: torch.FloatTensor):
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
    """
    # Compute the node heigths
    nh = batch.x @ v

    # Perform a lookup with the edge indices on node heights, this replaces the
    # node index with its node height and then compute the maximum over the
    # columns to compute the edge height.
    eh, _ = nh[batch.edge_index].max(dim=0)

    # Do the same thing for the faces.
    fh, _ = nh[batch.face].max(dim=0)

    # Compute which batch an edge belongs to. We take the first index of the
    # edge (or faces) and do a lookup on the batch index of that node in the
    # batch indices of the nodes.
    batch_index_nodes = batch.batch
    batch_index_edges = batch.batch[batch.edge_index[0]]
    batch_index_faces = batch.batch[batch.face[0]]

    return (
        compute_ecc(nh, batch_index_nodes, lin)
        - compute_ecc(eh, batch_index_edges, lin)
        + compute_ecc(fh, batch_index_faces, lin)
    )


def normalize(ect):
    """Returns the normalized ect, scaled to lie in the interval 0,1"""
    return ect / torch.amax(ect, dim=(2, 3)).unsqueeze(2).unsqueeze(2)


class ECTLayer(nn.Module):
    """Machine learning layer for computing the ECT.

    Parameters
    ----------
    v: torch.FloatTensor
        The direction vector that contains the directions. The shape of the
        tensor v is either [ndims, num_thetas] or [n_channels, ndims,
        num_thetas].
    config: ECTConfig
        The configuration config of the ECT layer.

    """

    def __init__(self, config: ECTConfig, v=None):
        super().__init__()
        self.config = config
        self.lin = nn.Parameter(
            torch.linspace(-config.radius, config.radius, config.bump_steps).view(
                -1, 1, 1, 1
            ),
            requires_grad=False,
        )

        # If provided with one set of directions.
        # For backwards compatibility.
        if v.ndim == 2:
            v.unsqueeze(0)

        # The set of directions is added
        if config.fixed:
            self.v = nn.Parameter(v.movedim(-1, -2), requires_grad=False)
        else:
            # Movedim to make geotorch happy, me not happy.
            self.v = nn.Parameter(torch.zeros_like(v.movedim(-1, -2)))
            geotorch.constraints.sphere(self, "v", radius=config.radius)
            # Since geotorch randomizes the vector during initialization, we
            # assign the values after registering it with spherical constraints.
            # See Geotorch documentation for examples.
            self.v = v.movedim(-1, -2)

        if config.ect_type == "points":
            self.compute_ect = compute_ect_points
        elif config.ect_type == "edges":
            self.compute_ect = compute_ect_edges
        elif config.ect_type == "faces":
            self.compute_ect = compute_ect_faces

    def forward(self, batch: Batch):
        """Forward method for the ECT Layer.


        Parameters
        ----------
        batch : Batch
            A batch of data containing the node coordinates, edges, faces and
            batch index. It should follow the pytorch geometric conventions.

        Returns
        ----------
        ect: torch.FloatTensor
            Returns the ECT of each data object in the batch. If the layer is
            initialized with v of the shape [ndims,num_thetas], the returned ECT
            has shape [batch,num_thetas,bump_steps]. In case the layer is
            initialized with v of the form [n_channels, ndims, num_thetas] the
            returned ECT has the shape [batch,n_channels,num_thetas,bump_steps]
        """
        # Movedim for geotorch.
        ect = self.compute_ect(batch, self.v.movedim(-1, -2), self.lin)
        if self.config.normalized:
            return normalize(ect)
        return ect.squeeze()
