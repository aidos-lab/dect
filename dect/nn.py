"""
NOTE: Under construction.
TODO: Needs implementation and refactoring.

Implementation of the ECT with learnable parameters.
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import geotorch
import torch
from torch import nn

from dect.ect import compute_ect_edges, compute_ect_mesh, compute_ect_points
from dect.ect_fn import scaled_sigmoid

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
    TODO: Update Description. Outdated.
    Configuration of the ECT Layer.

    Parameters
    ----------
    resolution : int
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

    resolution: int = 32
    scale: float = 8
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

        # The set of directions is added
        if config.fixed:
            self.v = nn.Parameter(v.movedim(-1, -2), requires_grad=False)
        else:
            # Movedim to make geotorch happy, me not happy.
            self.v = nn.Parameter(torch.zeros_like(v.movedim(-1, -2)))
            geotorch.constraints.sphere(self, "v", radius=1.0)

            # Since geotorch randomizes the vector during initialization, we
            # assign the values after registering it with spherical constraints.
            # See Geotorch documentation for examples.
            self.v = v.movedim(-1, -2)

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
            has shape [batch,num_thetas,resolution]. In case the layer is
            initialized with v of the form [n_channels, ndims, num_thetas] the
            returned ECT has the shape [batch,n_channels,num_thetas,resolution]
        """

        # Movedim for geotorch.
        # NOTE: This needs improvement!

        if self.config.ect_type == "points":
            ect = compute_ect_points(
                batch.x,
                self.v.movedim(-1, -2),
                self.config.radius,
                self.config.resolution,
                self.config.scale,
                batch.batch,
            )
        elif self.config.ect_type == "edges":
            ect = compute_ect_edges(
                batch.x,
                batch.edge_index,
                self.v.movedim(-1, -2),
                self.config.radius,
                self.config.resolution,
                self.config.scale,
                batch.batch,
            )
        elif self.config.ect_type == "faces":
            ect = compute_ect_mesh(
                batch,
                batch.edge_index,
                batch.face,
                self.v.movedim(-1, -2),
                self.config.radius,
                self.config.resolution,
                self.config.scale,
                batch.batch,
            )

        if self.config.normalized:
            return normalize(ect)
        return ect.squeeze()
