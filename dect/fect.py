"""
Implementation of the standard ECT for large graphs. Total size of the graph
depends on the memory of the GPU and the number of nodes and / or edges. - For
nodes only, it handles approx 1M nodes at a 1024 resolution. - For graphs,
current testing shows approx 100k nodes and 20k-50k edges.

Both compute at decent speed, inference times were approx 1.5 seconds. The
current code is not optimized and relies heavily on torch.scatter perform the 2d
bincount. It is therefore expected that a custom triton/cuda kernel will
significantly reduce the compute time.  

Type casting to the right types in torch is non-ideal, leading to a non-memory
optimized algorithm with much higher memory needs than needed. Case in point,
torch scatter needs int64 for the index vector, while in practice uint32
suffices, a factor of four. The index vector is of large size [2,1e6,1024] for
edges and [simplex_dim,num_nodes,num_directions] in general so this causes
unnecessary OOM errors.

Important, no guards for overflows (it happens silently) and no
differentiability. 
"""

import torch

# ---------------------------------------------------------------------------- #
#                                   Functions                                  #
# ---------------------------------------------------------------------------- #


def bincount(idx, resolution):
    """Calculates the histogram in resolution bins."""
    x = torch.zeros(size=(resolution, resolution), dtype=torch.int16)
    return x.scatter_(0, idx.to(torch.int64), 1, reduce="add")


def fast_ect(x, v):
    """Fast ECT for point clouds."""
    resolution = v.shape[1]
    nh = ((torch.matmul(x, v) + 1) * (resolution // 2)).to(torch.uint16)
    return bincount(nh, resolution)


def fast_ect_edges(x, ei, v):
    """Fast ECT for edges."""
    resolution = v.shape[1]
    nh = ((torch.matmul(x, v) + 1) * (resolution // 2)).to(torch.int32)
    eh = nh[ei].max(axis=0)[0]
    return bincount(nh, resolution), bincount(eh, resolution)
