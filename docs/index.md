# Welcome

This is the official implementation for the **Differentiable Euler Characteristic
Transform**, a geometrical-topological method for shape classification. Our
implementation is fully optimized for hardware acceleration,
yielding a blazingly fast implementation for machine learning research with
[`PyTorch`](https://pytorch.org/).

## Installation

For the installation we require an up-to-date installation of PyTorch, either
with or without CUDA support. Our package can be installed with either 
pip or added as a git submodule.

Installation as a git submodule (from the root folder) and afterward as and (editable) package can be done as follows. 
```sh
git submodule add https://github.com/aidos-lab/dect.git
cd dect
pip install [-e] .
```

Or use pip to install the git repository.
   
```sh
pip install git+https://github.com/aidos-lab/dect.git
```

## Usage

For a demonstration of our new ECT computation, we provide the
`notebooks/01_example_usage.ipynb` file and the code therein provides an intuitive
example with explanation. The code is provided on an as-is basis; see
[LICENSE.md](https://github.com/aidos-lab/dect/blob/main/LICENSE.md) for more
information. You are cordially invited to both contribute and provide feedback.
Do not hesitate to contact us!

```python
import torch 
from dect.directions import generate_2d_directions 
from dect.ect import compute_ect
from dect.ect_fn import scaled_sigmoid 

# Added for visualization.
import matplotlib.pyplot as plt

# Basic dataset with three points,three edges and one face.
points_coordinates = torch.tensor([[0.5, 0.0], [-0.5, 0.0], [0.5, 0.5]])

# Generate a set of structured directions along the unit circle.
v = generate_2d_directions(num_thetas=64)

# Compute the ECT.
ect = compute_ect(
    points_coordinates, 
    v=v,
    radius=1,
    resolution=64,
    scale=500,
    ect_fn=scaled_sigmoid
)

# Visualize as an image.
plt.imshow(ect.detach().squeeze().numpy().T)
plt.show()
```

## Compute the ECT of point clouds with channels. 

It is often of interest to compute the ECT of point clouds with different categorical types, 
such as atom numbers in molecules. The `compute_ect_channels` function provides a method to 
compute the ECT per categorical type, resulting in a set of ECTs. 
We include an example below. 

```{python}
import torch

from dect.directions import generate_2d_directions
from dect.ect import compute_ect_channels

v = generate_2d_directions()
x = torch.rand(size=(10, 2))
batch = torch.repeat_interleave(torch.tensor([0, 1]), repeats=5)
z = torch.randint(low=0, high=3, size=(10,))

ect = compute_ect_channels(x, v, radius=1, resolution=64, scale=200, index=batch, channels=z)

ect.shape # Result is an ECT of shape [2,3,64,64]
```

