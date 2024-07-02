# DECT - Differentiable Euler Characteristic Transform
[![arXiv](https://img.shields.io/badge/arXiv-2310.07630-b31b1b.svg)](https://arxiv.org/abs/2310.07630) ![GitHub contributors](https://img.shields.io/github/contributors/aidos-lab/dect) ![GitHub](https://img.shields.io/github/license/aidos-lab/dect) [![Maintainability](https://api.codeclimate.com/v1/badges/82f86d7e2f0aae342055/maintainability)](https://codeclimate.com/github/aidos-lab/dect/maintainability)

This is the official implementation for the **Differentiable Euler Characteristic
Transform**, a geometrical-topological method for shape classification. Our
implementation is fully optimized for hardware acceleration,
yielding a blazingly fast implementation for machine learning research with
[`PyTorch`](https://pytorch.org/).

<img src="https://github.com/aidos-lab/dect/blob/main/figures/ect_animation.gif?raw=true" width="100%">

If you find our work useful, please consider using the following citation:

```bibtex
@inproceedings{Roell24a,
  title         = {Differentiable Euler Characteristic Transforms for Shape Classification},
  author        = {Ernst R{\"o}ell and Bastian Rieck},
  year          = 2024,
  booktitle     = {International Conference on Learning Representations},
  eprint        = {2310.07630},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  repository    = {https://github.com/aidos-lab/DECT},
  url           = {https://openreview.net/forum?id=MO632iPq3I},
}
```

## Installation

For the installation we require an up-to-date installation of PyTorch, either
with or without CUDA support. The DECT implementation also relies on the
`torch-scatter` package, highly optimized for grouped operations such summing 
of a vector given an index vector. 

- First install `torch` and `torch-scatter`.

   ```python
   pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu117
   pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+${CUDA}.html
   ```

- Then install our package in either one of two ways:
   - As a git submodule 

   ```sh
   git submodule add https://github.com/aidos-lab/dect.git
   ```

   - Or as a pip installable package. 
   
   ```sh
   pip install git+https://github.com/aidos-lab/dect.git
   ```

## Usage

For example usage, we provide the `notebooks/example.ipynb` file and the code therein reproduces the 
ECT of the gif of this readme. 
The code is provided on an as-is basis; see [LICENSE.md](https://github.com/aidos-lab/dect/blob/main/LICENSE.md) for more information.
You are cordially invited to both contribute and provide feedback. Do not hesitate to contact us!

```python
import torch
from torch_geometric.data import Data, Batch
from dect.ect import EctConfig, EctLayer
from dect.directions import generate_uniform_2d_directions


v = generate_uniform_2d_directions(num_thetas=64, device="cpu")

layer = EctLayer(EctConfig(), V=v)

points_coordinates = torch.tensor(
    [[0.5, 0.0], [-0.5, 0.0], [0.5, 0.5]], requires_grad=True
)

data = Data(x=points_coordinates)
batch = Batch.from_data_list([data])

ect = layer(batch)
```


## License

Our code is released under a BSD-3-Clause license. This license essentially
permits you to freely use our code as desired, integrate it into your projects,
and much more --- provided you acknowledge the original authors. Please refer to
[LICENSE.md](LICENSE.md) for more information. 

## Contributing

We welcome contributions and suggestions for our DECT package! Here are some
basic guidelines for contributing:

### How to Submit an Issue

1. **Check Existing Issues**: Before submitting a new issue, please check if it
   has already been reported.

2. **Open a New Issue**: If your issue is new, open a new issue in the
   repository. Provide a clear and detailed description of the problem,
   including steps to reproduce the issue if applicable.

3. **Include Relevant Information**: Include any relevant information, such as
   system details, version numbers, and screenshots, to help us understand and
   resolve the issue more efficiently.

### How to Contribute

If you're unfamiliar with contributing to open source repositories, here is a
basic roadmap:

1. **Fork the Repository**: Start by forking the repository to your own GitHub
   account.

2. **Clone the Repository**: Clone the forked repository to your local machine.

   ```sh
   git clone https://github.com/your-username/dect.git
   ```

3. **Create a Branch**: Create a new branch for your feature or bug fix.

   ```sh
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes**: Implement your changes in the new branch.

5. **Commit Changes**: Commit your changes with a descriptive commit message.

   ```sh
   git commit -m "Description of your changes"
   ```

6. **Push Changes**: Push the changes to your forked repository.

   ```sh
   git push origin feature/your-feature-name
   ```

7. **Submit a Pull Request**: Open a pull request to the main repository with a
   clear description of your changes and the purpose of the contribution.

### Need Help?

If you need any help or have questions, feel free to reach out to the authors or
submit a pull request. We appreciate your contributions and are happy to assist!
