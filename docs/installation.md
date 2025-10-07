# Installation

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
