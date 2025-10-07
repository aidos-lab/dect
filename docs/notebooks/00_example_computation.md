```python
import torch 
```


```python
# Basic dataset with three points,three edges and one face.
points_coordinates = torch.tensor([[0.5, 0.0], [-0.5, 0.0], [0.5, 0.5]])

# edge_index describes the edges in the pytorch geometric convention, i.e., the first edge is 0,1; the second edge is 1,2 and the third edge is 2,0.
edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
face_index=torch.tensor([[0], [1], [2]], dtype=torch.long)

points_coordinates
```




    tensor([[ 0.5000,  0.0000],
            [-0.5000,  0.0000],
            [ 0.5000,  0.5000]])




```python
# We pick two direction between 0 and 2*pi

n_directions = 2
theta1 = torch.tensor(0.0)
theta2 = torch.tensor(torch.pi/2.)

# The xi needs to be transposed, as we want to multiply the points_coordinates with the direction vectors via matrix multiplication
xi = torch.tensor([[torch.sin(theta1),torch.cos(theta1)], [torch.sin(theta2),torch.cos(theta2)]]).T
xi
```




    tensor([[ 0.0000e+00,  1.0000e+00],
            [ 1.0000e+00, -4.3711e-08]])




```python
# Next we compute the node heights as the matrix product of the vertex coordinates 
# and the directions. This results in a matrix of shape (n_points, n_directions)

node_heigth = points_coordinates @ xi
assert node_heigth.shape == (points_coordinates.shape[0], n_directions)
node_heigth
```




    tensor([[ 0.0000,  0.5000],
            [ 0.0000, -0.5000],
            [ 0.5000,  0.5000]])




```python
# The "height" of each edge is defined as the maximum of the node heights of the 
# vertices it is spanned by as only then the complete edge is included. Since 
# the edge indices are given as tuples, we look up the edge height tuples in the 
# node heights vector (indexing is the same) and compute the column-wise maximum. 

edge_height_tuples = node_heigth[edge_index]
edge_height = edge_height_tuples.max(dim=0)[0]
edge_height
```




    tensor([[0.0000, 0.5000],
            [0.5000, 0.5000],
            [0.5000, 0.5000]])




```python
# For the face heights we perform the same computation.
face_height_tuples = node_heigth[face_index]
face_height = face_height_tuples.max(dim=0)[0]
face_height
```




    tensor([[0.5000, 0.5000]])




```python
# For the first direction the two critical points are 0 and 1/2 and these are the places the ect changes. 
# Below zero the ecc is zero, between 0 and 1/2 it is 2-1 = 1 (2 points 1 edges)
# and above 1/2 it is 3-3+1=1. 

# For the second direction the critical points are -1/2 and 1/2. Below -1/2 the ecc is 0, between -1/2 and 1/2,
# the ect is 1 (1 point) and above 1/2 it is 3-3+1=1 (3 points, 3 edges, 1 face).

# We find these numbers by counting all the points edges and faces below a certain
# value. 
```


```python
# Instead of counting points, we assign each element an indicator function that 
# zero below the critical point and 1 above it. 
# To do so, we translate the indicator function for each point, edge and face. 

# Discretize interval in 25 steps
interval = torch.stack([torch.linspace(-1,1,25) for _ in range(n_directions)], dim=1).view(-1, n_directions, 1)

# we shift the interval by the height of each element and then apply
# the Heaviside step function, which is 0 for negative values and 1 for 
# non-negative values.
translated_nodes =  interval - node_heigth.view(1,n_directions,-1) 

ecc_points = torch.heaviside(translated_nodes,values=torch.tensor([1.0]))

# For each direction and point, this will give us an indicator function, i.e., the shape needs to be (25, n_directions, n_points)
assert ecc_points.shape == (25, n_directions, points_coordinates.shape[0])
ecc_points[:,0]
```




    tensor([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [1., 0., 1.],
            [1., 0., 1.],
            [1., 0., 1.],
            [1., 0., 1.],
            [1., 0., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])




```python
# Note that for the first direction the 0 is at index 13 and 1/2 is at index 18. Indeed this is where 
# the curves change value. 

# We do the same for the faces and edges.
```


```python
# Discretize interval in 25 steps
interval = torch.stack([torch.linspace(-1,1,25) for _ in range(n_directions)], dim=1).view(-1, n_directions,1)

translated_edges = interval - edge_height.view(1,n_directions,-1)
ecc_edges = torch.heaviside(translated_edges,values=torch.tensor([1.0]))

# For each direction and edge, this will give us an indicator function, i.e., the shape needs to be (25, n_directions, n_edges)
assert ecc_edges.shape == (25, n_directions, edge_index.shape[1])
ecc_edges[:,0]
```




    tensor([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])




```python
# Discretize interval in 25 steps
interval = torch.stack([torch.linspace(-1,1,25).view(-1,1) for _ in range(n_directions)], dim=1).view(-1, n_directions,1)

translated_faces = interval - face_height.view(1,n_directions,-1)  
ecc_faces = torch.heaviside(translated_faces,values=torch.tensor([1.0]))

# For each direction and edge, this will give us an indicator function, i.e., the shape needs to be (25, n_directions, n_faces)
assert ecc_faces.shape == (25, n_directions, face_index.shape[1])
ecc_faces[:,0]
```




    tensor([[0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [1.],
            [1.],
            [1.],
            [1.],
            [1.],
            [1.],
            [1.]])




```python
# The ect along this direction is then computed by first computing the sum of
# columns in each of the three matrices and then by computing the 
# alternating sum of the three matrices.

ecc = ecc_points.sum(axis=-1) - ecc_edges.sum(axis=-1) + ecc_faces.sum(axis=-1) 
ecc
```




    tensor([[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
            [1., 1.],
            [1., 1.],
            [1., 1.],
            [1., 1.],
            [1., 1.],
            [1., 1.],
            [1., 1.],
            [1., 1.],
            [1., 1.],
            [1., 1.],
            [1., 1.],
            [1., 1.]])




```python
# We can indeed verify that for the first direction at index 13 the value changes from 0 to 1 (which is)
# the origin in our coordinate system.
```
