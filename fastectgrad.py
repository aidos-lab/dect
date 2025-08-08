"""
Fast ECT calculation for point cloud
backprop with custom gradient.
"""

import matplotlib.pyplot as plt
import torch

from dect.directions import generate_2d_directions


def bincount(idx, resolution):
    """Calculates the histogram in resolution bins."""
    x = torch.zeros(size=(resolution, resolution), dtype=torch.float32, device="cuda")
    return x.scatter_(0, idx.to(torch.int64), 1, reduce="add")


def fast_ect_fn(x, v):
    """Fast ECT for point clouds."""
    resolution = v.shape[1]
    nh = ((torch.matmul(x, v) + 1) * (resolution // 2)).to(torch.uint32)
    return bincount(nh, resolution), nh


# x = torch.rand(size=(5, 2)) * 0.5
# ect = fast_ect_fn(x, v)
# plt.imshow(ect.cumsum(dim=0).numpy())
# plt.show()


class FastECT(torch.autograd.Function):
    @staticmethod
    def forward(x, v):
        ect, idx = fast_ect_fn(x, v)
        return ect.cumsum(dim=0), ect, idx

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        (ect, ect_grad, idx) = outputs
        (_, v) = inputs
        ctx.save_for_backward(ect, ect_grad, idx, v)

    @staticmethod
    def backward(ctx, grad_output, _, __):
        (ect, ect_grad, idx, v) = ctx.saved_tensors
        grad = ect_grad * grad_output / v.shape[1]
        # Do not know if this will be correct.
        ect_final_grad = torch.gather(grad, dim=0, index=idx.to(torch.int64))
        out = ect_final_grad @ v.T
        return -1 * out, None


def fastect(x, v):
    ect, _, _ = FastECT.apply(x, v)
    return ect


v = generate_2d_directions(num_thetas=2048).cuda()
x_true = 0.5 * torch.rand(size=(10000, 2)).cuda()
x = torch.nn.Parameter(0.2 * (torch.rand(size=(10000, 2), device="cuda") - 0.5))


optimizer = torch.optim.Adam([x], lr=0.01)


for epoch in range(200):
    optimizer.zero_grad()
    ect_true = fastect(x_true, v)
    ect_pred = fastect(x, v)
    loss = torch.nn.functional.mse_loss(ect_pred, ect_true)
    loss.backward()
    optimizer.step()


# plt.imshow(ect_true.detach().numpy())
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0, 0].imshow(ect_pred.detach().cpu().numpy())
axes[0, 1].imshow(ect_true.detach().cpu().numpy())
x_true = x_true.cpu().numpy()
x = x.detach().cpu().numpy()
axes[1, 0].scatter(x_true[:, 0], x_true[:, 1])
axes[1, 0].scatter(x[:, 0], x[:, 1])
axes[1, 0].set_xlim([-1, 1])
axes[1, 0].set_ylim([-1, 1])
plt.show()
