"""
Fast ECT calculation for point cloud
backprop with custom gradient.
"""

import matplotlib.pyplot as plt
import torch


class Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(x, _):
        return torch.heaviside(x, values=torch.tensor(0.0))

    @staticmethod
    def setup_context(ctx, inputs, _):
        (x, slope) = inputs
        ctx.slope = slope
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grid_size = len(input_)
        ind_leq = input_ <= 1 / (1.5 * grid_size)
        ind_geq = input_ >= -1 / (1.5 * grid_size)
        grad = (1 / ctx.slope) * (ind_leq & ind_geq) * grad_output
        return grad, None


def fastsigmoid(slope=0.01):
    """Sigmoid surrogate gradient enclosed with a parameterized slope."""
    slope = slope

    def inner(x):
        return Sigmoid.apply(x, slope)

    return inner


lin = torch.linspace(0, 1, 200)
h = torch.nn.Parameter(data=torch.tensor(0.5))

h_true = torch.tensor(0.75)
sig = fastsigmoid()


optimizer = torch.optim.Adam([h], lr=0.005)

sig_true = sig(lin - h_true.unsqueeze(0))


hs = []

for epoch in range(400):
    optimizer.zero_grad()
    sig_pred = sig(lin - h.unsqueeze(0))

    loss = torch.nn.functional.mse_loss(sig_true, sig_pred)

    loss.backward()
    optimizer.step()

    hs.append(h.item())

    print(
        epoch,
        "Loss:",
        loss.item(),
        "h",
        h.item(),
        # "Sig_true",
        # sig_true.item(),
        # "Sig Pred",
        # sig_pred.item(),
    )

plt.plot(hs)
plt.show()
print(1 / 200)
