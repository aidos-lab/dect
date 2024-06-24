"""
Tests the ect functions.
"""

import torch

from dect.ect import compute_ecc, normalize


class TestEct:
    """
    1. When normalized, the ect needs to be normalized.
    2. The dimensions need to correspond. e.g. the batches need not to be mixed up.
    3. Test that when one of the inputs has a gradient the out has one too.
    """

    def test_ecc_single(self):
        """
        Check that the dimensions are correct.
        lin of size [bump_steps, 1, 1, 1]
        """
        lin = torch.linspace(-1, 1, 13).view(-1, 1, 1, 1)
        index = torch.tensor([0, 0, 0], dtype=torch.long)
        nh = torch.tensor([[0.0], [0.5], [0.5]])
        scale = 100
        ecc = compute_ecc(nh, index, lin, scale)
        assert ecc.shape == (1, 1, 13, 1)

        # Check that min and max are 0 and 3
        torch.testing.assert_close(ecc.max(), torch.tensor(3.0))
        torch.testing.assert_close(ecc.min(), torch.tensor(0.0))

    def test_ecc_multi_set_directions(self):
        """
        Check that the dimensions are correct.
        lin of size [bump_steps, 1, 1, 1]
        """
        lin = torch.linspace(-1, 1, 13).view(-1, 1, 1, 1)
        index = torch.tensor([0, 0, 0], dtype=torch.long)
        nh = torch.tensor([[0.0, 0.0], [0.5, 0.5], [0.5, 0.5]])
        scale = 100
        ecc = compute_ecc(nh, index, lin, scale)
        assert ecc.shape == (1, 1, 13, 2)

    def test_ecc_multi_batch(self):
        """
        Check that the dimensions are correct.
        lin of size [bump_steps, 1, 1, 1]
        """
        lin = torch.linspace(-1, 1, 13).view(-1, 1, 1, 1)
        index = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        nh = torch.tensor([[0.0], [0.5], [0.5], [0.7], [0.7]])
        scale = 100
        ecc = compute_ecc(nh, index, lin, scale)
        assert ecc.shape == (2, 1, 13, 1)

        # Check that min and max are 0 and 1
        torch.testing.assert_close(ecc[0].max(), torch.tensor(2.0))
        torch.testing.assert_close(ecc[0].min(), torch.tensor(0.0))

        torch.testing.assert_close(ecc[1].max(), torch.tensor(3.0))
        torch.testing.assert_close(ecc[1].min(), torch.tensor(0.0))

    def test_ecc_normalized(self):
        """
        Check that the dimensions are correct.
        lin of size [bump_steps, 1, 1, 1]
        """
        lin = torch.linspace(-1, 1, 13).view(-1, 1, 1, 1)
        index = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        nh = torch.tensor([[0.0], [0.5], [0.5], [0.7], [0.7]])
        scale = 100
        ecc = compute_ecc(nh, index, lin, scale)
        assert ecc.shape == (2, 1, 13, 1)
        ecc_normalized = normalize(ecc)

        # Check that min and max are 0 and 1
        torch.testing.assert_close(ecc_normalized[0].max(), torch.tensor(1.0))
        torch.testing.assert_close(ecc_normalized[0].min(), torch.tensor(0.0))

        torch.testing.assert_close(ecc_normalized[1].max(), torch.tensor(1.0))
        torch.testing.assert_close(ecc_normalized[1].min(), torch.tensor(0.0))
