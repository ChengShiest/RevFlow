import unittest

import torch

from revflow.checkpoints import DEFAULT_CONFIG, read_config
from revflow.models import RevFlow
from revflow.native_matching.field import distribution_gradient, kernel_loss
from revflow.native_matching.model import guided_pair


class NativeMatchingTest(unittest.TestCase):
    def test_nm_parent_and_recipe(self):
        config = read_config(DEFAULT_CONFIG.with_name("revflow_xl_nm_256.yaml"))
        self.assertEqual(config["native_matching"]["kernel"], "image_marginals")
        with torch.device("meta"):
            model = RevFlow(**config["model"])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 852_117_460)

    def test_unpaired_kernel_and_calibration(self):
        torch.manual_seed(9)
        x, y = torch.randn(2, 3, 256, 32), torch.randn(2, 3, 256, 32)
        kind = "image_marginals"
        # Each stage may permute images within its own marginal independently.
        reordered = torch.cat((y[..., :16], y[..., 16:].flip(1)), dim=-1)
        torch.testing.assert_close(kernel_loss(x, y, 16, kind), kernel_loss(x, reordered, 16, kind))
        gradient, coefficient = distribution_gradient(x, y, 16, kind)
        self.assertGreater(coefficient, 0.)
        self.assertTrue(gradient.isfinite().all())
        self.assertEqual(distribution_gradient(x, y, 16, kind, coefficient)[1], coefficient)

    def test_ddt_two_calls_and_lo_gradient(self):
        model = RevFlow(input_size=2, hi_channels=8, lo_channels=4, hidden_size=32, depth=2,
                        base_model_depth=1, num_heads=4, num_classes=3, ddt_width=32, ddt_heads=4)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "modulation" in name or ".linear.weight" in name:
                    p.normal_(std=.02)
        calls = []
        handle = model.blocks[0].register_forward_hook(lambda *_: calls.append(1))
        out = guided_pair(model, torch.randn(2, 4, 12), torch.ones(2), torch.tensor([0, 2]), 3., 2.)
        handle.remove()
        self.assertEqual(len(calls), 2)
        out[..., 8:].square().mean().backward()
        self.assertGreater(model.ddt_hi.linear.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
