import unittest

import torch

from revflow.checkpoints import DEFAULT_CONFIG, read_config
from revflow.models import RevFlow
from revflow.flow import sample, training_loss


class RevFlowTest(unittest.TestCase):
    def test_reference_size(self):
        with torch.device("meta"):
            model = RevFlow(**read_config(DEFAULT_CONFIG)["model"])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 689_948_160)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 689_653_248)

    def test_two_stage_flow(self):
        torch.manual_seed(7)
        model = RevFlow(input_size=2, hi_channels=8, lo_channels=4, hidden_size=32,
                        depth=2, num_heads=4, base_model_depth=1, num_classes=3)
        # Fresh readouts are zero-initialized. Open the gates to exercise both conditioning paths.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "adaLN_modulation.1" in name or ".linear.weight" in name:
                    param.normal_(std=.03)
        config = read_config(DEFAULT_CONFIG)
        labels = torch.tensor([0, 2])
        hi, lo = torch.randn(2, 4, 8), torch.randn(2, 4, 4)
        loss = training_loss(model, hi, lo, labels, config["flow"])
        loss.backward()
        for parameter in (model.condition_projection.weight, model.y_embedder.embedding_table.weight,
                          model.base_hi.linear.weight, model.base_lo.linear.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        setup = dict(config["sampling"], hi_steps=2, lo_steps=2)
        calls = []
        hook = model.blocks[0].register_forward_pre_hook(lambda _, args: calls.append(args[1].shape[1]))
        sampled = sample(model, labels, config["flow"], setup, initial_noise=(hi, lo))
        hook.remove()
        self.assertEqual(calls, [1, 1, 4, 4])
        for result, original in zip(sampled, (hi, lo)):
            self.assertEqual(result.shape, original.shape)
            self.assertTrue(result.isfinite().all())


if __name__ == "__main__":
    unittest.main()
