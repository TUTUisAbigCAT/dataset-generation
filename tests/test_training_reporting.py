"""Execute the actual epoch-loop AST on tiny test tensors, without a CUDA fallback."""
import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from epoch_validation import weighted_loss
from scripts.verify_upstream_boundary import expected_training_source

ROOT = Path(__file__).resolve().parents[1]


class TrainingReportingTests(unittest.TestCase):
    def test_short_last_batch_and_last_epoch_are_reported_after_updates(self):
        source = ast.parse((ROOT / "training_wavestitch.py").read_text())
        loop = next(node for node in ast.walk(source) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == "epoch")
        module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))

            def forward(self, batch, time):
                return (batch[:, :, :1] * self.weight).permute(0, 2, 1)

        model = Model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        losses, reports = [], []

        def criterion(predicted, target):
            loss = torch.nn.functional.mse_loss(predicted, target)
            losses.append(float(loss.detach()))
            return loss

        def validate(epoch, live_model, mean_loss):
            self.assertIs(live_model, model)
            self.assertTrue(model.training)
            self.assertIsInstance(mean_loss, float)
            self.assertEqual(int(optimizer.state[model.weight]["step"]), epoch * 2)
            expected = (losses[-2] * 8 + losses[-1] * 3) / 11
            self.assertAlmostEqual(mean_loss, expected)
            reports.append((epoch, mean_loss))

        scope = dict(torch=torch, np=np, model=model, optimizer=optimizer, criterion=criterion,
                     device=torch.device("cpu"), args=SimpleNamespace(epochs=3),
                     dataloader=[torch.ones(8, 6, 2), torch.full((3, 6, 2), 3.)],
                     diffusion_config={"T": 2, "alpha_bars": torch.tensor([[.9], [.8]])},
                     non_hier_cols=torch.tensor([0]), randint=torch.randint,
                     normal=torch.normal, sqrt=torch.sqrt,
                     validator=SimpleNamespace(run=validate, should_stop=False))
        with torch.random.fork_rng(), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(module, "actual_epoch_loop", "exec"), scope)
        self.assertEqual([row[0] for row in reports], [1, 2, 3])
        self.assertIsInstance(scope["total_loss"], float)
        self.assertEqual(scope["total_samples"], 11)

    def test_total_loss_uses_sample_weights(self):
        rows = [{"epsilon_mse": 1., "batch_samples": 8}, {"epsilon_mse": 9., "batch_samples": 3}]
        self.assertAlmostEqual(weighted_loss(rows), 35 / 11)
        with self.assertRaises(ValueError):
            weighted_loss([])

    def test_exact_allowlist_does_not_allow_loss_or_optimizer_changes(self):
        original = (ROOT / "third_party/WaveStitch/training_wavestitch.py").read_bytes()
        actual = (ROOT / "training_wavestitch.py").read_bytes()
        expected = expected_training_source(original)
        self.assertEqual(actual, expected)
        for before, after in ((b"nn.MSELoss()", b"nn.L1Loss()"),
                              (b"optimizer.step()", b"optimizer.step(); optimizer.step()"),
                              (b"total_loss / total_samples", b"total_loss / len(dataloader)")):
            self.assertIn(before, actual)
            self.assertNotEqual(actual.replace(before, after), expected)


if __name__ == "__main__":
    unittest.main()
