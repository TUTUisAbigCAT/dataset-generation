"""CPU helper/actual-main-fragment checks; no CUDA/model/CLI execution claimed."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_denoising as audit


def main_tree():
    tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")


def execute_node(node, namespace):
    module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]))
    exec(compile(module, str(audit.__file__) + "::cpu-test-fragment", "exec"), namespace)


class MaskedDenoisingTests(unittest.TestCase):
    def test_masks_come_from_original_preprocessor_and_are_copied(self):
        masks = {("one.npz", 0): np.array([True, False, True]),
                 ("two.npz", 750): np.array([False, True, True])}
        processor = SimpleNamespace(missing_value_policy="masked",
            window_observed_mask=mock.Mock(side_effect=lambda name, start, length: masks[(name, start)]))
        selected = list(masks)
        result = audit._validation_observed_masks(processor, selected, 3)
        self.assertTrue(np.array_equal(result, np.stack(list(masks.values()))))
        self.assertEqual(processor.window_observed_mask.call_args_list,
            [mock.call("one.npz", 0, 3), mock.call("two.npz", 750, 3)])
        result[0, 0] = False
        self.assertTrue(masks[("one.npz", 0)][0])

    def test_preserve_and_legacy_preprocessors_do_not_request_masks(self):
        for policy in ("preserve", None):
            processor = SimpleNamespace(window_observed_mask=mock.Mock(side_effect=AssertionError("must not call")))
            if policy is not None:
                processor.missing_value_policy = policy
            self.assertIsNone(audit._validation_observed_masks(processor, [("one.npz", 0)], 3))
            processor.window_observed_mask.assert_not_called()

    def test_invalid_original_masks_are_rejected_without_cast_or_fallback(self):
        for mask in (np.array([1, 0, 1]), np.zeros(3, dtype=bool), np.ones((1, 3), dtype=bool), np.ones(2, dtype=bool)):
            with self.subTest(mask=mask):
                processor = SimpleNamespace(missing_value_policy="masked", window_observed_mask=lambda *args: mask)
                with self.assertRaises(ValueError):
                    audit._validation_observed_masks(processor, [("one.npz", 0)], 3)
        with self.assertRaises(ValueError):
            audit._validation_observed_masks(SimpleNamespace(missing_value_policy="unknown"), [], 3)

    def test_every_metric_uses_same_observed_positions(self):
        predicted = torch.tensor([[1., 2., 100.]], dtype=torch.float64)
        target = torch.tensor([[3., -2., -200.]], dtype=torch.float64)
        truth = torch.tensor([.7, .9, 999.], dtype=torch.float64)
        estimate = torch.tensor([.8, 1.2, -999.], dtype=torch.float64)
        result = audit._teacher_forced_window_metrics(predicted, target, truth, estimate,
            torch.tensor([True, True, False]), torch.nn.MSELoss())
        expected = dict(epsilon_mse=10., predicted_epsilon_mean=1.5,
            target_epsilon_mean=.5, epsilon_mean_error=1., reference_mean_ms=800.,
            reconstructed_mean_ms=1000., x0_window_mean_absolute_error_ms=200.,
            x0_window_mean_signed_bias_ms=200.)
        for name, value in expected.items():
            self.assertAlmostEqual(result[name].item(), value, places=10, msg=name)

    def test_preserve_per_window_arithmetic_matches_previous_expressions(self):
        pred = torch.tensor([[1., -2., 10.]])
        target = torch.tensor([[3., 2., -5.]])
        truth = torch.tensor([.5, .8, 0.], dtype=torch.float64)
        estimate = torch.tensor([.6, .7, -.1], dtype=torch.float64)
        criterion = torch.nn.MSELoss()
        result = audit._teacher_forced_window_metrics(pred, target, truth, estimate, None, criterion)
        delta = (estimate.mean()-truth.mean())*1000
        previous = dict(epsilon_mse=criterion(pred, target), predicted_epsilon_mean=pred.double().mean(),
            target_epsilon_mean=target.double().mean(), epsilon_mean_error=(pred.double()-target.double()).mean(),
            reference_mean_ms=truth.mean()*1000, reconstructed_mean_ms=estimate.mean()*1000,
            x0_window_mean_absolute_error_ms=delta.abs(), x0_window_mean_signed_bias_ms=delta)
        for name in previous:
            self.assertTrue(torch.equal(result[name], previous[name]), name)

    def test_actual_main_batch_loss_and_row_fragments_use_mask_on_cpu(self):
        # Execute the real batch-reduction and per-row AST fragments with fixed
        # CPU tensors. This does not run the CUDA-only main or claim GPU testing.
        main = main_tree()
        evaluate = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate")
        batch_branch = next(node for node in ast.walk(evaluate) if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "observed_batch")
        row_loop = next(node for node in ast.walk(evaluate) if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name) and node.target.id == "i")
        predicted = torch.tensor([[[1., -2., 100.]], [[-3., 200., 300.]]])
        target = torch.zeros_like(predicted)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        context = dict(torch=torch, predicted=predicted, target=target, observed_batch=mask,
            criterion=torch.nn.MSELoss(), masked_noise_mse=mock.Mock(wraps=audit.masked_noise_mse))
        execute_node(batch_branch, context)
        context["masked_noise_mse"].assert_called_once_with(predicted, target, mask)
        self.assertAlmostEqual(context["batch_mse"].item(), 14/3, places=6)
        context.update(batch=torch.tensor([[[1.], [2.], [1000.]], [[3.], [1000.], [1000.]]]),
            estimate=torch.tensor([[[2., 3., -1000.]], [[4., -1000., -1000.]]]),
            preprocessor=SimpleNamespace(scaler={"signal_std_seconds": .1, "signal_mean_seconds": .7}),
            rows=[], identities=[{"subject_id":"one"}, {"subject_id":"two"}], offset=0,
            timesteps=torch.tensor([0, 1]), alpha_bars=torch.tensor([.9, .8]),
            state_name="trained", scenario="test", _teacher_forced_window_metrics=audit._teacher_forced_window_metrics)
        execute_node(row_loop, context)
        rows = context["rows"]
        self.assertEqual([row["epsilon_mse"].item() for row in rows], [2.5, 9.])
        for row, reference, reconstructed in zip(rows, (850., 1000.), (950., 1100.)):
            self.assertAlmostEqual(row["reference_mean_ms"].item(), reference, places=9)
            self.assertAlmostEqual(row["reconstructed_mean_ms"].item(), reconstructed, places=9)
        context["observed_batch"] = None
        execute_node(batch_branch, context)
        self.assertTrue(torch.equal(context["batch_mse"], torch.nn.functional.mse_loss(predicted, target)))

    def test_actual_main_exports_mask_and_describes_observed_only_protocol(self):
        main = main_tree()
        export_branch = next(node for node in main.body if isinstance(node, ast.If)
            and any(isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Subscript)
                    and isinstance(item.targets[0].value, ast.Name) and item.targets[0].value.id == "raw"
                    for item in node.body))
        protocol_branch = next(node for node in main.body if isinstance(node, ast.If)
            and any(isinstance(item, ast.Expr) and isinstance(item.value, ast.Call)
                    and isinstance(item.value.func, ast.Attribute) and item.value.func.attr == "update"
                    and isinstance(item.value.func.value, ast.Subscript)
                    and isinstance(item.value.func.value.value, ast.Name)
                    and item.value.func.value.value.id == "report" for item in node.body))
        observed = torch.tensor([[True, False, True]])
        context = dict(raw={}, observed_masks=observed, report={"protocol":{}, "runtime":{"host_boundaries":[]}})
        execute_node(export_branch, context)
        execute_node(protocol_branch, context)
        self.assertIs(context["raw"]["observed_mask"], observed)
        self.assertEqual(context["report"]["protocol"]["missing_value_policy"], "masked")
        self.assertIn("not all-point", context["report"]["protocol"]["loss"])
        self.assertFalse(context["report"]["protocol"]["missing_placeholder_scored_as_truth"])
        self.assertTrue(context["report"]["protocol"]["raw_observed_mask_exported"])
        legacy = dict(raw={}, observed_masks=None, report={"protocol":{"loss":"original"}, "runtime":{"host_boundaries":[]}})
        previous = copy.deepcopy(legacy)
        execute_node(export_branch, legacy)
        execute_node(protocol_branch, legacy)
        self.assertEqual(legacy["raw"], previous["raw"])
        self.assertEqual(legacy["report"], previous["report"])

    def identity_fixture(self, directory, *, policy=None, overrides=None):
        expected = dict(dataset="DREAMT", data_directory=str(directory.resolve()),
            metadata_path=str((directory/"participant_info.csv").resolve()),
            splits={"train":["one.npz"],"val":[],"test":[]}, scaler={"signal_mean_seconds":.8},
            metadata_file_sha256="m"*64, train_npz_sha256={"one.npz":"n"*64},
            window_epochs=4, zero_value_policy="recorded-zero-policy")
        if policy is not None:
            expected["missing_value_policy"] = policy
        calls = []
        actual = dict(expected, **(overrides or {}))
        class FakePreprocessor:
            def __init__(self, dataset, cyc, data_config):
                calls.append(data_config)
                self.window_size, self.window_epochs = 3000, 4
            def data_identity(self):
                return actual
        args = SimpleNamespace(dataset="DREAMT", window_size=3000, s4_lmax=3000, propCycEnc=False)
        return args, {"data_identity":expected}, FakePreprocessor, calls

    def test_manual_legacy_identity_branch_freezes_missing_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for policy in (None, "masked"):
                args, manifest, processor_class, calls = self.identity_fixture(folder, policy=policy)
                _, info = audit.bind_training_data(args, manifest, folder/"manifest.json", processor_class)
                self.assertEqual(calls[0]["missing_value_policy"], policy or "preserve")
                self.assertTrue(info["checks"]["missing_value_policy"])
                self.assertTrue(info["checks"]["zero_value_policy"])

    def test_binding_rejects_changed_missing_or_zero_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for field, bad in (("missing_value_policy","preserve"), ("zero_value_policy","different")):
                args, manifest, processor_class, _ = self.identity_fixture(folder, policy="masked", overrides={field:bad})
                with self.assertRaisesRegex(ValueError, field):
                    audit.bind_training_data(args, manifest, folder/"manifest.json", processor_class)


if __name__ == "__main__":
    unittest.main()
