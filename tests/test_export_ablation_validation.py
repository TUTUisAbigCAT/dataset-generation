"""CPU fixture/mocked-orchestration checks; no GPU or DREAMT-quality claim."""
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_utils import Preprocessor
from diffusion_schedule import build_diffusion_config, diffusion_identity
from epoch_validation import EpochValidator
from scripts.evaluate_denoising import DEFAULT_TRAINING_ARGS
from scripts import export_ablation_validation as export


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2))
        self.register_buffer("fixed", torch.randn(2))


class ExportValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = (ROOT/"work/export_ablation_validation_tests").resolve()
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.base = Path(self.temp.name)
        self.data = self.base/"data"
        shutil.copytree(ROOT/"example_data", self.data)
        for path in self.data.glob("*.npz"):
            with np.load(path, allow_pickle=False) as stored:
                values = {name:stored[name].copy() for name in stored.files}
            values["data"].reshape(-1)[::71] = 0
            np.savez(path, **values)
        self.run = self.base/"reference"
        for arm in export.ARMS:
            self.create_arm(arm)

    def tearDown(self):
        target = Path(self.temp.name).resolve()
        if not target.is_relative_to(self.temp_root) or target == self.temp_root:
            raise RuntimeError("Refuse cleanup outside the test temporary root")
        self.temp.cleanup()

    def create_arm(self, arm):
        root = self.run/arm
        workspace = root/"workspace"
        workspace.mkdir(parents=True)
        config = dict(data_dir=str(self.data), window_epochs=4, missing_value_policy=arm,
            diffusion_schedule="linear_power", terminal_alpha_bar=1e-8,
            validation=dict(max_windows=64,batch_size=8,seed=20261020,kl_variance_floor_ms2=1.,reference_mask_policy="observed_only"),
            early_stopping=dict(enabled=False))
        export.save_json(workspace/"dataset_config.json",config)
        processor = Preprocessor("DREAMT",data_config=config)
        args = SimpleNamespace(**dict(DEFAULT_TRAINING_ARGS,dataset="DREAMT",batch_size=8,epochs=10,stride=750,
                                     diffusion_schedule="linear_power",terminal_alpha_bar=1e-8))
        schedule = build_diffusion_config(args)
        # Only construct frozen protocol tensors on CPU. No model or sampling is
        # involved; the production CUDA guard is patched solely for this fixture.
        with mock.patch("epoch_validation._require_device",return_value=torch.device("cpu")):
            validator = EpochValidator(processor,args,None,schedule,torch.device("cpu"),root/"training/validation",config)
        torch.manual_seed(42)
        model = TinyModel()
        model.weight.data.fill_(3. if arm == "preserve" else 4.)
        checkpoint = workspace/"saved_models/DREAMT/model.pth"
        checkpoint.parent.mkdir(parents=True)
        torch.save(model.state_dict(),checkpoint)
        sources = {name:export.sha256(ROOT/name) for name in export.RUNTIME_SOURCES}
        sources["dataset_config.json"] = export.sha256(workspace/"dataset_config.json")
        manifest = dict(status="completed",checkpoint=str(checkpoint),checkpoint_sha256=export.sha256(checkpoint),
            checkpoint_epoch=10,completed_epochs=10,batches_per_epoch=1,updates=10,requested_epochs=10,
            training_args=vars(args),source_sha256=sources,data_identity=processor.data_identity(),
            diffusion_identity=diffusion_identity(args,schedule),missing_value_policy=arm,threads=4)
        export.save_json(root/"training/run_manifest.json",manifest)
        final_dir = root/"training/validation/epoch_0010"
        final_dir.mkdir()
        export.save_json(final_dir/"metrics.json",dict(status="completed",epoch=10,
            aggregate=dict(gaussian_kl_real_to_generated=1. if arm=="preserve" else 2.)))
        # A corrupt best manifest must remain irrelevant to final export.
        (root/"training/best_kl_manifest.json").write_text("not JSON",encoding="utf-8")
        (root/"training/best_kl_model.pth").write_bytes(b"different best weights")

    def mutate_json(self,path,mutate):
        data = export.read_json(path)
        mutate(data)
        export.save_json(path,data)

    def test_real_reference_identity_and_checkpoint_checks_choose_final_only(self):
        before = {path:export.sha256(path) for path in self.run.rglob("*") if path.is_file()}
        contexts,checks = export.load_reference_pair(self.run)
        self.assertTrue(all(checks.values()))
        self.assertEqual(contexts["preserve"]["epoch"],10)
        self.assertEqual(contexts["masked"]["updates"],10)
        for arm,c in contexts.items():
            self.assertEqual(c["checkpoint"].name,"model.pth")
            self.assertTrue(c["checkpoint_identity"]["recorded_digest_verified"])
            self.assertEqual(c["identity"].get("missing_value_policy","preserve"),arm)
        self.assertEqual(before,{path:export.sha256(path) for path in before})

    def test_wrong_or_missing_final_weights_fail_without_best_fallback(self):
        checkpoint = self.run/"preserve/workspace/saved_models/DREAMT/model.pth"
        checkpoint.write_bytes(b"incorrect checkpoint")
        with self.assertRaisesRegex(ValueError,"digest"):
            export.load_reference_pair(self.run)
        checkpoint.unlink()
        with self.assertRaises(FileNotFoundError):
            export.load_reference_pair(self.run)

    def test_checkpoint_hash_and_final_epoch_metadata_are_required(self):
        path = self.run/"preserve/training/run_manifest.json"
        original = path.read_bytes()
        for patch in ({"checkpoint_sha256":None},{"checkpoint_epoch":9},{"checkpoint_epoch":True},
                      {"checkpoint_selection":{"selection_rule":"best"}},{"updates":9}):
            with self.subTest(patch=patch):
                path.write_bytes(original)
                self.mutate_json(path,lambda d:d.update(patch))
                with self.assertRaises(ValueError):
                    export.load_reference_pair(self.run)

    def test_unequal_final_training_budgets_are_rejected(self):
        path = self.run/"masked/training/run_manifest.json"
        self.mutate_json(path,lambda d:d.update(updates=20,batches_per_epoch=2))
        with self.assertRaisesRegex(ValueError,"updates"):
            export.load_reference_pair(self.run)

    def test_changed_frozen_configuration_and_runtime_source_are_rejected(self):
        config = self.run/"preserve/workspace/dataset_config.json"
        original = config.read_bytes()
        config.write_bytes(original+b" ")
        with self.assertRaisesRegex(ValueError,"configuration hash"):
            export.load_reference_pair(self.run)
        config.write_bytes(original)
        self.mutate_json(self.run/"preserve/training/run_manifest.json",
            lambda d:d["source_sha256"].update({"validation_sampling.py":"0"*64}))
        with self.assertRaisesRegex(ValueError,"source differs"):
            export.load_reference_pair(self.run)

    def test_pair_windows_mask_hashes_and_test_sealing_are_checked(self):
        path = self.run/"masked/training/validation/protocol.json"
        original = path.read_bytes()
        mutations = [lambda p:p["windows"][0]["reference_mask"].update(sha256="0"*64),
                     lambda p:p["windows"][0].update(start_sample=750),
                     lambda p:p.update(test_npz_opened=True),
                     lambda p:p["windows"][0].update(filename="EXAMPLE004.npz")]
        for mutation in mutations:
            path.write_bytes(original)
            self.mutate_json(path,mutation)
            with self.assertRaises(ValueError):
                export.load_reference_pair(self.run)

    def test_reconstructed_protocol_allows_only_artifact_option_change(self):
        context,_ = export.load_reference_pair(self.run)
        original = context["preserve"]["protocol"]
        actual = deepcopy(original)
        actual["options"]["save_artifacts"] = True
        actual["device"] = "cuda:0"
        self.assertTrue(all(export.compare_protocols(original,actual).values()))
        for field in ("windows","condition_columns","scaler","diffusion_identity","initialization_seed","reference_mask"):
            changed = deepcopy(actual)
            changed[field] = None
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,field):
                export.compare_protocols(original,changed)
        for field in ("seed","batch_size","kl_variance_floor_ms2"):
            changed = deepcopy(actual)
            changed["options"][field] += 1
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"options"):
                export.compare_protocols(original,changed)

    def test_no_cuda_fails_readably_before_any_model_execution(self):
        output = self.base/"no_cuda"
        with mock.patch.object(export,"require_cuda",side_effect=RuntimeError("CUDA unavailable; no CPU fallback")), \
             mock.patch.object(export,"fetchModel") as model:
            with self.assertRaisesRegex(RuntimeError,"CUDA unavailable"):
                export.main(["--reference-run",str(self.run),"--output-dir",str(output)])
        model.assert_not_called()
        summary = export.read_json(output/"summary.json")
        self.assertEqual(summary["status"],"failed")
        self.assertFalse(summary["training_executed"])

    def test_existing_output_is_not_overwritten(self):
        output = self.base/"exists"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            export.main(["--reference-run",str(self.run),"--output-dir",str(output)])
        self.assertEqual(list(output.iterdir()),[])

    def test_output_cannot_modify_reference_run_or_create_descendants(self):
        before = {path.relative_to(self.run):export.sha256(path)
                  for path in self.run.rglob("*") if path.is_file()}
        with mock.patch.object(export,"load_reference_pair") as load:
            for output in (self.run,self.run/"new_export",self.run/"new_parent/deeper_export"):
                with self.subTest(output=output),self.assertRaisesRegex(ValueError,"outside the reference run"):
                    export.main(["--reference-run",str(self.run),"--output-dir",str(output)])
        load.assert_not_called()
        self.assertFalse((self.run/"new_export").exists())
        self.assertFalse((self.run/"new_parent").exists())
        self.assertEqual(before,{path.relative_to(self.run):export.sha256(path)
                                 for path in self.run.rglob("*") if path.is_file()})

    def test_export_runs_one_generation_and_no_training_or_checkpoint_selection(self):
        contexts,_ = export.load_reference_pair(self.run)
        context = contexts["masked"]
        processor = Preprocessor.from_identity(context["identity"])
        seen = {}

        def fetch(*args):
            model = TinyModel()
            model.to = mock.Mock(return_value=model)  # CUDA movement is mocked, not executed.
            seen["model"] = model
            return model

        class FakeValidator:
            def __init__(self,preprocessor,args,model,schedule,device,output,config):
                self.output = output
                self.output.mkdir(parents=True)
                self.protocol = deepcopy(context["protocol"])
                self.protocol["options"]["save_artifacts"] = config["validation"]["save_artifacts"]
                self.run = mock.Mock(side_effect=AssertionError("must not train or select best"))
                self.calls = 0
                seen["validator"] = self
                export.save_json(self.output/"protocol.json",self.protocol)

            def _generate(self,model):
                self.calls += 1
                torch.testing.assert_close(model.weight,torch.full((2,),4.))
                return ({"aggregate":{"gaussian_kl_real_to_generated":2.25},"diagnostics":{},
                         "_artifact_payload":torch.ones(2)}, {"mock_copy":True}, [{"mock_sampling":True}])

            def save_generation_artifacts(self,path,metrics,*,epoch):
                self.saved_epoch = epoch
                path.mkdir(parents=True)
                metrics.pop("_artifact_payload")
                return {"fixture_only":True,"directory":str(path)}

        real_load = torch.load
        with mock.patch.object(export,"bind_training_data",return_value=(processor,{"fixture_identity":True})), \
             mock.patch.object(export,"fetchModel",side_effect=fetch), \
             mock.patch.object(export,"assert_cuda_model"), \
             mock.patch.object(export,"build_diffusion_config",return_value={"T":200}), \
             mock.patch.object(export,"EpochValidator",FakeValidator), \
             mock.patch.object(export,"preserve_random_state",side_effect=lambda device:nullcontext()) as rng, \
             mock.patch.object(export.torch,"load",side_effect=lambda stream,**kwargs:real_load(stream,map_location="cpu",weights_only=True)):
            result = export.export_arm(context,self.base/"exported",torch.device("cuda:0"))
        self.assertEqual(seen["validator"].calls,1)
        seen["validator"].run.assert_not_called()
        self.assertEqual(seen["validator"].saved_epoch,10)
        rng.assert_called_once_with(torch.device("cuda:0"))
        self.assertEqual(result["kl_difference_from_original_final"],.25)
        self.assertFalse(result["best_checkpoint_selected"])
        self.assertFalse(result["training_executed"])
        self.assertEqual(result["checkpoint_sha256"],context["checkpoint_identity"]["sha256"])
        self.assertTrue((self.base/"exported/validation/epoch_0010/metrics.json").is_file())

    def test_changed_weights_after_initial_hash_are_rejected_before_torch_load(self):
        contexts,_ = export.load_reference_pair(self.run)
        context = contexts["preserve"]
        processor = Preprocessor.from_identity(context["identity"])
        context["checkpoint"].write_bytes(b"replacement after initial verification")
        model = mock.Mock()
        model.to.return_value = model
        with mock.patch.object(export,"bind_training_data",return_value=(processor,{})), \
             mock.patch.object(export,"fetchModel",return_value=model), \
             mock.patch.object(export,"assert_cuda_model"),mock.patch.object(export.torch,"load") as load:
            with self.assertRaisesRegex(ValueError,"changed after"):
                export.export_arm(context,self.base/"changed",torch.device("cuda:0"))
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
