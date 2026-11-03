"""Host startup regression tests: stdlib subprocesses, no model execution."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from host_runtime import configure_host_threads


# Keep expected values independent of the implementation's policy dictionary.
EXPECTED_FIXED = {
    "BLIS_THREAD_IMPL": "single",
    "BLIS_NUM_THREADS": "1",
    "BLIS_JC_NT": "1",
    "BLIS_PC_NT": "1",
    "BLIS_IC_NT": "1",
    "BLIS_JR_NT": "1",
    "BLIS_IR_NT": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "PYTHONFAULTHANDLER": "1",
}
CONTROLLED_KEYS = (*EXPECTED_FIXED, "OMP_NUM_THREADS", "MKL_NUM_THREADS")
NUMERICAL_ROOTS = ("numpy", "torch", "scipy", "pandas", "sklearn", "blis")


def hostile_environment():
    environment = dict(os.environ)
    environment.update({key: "64" for key in EXPECTED_FIXED})
    environment.update(BLIS_THREAD_IMPL="pthreads", OMP_DYNAMIC="TRUE", MKL_DYNAMIC="TRUE",
                       PYTHONFAULTHANDLER="0", OMP_NUM_THREADS="23", MKL_NUM_THREADS="29")
    return environment


def temporary_directory():
    parent = ROOT / "work/host_thread_runtime_tests"
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


def child_json(source, *arguments, environment=None):
    # -S also excludes site initialization: the probe observes the entry point,
    # and never needs an installed numerical package.
    result = subprocess.run(
        [sys.executable, "-B", "-S", "-c", textwrap.dedent(source), str(ROOT), *map(str, arguments)],
        cwd=ROOT, env=hostile_environment() if environment is None else environment,
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


IMPORT_PROBE = r"""
    import builtins
    import json
    import os
    import runpy
    import sys
    sys.path.insert(0, sys.argv[1])
    target, arguments = sys.argv[2], json.loads(sys.argv[3])
    keys = json.loads(sys.argv[4])
    numerical_roots = set(json.loads(sys.argv[5]))
    original_import = builtins.__import__
    record = {}

    class ImportBoundaryReached(BaseException):
        pass

    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in numerical_roots:
            record.update(first_numerical_import=name,
                          environment={key: os.environ.get(key) for key in keys},
                          loaded_numerical_modules=sorted(numerical_roots.intersection(sys.modules)))
            raise ImportBoundaryReached()
        return original_import(name, *args, **kwargs)

    builtins.__import__ = guarded_import
    sys.argv = [target, *arguments]
    try:
        runpy.run_path(target, run_name='__main__')
    except ImportBoundaryReached:
        pass
    else:
        raise AssertionError('Entry point did not reach a numerical import')
    finally:
        builtins.__import__ = original_import
    print(json.dumps(record))
"""


class HostThreadPolicyTests(unittest.TestCase):
    def assert_policy(self, environment, omp="4", mkl="4"):
        self.assertEqual(environment, dict(EXPECTED_FIXED, OMP_NUM_THREADS=omp, MKL_NUM_THREADS=mkl))

    def test_explicit_count_overrides_inherited_blis_manual_loop_controls(self):
        environment = hostile_environment()
        environment["UNRELATED_SETTING"] = "keep me"
        before_process = dict(os.environ)
        report = configure_host_threads(4, environ=environment)
        self.assert_policy({key: environment[key] for key in CONTROLLED_KEYS})
        self.assertEqual(environment["UNRELATED_SETTING"], "keep me")
        self.assertEqual(os.environ, before_process)
        self.assertEqual(report["policy"], "single_thread_blis")
        self.assertEqual(report["requested_host_threads"], 4)
        self.assert_policy(report["environment"])
        self.assertNotIn("UNRELATED_SETTING", report["environment"])

    def test_bootstrap_preserves_existing_omp_and_mkl_counts(self):
        environment = hostile_environment()
        report = configure_host_threads(environ=environment)
        self.assert_policy(report["environment"], omp="23", mkl="29")
        self.assertIsNone(report["requested_host_threads"])
        configure_host_threads(7, environ=environment)
        repeated = configure_host_threads(environ=environment)
        self.assert_policy(repeated["environment"], omp="7", mkl="7")

    def test_bootstrap_does_not_invent_missing_omp_or_mkl_counts(self):
        environment = {}
        report = configure_host_threads(environ=environment)
        self.assertEqual(environment, EXPECTED_FIXED)
        self.assert_policy(report["environment"], omp=None, mkl=None)

    def test_invalid_counts_fail_before_mutating_environment(self):
        for invalid in (True, False, 0, -1, 1.5, "4", [], float("nan"), float("inf")):
            with self.subTest(threads=invalid):
                environment = hostile_environment()
                original = environment.copy()
                with self.assertRaises(ValueError):
                    configure_host_threads(invalid, environ=environment)
                self.assertEqual(environment, original)

    def test_report_is_a_serializable_snapshot_of_the_applied_policy(self):
        environment = {}
        report = configure_host_threads(4, environ=environment)
        environment["BLIS_JC_NT"] = "64"
        configure_host_threads(9, environ=environment)
        self.assert_policy(json.loads(json.dumps(report))["environment"])

    def test_default_environment_policy_is_inherited_by_a_real_child(self):
        observed = child_json(r'''
            import json
            import subprocess
            import sys
            sys.path.insert(0, sys.argv[1])
            from host_runtime import configure_host_threads
            report = configure_host_threads(6)
            code = "import json, os; print(json.dumps({key: os.environ.get(key) for key in " + repr(list(report['environment'])) + "}))"
            result = subprocess.run([sys.executable, '-B', '-S', '-c', code],
                                    check=True, capture_output=True, text=True, timeout=15)
            print(json.dumps({'report': report, 'child': json.loads(result.stdout)}))
        ''')
        self.assert_policy(observed["child"], omp="6", mkl="6")
        self.assertEqual(observed["child"], observed["report"]["environment"])

    def test_helper_and_orchestrator_import_without_numerical_libraries(self):
        observed = child_json(r'''
            import builtins
            import json
            import sys
            sys.path.insert(0, sys.argv[1])
            blocked = set(json.loads(sys.argv[2]))
            original = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name.split('.')[0] in blocked:
                    raise AssertionError('Eager numerical import: ' + name)
                return original(name, *args, **kwargs)
            builtins.__import__ = guarded
            import host_runtime
            from scripts import run_dreamt
            print(json.dumps({'loaded': sorted(blocked.intersection(sys.modules))}))
        ''', json.dumps(NUMERICAL_ROOTS))
        self.assertEqual(observed["loaded"], [])

    def probe(self, script, arguments=()):
        observed = child_json(IMPORT_PROBE, script, json.dumps(list(arguments)),
                              json.dumps(CONTROLLED_KEYS), json.dumps(NUMERICAL_ROOTS))
        self.assertEqual(observed["loaded_numerical_modules"], [])
        self.assertIn(observed["first_numerical_import"].split(".")[0], ("numpy", "torch"))
        return observed["environment"]

    def test_dreamt_applies_cli_threads_before_first_numerical_import(self):
        with temporary_directory() as folder:
            output = Path(folder) / "run"
            environment = self.probe(ROOT / "scripts/run_dreamt.py",
                                     ("--threads", "7", "--output-dir", str(output)))
            self.assert_policy(environment, omp="7", mkl="7")
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["host_thread_runtime"]["environment"], environment)
            self.assertEqual(status["host_thread_runtime"]["requested_host_threads"], 7)
            self.assertEqual(status["stages"], [])
            self.assertFalse((output / "workspace").exists())

    def test_direct_worker_and_sweep_bootstrap_before_numerical_imports(self):
        for name in ("run_simulation.py", "run_all88_s092_sweep.py"):
            with self.subTest(script=name):
                self.assert_policy(self.probe(ROOT / "scripts" / name), omp="23", mkl="29")

    def test_frozen_workspace_contains_exact_helper_and_safe_worker_startup(self):
        with temporary_directory() as folder:
            destination = Path(folder) / "workspace"
            child_json(r'''
                import json
                from pathlib import Path
                import sys
                sys.path.insert(0, sys.argv[1])
                from scripts.run_dreamt import snapshot_source
                snapshot_source(Path(sys.argv[2]))
                print(json.dumps({'copied': True}))
            ''', destination)
            self.assertEqual((ROOT / "host_runtime.py").read_bytes(),
                             (destination / "host_runtime.py").read_bytes())
            self.assertEqual((ROOT / "scripts/run_simulation.py").read_bytes(),
                             (destination / "scripts/run_simulation.py").read_bytes())
            self.assert_policy(self.probe(destination / "scripts/run_simulation.py"), omp="23", mkl="29")

    def test_slurm_exports_blis_controls_before_first_python_command(self):
        source = (ROOT / "submit_all88_s092_sweep.sh").read_text(encoding="utf-8")
        exported = {}
        for line in source.replace("\\\n", " ").splitlines():
            tokens = shlex.split(line, comments=True)
            if "python" in tokens:
                break
            if tokens and tokens[0] == "export":
                exported.update(token.split("=", 1) for token in tokens[1:])
        else:
            self.fail("Slurm script has no Python launch")
        self.assertEqual({key: exported.get(key) for key in EXPECTED_FIXED}, EXPECTED_FIXED)
        self.assertEqual(exported["OMP_NUM_THREADS"], "${SLURM_CPUS_PER_TASK:-4}")
        self.assertEqual(exported["MKL_NUM_THREADS"], "$OMP_NUM_THREADS")
        self.assertIn("#SBATCH --array=0-11", source)
        self.assertIn("#SBATCH --cpus-per-task=4", source)
        self.assertIn("#SBATCH --gpus-per-node=1", source)


if __name__ == "__main__":
    unittest.main()
