# Third-party sources

## WaveStitch

Upstream: https://github.com/adis98/WaveStitch

Pinned commit: `2b6628026c0689815ce79becf11536a571d86b1a`.

The original files under `third_party/WaveStitch/` are preserved as a reproducibility reference. Their SHA256 values are recorded in `third_party/WaveStitch/SOURCE_MANIFEST.json`. The imported model, utilities, and training components derive from this source. The original MIT copyright and permission notice is preserved in `third_party/WaveStitch/LICENSE` and the root `LICENSE`; that notice names the upstream copyright holder and does not identify the author of every adaptation here.

The upstream README identifies components originating from SSSD, TimeGAN, and S4. Existing source comments and notices are retained. No new authorship or licensing claim is made for those components.

The local runtime sampler applies one explicit correction to the extracted WaveStitch pipeline: for positive diffusion steps, it multiplies standard Gaussian noise by the square root of the posterior variance, rather than by the posterior variance itself. The zero-step behavior, reverse mean, random draws, conditions, and guidance coefficient are retained. This correction is a local adaptation, not a claim that the pinned upstream source already contains it. The vendored source and its recorded hashes remain unchanged. The source-boundary checker permits exactly this expression change within the sampling block, and evaluation provenance distinguishes the unchanged upstream snapshot from the corrected runtime excerpt.

The imported local `TSImputers/S4Model.py` also repairs `cauchy_slow`: before the Cauchy sum it expands both `v` and `w` with their complex conjugates. This follows the conjugate-pair convention in the [official S4 Cauchy reference implementation](https://github.com/state-spaces/s4/blob/main/src/models/functional/cauchy.py). The PyTorch fallback is always defined and is always used for CPU inputs. A local device/dtype dispatcher selects the optional CUDA extension only for supported CUDA inputs; otherwise it uses PyKeOps when installed, or the same-device PyTorch fallback. Accelerator inputs materialize conjugate views as contiguous tensors, and PyKeOps selects its backend from the input device. No implicit device transfer or precision downgrade is introduced. These routing changes are local adaptations, not changes attributed to the pinned upstream source.

The source-boundary checker permits exactly three literal block replacements for PyKeOps input/device handling, the conjugate fallback and dispatcher, and the S4 call site; the other imported core files and every vendored snapshot remain unchanged. The network architecture and parameter names are unchanged, but the fallback operator and backend routing differ, so imported-core provenance is no longer byte-identical. Production checkpoint evaluation requires the recorded S4 source hash to match the executing source; historical weights must be reproduced in their original frozen workspace instead of silently switching operators. The separate backend checker validates numerical results and gradients on the requested actual device; CPU results are not evidence that a CUDA backend has passed.

The separate local `diffusion_schedule.py` adapter adds an explicitly configured `linear_power` schedule: it raises the original linear schedule's alpha values to a common power so that the final cumulative alpha approaches a positive target (new-run default `1e-8`). This schedule adaptation leaves the 200-step epsilon objective, model architecture, optimizer and training batch formulas unchanged; the separate S4 fallback correction is described above. The imported upstream `training_utils.py` and all vendored files remain byte-identical; the local training entry point has an allowlisted adapter import and schedule configuration fields. Training and evaluation use the same recorded schedule, with full effective beta values and tensor hashes in each run's provenance. Historical manifests without schedule fields retain the original linear schedule, subject to the source-identity check. This schedule adaptation is local and is not attributed to the pinned upstream implementation.

## WatchSleepNet

The user-provided `WatchSleepNet_public` source tree supplied the data-loader reference `modeling/data_setup.py` and the Fir submission template `train_transfer_sleepconvnet.sh`. The frozen reference files are retained under `third_party/WatchSleepNet/` for numerical comparisons and submission-script tests. `SOURCE_MANIFEST.json` records the loader hash and original relative source path; a machine-specific absolute path is intentionally omitted.

The supplied WatchSleepNet README displays an MIT license badge. The supplied tree did not contain a separate license text or a verifiable upstream repository URL, so this project does not fabricate a copyright notice or claim that WaveStitch's copyright notice covers WatchSleepNet. These files are attributed to the provided WatchSleepNet source, with their existing content preserved.

## Example data

All four records under `example_data/` and the accompanying participant CSV are artificial fixtures created by `scripts/create_example_data.py`. They are not DREAMT participant observations. The generation seed, dimensions, and hashes are recorded in `example_data/EXAMPLE_DATA.json`.
