# Third-party sources

## WaveStitch

Upstream: https://github.com/adis98/WaveStitch

Pinned commit: `2b6628026c0689815ce79becf11536a571d86b1a`.

The original files under `third_party/WaveStitch/` are preserved as a reproducibility reference. Their SHA256 values are recorded in `third_party/WaveStitch/SOURCE_MANIFEST.json`. The imported model, utilities, and training components derive from this source. The original MIT copyright and permission notice is preserved in `third_party/WaveStitch/LICENSE` and the root `LICENSE`; that notice names the upstream copyright holder and does not identify the author of every adaptation here.

The upstream README identifies components originating from SSSD, TimeGAN, and S4. Existing source comments and notices are retained. No new authorship or licensing claim is made for those components.

## WatchSleepNet

The user-provided `WatchSleepNet_public` source tree supplied the data-loader reference `modeling/data_setup.py` and the Fir submission template `train_transfer_sleepconvnet.sh`. The frozen reference files are retained under `third_party/WatchSleepNet/` for numerical comparisons and submission-script tests. `SOURCE_MANIFEST.json` records the loader hash and original relative source path; a machine-specific absolute path is intentionally omitted.

The supplied WatchSleepNet README displays an MIT license badge. The supplied tree did not contain a separate license text or a verifiable upstream repository URL, so this project does not fabricate a copyright notice or claim that WaveStitch's copyright notice covers WatchSleepNet. These files are attributed to the provided WatchSleepNet source, with their existing content preserved.

## Example data

All four records under `example_data/` and the accompanying participant CSV are artificial fixtures created by `scripts/create_example_data.py`. They are not DREAMT participant observations. The generation seed, dimensions, and hashes are recorded in `example_data/EXAMPLE_DATA.json`.
