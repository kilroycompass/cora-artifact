# CORA reproducibility artifact

This public repository contains the experimental artifact for the Conformal
Operating-point Recalibration Audit (CORA): measured score caches, fitted model
states, replay outputs, dataset metadata, provenance, tests, and reproduction
code. The manuscript, TeX sources, bibliography, figures, tables, and PDFs are
intentionally excluded.

Repository: https://github.com/kilroycompass/cora-artifact

## Contents

- `experiments/results/cora`: 120 original measured score caches, 180 replay
  grids, controls, traces, model records, and complete metrics.
- `experiments/results/tifs`: 79 additional fitted score caches, 186
  score/target cases, 6,120 query replays, fitted states, and complete metrics.
- `experiments`: preparation, fitting, replay, analysis, reporting, certificate,
  and test code.
- `data`: public-release metadata, verified source manifests, timestamp-alignment
  evidence, split metadata, and window counts. Large raw traffic and prepared
  tensors are excluded.
- `provenance`: frozen protocol settings, dependency versions, baseline
  attribution, and SHA-256 manifests for all 199 measured score inputs.
- `CITATION.cff`: citation metadata for the artifact.
- `SHA256SUMS`: hashes of every file in the downloadable artifact bundle.

The original experiment contains 9,720 replay configurations and 87,480
method/configuration pairs, of which 87,420 are supported. The extension contains
55,080 method/configuration pairs, of which 55,060 are supported. Unsupported
conditions remain explicit rather than being counted as successful zero-error
cases.

## Verify and reproduce

Use Python 3.11 and install `requirements.txt`.

    python experiments/reproduce.py --verify-only

This verifies all 199 measured score inputs against the two provenance
manifests. To run the 27 protocol, certificate, and window-control checks and
regenerate cached replay analyses:

    python experiments/reproduce.py

For a fresh replay, use `--recompute`. Existing generated replay/control
directories are moved into a timestamped local `build/recompute-*` backup before
recomputation. Measured scores and fitted model states are preserved.

The reporting programs may create local `paper/generated` and `paper/figures`
directories as machine-generated views of the metrics. These paths are ignored
and are not part of this repository.

## Optional raw-data workflow

Raw public datasets are not redistributed. The retained source metadata and
downloaders identify and verify the publisher releases. Relevant stages are:

    python experiments/prepare_cic.py cic17
    python experiments/prepare_cic.py cic18
    python experiments/fetch_netflow.py NF-UNSW-NB15-v3 NF-ToN-IoT-v3 NF-BoT-IoT-v3
    python experiments/prepare_netflow.py
    python experiments/prepare_iot.py
    python experiments/train.py
    python experiments/fetch_5gnidd.py
    python experiments/extract_5g_times.py --stations 1
    python experiments/tifs_windows.py
    python experiments/tifs_extension.py --stage query

The 5G epoch extraction requires `ra` from Argus 3.0.8.2. BS2 and HIKARI are
recorded exclusions because their released feature inputs did not provide a
usable verified chronology. No row-order or synthetic timestamps substitute for
missing time.

## Scope

All results come from actual fitted scores and recorded public labelled traffic.
The feasibility certificate is conditional on the recorded trace and fallback;
it does not establish future-site coverage. Reused streams, repeated fits, and
query schedules do not create independent networks. Labels, telemetry, and
models are assumed trustworthy. Adaptive evasion, label poisoning, and analyst
error are outside the evaluated scope.

No software license is granted by the repository as a whole. Redistributed
third-party reference implementations retain their included upstream licences.

