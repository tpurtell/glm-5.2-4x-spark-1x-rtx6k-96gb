# GLM-5.2 calibrated EXL3 K3 quantization

This directory contains the reproducible, restartable quantization path for the
GLMRT GLM-5.2 EXL3 model. It quantizes only the routed experts in decoder
layers 3 through 77 to native EXL3 K3/MCG tensors. All other tensors remain in
their source format. The generated artifact is intended for GLMRT's TP4 Spark
EXL3 loader; it is not an NVFP4 checkpoint and does not keep a second resident
expert representation.

The production topology is exactly two compute-capability 12.0 coordinator
GPUs. The image, Python packages, Rust toolchain, GPTQModel fork, input model,
corpus, hardware identities, and numerical recipe are all content-bound in the
preflight report and execution plan. A resume fails closed if any immutable
input changes.

## Prepare the pinned source

`build.sh` initializes and verifies the GPTQModel submodule automatically. To
prepare it without running the full build:

```bash
git submodule sync -- third_party/gptqmodel
git submodule update --init --checkout -- third_party/gptqmodel
python3 scripts/verify-gptqmodel-source.py \
  --source third_party/gptqmodel \
  --lock third_party/gptqmodel.lock.json
```

Do not substitute upstream GPTQModel or a PyPI wheel. The GLMRT fork contains
the GLM-5.2 model definition, EXL3 Hessian/projection checkpointing, bounded
layer state, and crash-consistent resume support used by this recipe.

## Build the calibration corpus

The corpus builder creates source-disjoint calibration, screening, and
held-out JSONL files. It records the exact committed training-data inputs,
builder revision, tokenizer hash, selection seed, per-record provenance, and
split hashes in `manifest.json`. Both the GLMRT checkout and training-data
checkout must be clean for the paths consumed by the builder.

The pinned quantization image provides the required Python packages. Mount the
two Git checkouts at their real absolute paths so provenance remains valid:

```bash
repo_root="$(git rev-parse --show-toplevel)"
training_root="$(git -C ../training-data rev-parse --show-toplevel)"
source_snapshot=/path/to/GLM-5.2/snapshot
corpus_root=/path/to/glm-5.2-exl3-k3-corpus
quant_image=glmrt-quant-coordinator:exl3-k3

docker run --rm \
  -v "$repo_root:$repo_root:ro" \
  -v "$training_root:$training_root:ro" \
  -v "$source_snapshot:$source_snapshot:ro" \
  -v "$(dirname "$corpus_root"):$(dirname "$corpus_root")" \
  "$quant_image" \
  python "$repo_root/python/tools/build_glm52_calibration_corpus.py" \
    --training-data "$training_root" \
    --tokenizer "$source_snapshot/tokenizer.json" \
    --output "$corpus_root"
```

The defaults select about 1.08 million calibration prompt tokens and 65,536
held-out prompt tokens across general, code/agentic, math, reasoning
termination, and structured-output axes. To create a different calibration,
change the explicit token budgets or seed; that produces a different manifest
and therefore a different quantization plan.

## Build and qualify the image

Build arguments bind the fully hashed dependency locks and exact GPTQModel
revision into the image:

```bash
quant_req_sha="$(sha256sum quantization/requirements.amd64.lock | awk '{print $1}')"
quant_build_req_sha="$(sha256sum quantization/build-requirements.lock | awk '{print $1}')"
gptq_revision="$(git -C third_party/gptqmodel rev-parse HEAD)"
quant_image=glmrt-quant-coordinator:exl3-k3

docker build --platform linux/amd64 \
  -f docker/Dockerfile.quantization \
  --build-arg TARGETARCH=amd64 \
  --build-arg GLMRT_QUANT_REQUIREMENTS_SHA256="$quant_req_sha" \
  --build-arg GLMRT_QUANT_BUILD_REQUIREMENTS_SHA256="$quant_build_req_sha" \
  --build-arg GLMRT_GPTQMODEL_COMMIT="$gptq_revision" \
  -t "$quant_image" .
```

Run preflight with exactly the two GPUs that will perform the quantization. The
report is a required plan input, not merely diagnostic output:

```bash
state_root=/path/to/glm-5.2-exl3-k3-run
image_digest="$(docker image inspect --format '{{.Id}}' "$quant_image")"

docker run --rm --gpus '"device=0,1"' --ipc=host \
  -e GLMRT_QUANT_IMAGE_DIGEST="$image_digest" \
  -v "$(dirname "$state_root"):$(dirname "$state_root")" \
  "$quant_image" \
  python /opt/glmrt/quantization/preflight.py \
    --require-image-digest \
    --output "$state_root/coordinator-preflight.json"
```

Preflight checks the target platform/CUDA architecture, both physical GPU
identities and power limits, driver, package locks, free-threaded Python,
Rust/uv versions, cuSPARSELt metadata, image digest, and exact GPTQModel source
tree. Keep the report with the run state.

## Plan, run, and resume

Use fast scratch storage for projection checkpoints. Put rolling layer state,
active-layer source staging, and offload state on durable NVMe. The source
snapshot and all state paths must be visible inside the container at the same
absolute paths.

```bash
output_root=/path/to/GLM-5.2-EXL3-K3-calibrated
projection_root=/fast-scratch/glm-5.2-exl3-k3/projection-checkpoints
run_state="$state_root/run-state"
active_source="$run_state/active-layer-source"
offload_root="$state_root/offload"

quant_args=(
  python -X faulthandler
  /opt/glmrt/quantization/quantize_glm52_gptqmodel.py
  --snapshot "$source_snapshot"
  --calibration-jsonl "$corpus_root/calibration.jsonl"
  --calibration-manifest "$corpus_root/manifest.json"
  --preflight-report "$state_root/coordinator-preflight.json"
  --output "$output_root"
  --run-state-dir "$run_state"
  --projection-checkpoint-dir "$projection_root"
  --active-layer-source-dir "$active_source"
  --offload-dir "$offload_root"
)

docker run --rm --gpus '"device=0,1"' --ipc=host \
  -v "$source_snapshot:$source_snapshot:ro" \
  -v "$(dirname "$corpus_root"):$(dirname "$corpus_root"):ro" \
  -v "$(dirname "$state_root"):$(dirname "$state_root")" \
  -v "$(dirname "$projection_root"):$(dirname "$projection_root")" \
  -v "$(dirname "$output_root"):$(dirname "$output_root")" \
  "$quant_image" "${quant_args[@]}" --plan-only
```

Remove `--plan-only` to start a fresh run. If the process is interrupted and
the image and source are unchanged, run the identical command with `--resume`.
Do not add `--resume` to a fresh run and do not edit the run-state files.
`--stop-after-layer N` is available for a
qualification run; it stops only after layer `N` has committed a complete
rolling boundary and is intentionally excluded from the immutable numerical
plan. Resume also handles interruption during final export: an export with its
final commit marker is fully hash-verified, fsynced, and atomically published;
an unmistakably partial stage is discarded and rebuilt from the durable packed
projection checkpoints without repeating trellis search. A malformed committed
stage fails closed.

If checkpoint-only execution code must change after a failure, first commit
and pin the corrected GPTQModel source, rebuild the quantization image, and
produce a new two-GPU preflight report. Then reuse the parent command and add:

```bash
--execution-upgrade --resume
```

This does not regenerate or edit the immutable parent plan. It writes a signed
`glmrt-execution-upgrade.json` that binds the parent plan, old and new image,
GPTQModel and toolchain identities, unchanged Python/Torch and GPU UUIDs, the
exact error-journal frontier, and the retained rolling layer boundary. Repeated
upgrades form an authenticated history. The active upgrade is copied into the
finished artifact and referenced by its run manifest. A changed model,
calibration stream, numerical recipe, storage root, batch size, runtime, or GPU
identity fails closed. If interruption lands between archiving the active
upgrade and atomically installing its successor, the next upgrade launch
removes only an exact duplicate of the still-active record; differing or
unlinked history continues to fail closed.

The current router-candidate payload contract is
`gptqmodel.exl3-router-candidate-capture-v3`. It requires an explicit model
score adapter, reproduces the live router with its original top-k width, and
then treats that selected set as authoritative while ranking the adjacent
recovery candidates. This avoids relying on nested `torch.topk` results across
different widths at a tied score boundary. Changing this contract selects a
new capture-spool key; obsolete scratch for the same layer/subset is removed
when the replacement spool opens, so malformed evidence cannot be reused and
does not remain as a second 13 GB layer capture.

For GLM-MoE-DSA under the pinned Transformers runtime, that adapter reproduces
the router exactly: FP32 logits pass through sigmoid, the FP32 correction bias
is added for selection, each group's score is the sum of its strongest two
corrected experts, non-selected groups are masked, and top-k is taken from the
remaining corrected scores. The recovered ranking must reproduce the live
top-k set for every row before any near-route evidence is accepted.

Projection checkpoint tensor/manifest pairs are atomically published as
`0644`, even when the container runs as root, so the host-side independent
validator can read them without changing ownership. Older in-progress images
that published `0600` pairs may be repaired after the container stops with a
single root-side `chmod a+r` over files in the checkpoint store; permissions
are not part of checkpoint content identity.

An already-complete projection store may seed a fresh run with
`--projection-checkpoint-seed-dir /path/to/store`. The quantizer hashes every
seed artifact, verifies complete signed coverage, and accepts only an exact
numerical-family match (or a specifically reviewed, allowlisted
checkpoint-only GPTQModel transition). Never copy individual projection files
by hand.

For an allowlisted transition, `family_join` remains the seed's numerical
compatibility identity in both reused and newly written projection keys. It is
not the identity of the process that produced a new checkpoint. The actual
GPTQModel revision and source-tree hash, image digest, GPU identities, and
preflight are recorded under `provenance.run.coordinator` and bound by the
immutable plan hash. If an execution upgrade is active, new projection records
also carry `provenance.run.execution_upgrade`, while the complete signed
upgrade chain is retained beside the plan. This separation permits numerically
identical projection reuse without losing executable provenance.

The rolling boundary stores the activation continuation and router
`prev_topk_indices` separately, while completed packed projections remain in
the projection store. The first unseeded projection also builds GPTQModel's
EXL3 CUDA extension into `run-state/jit/exllamav3`; the loader fingerprints its
sources, compiler flags, Python/Torch/CUDA versions, and target architectures,
so later containers reuse only a compatible binary. Publication is atomic: the
final model directory is created only after all 78 decoder layers have
completed and the deferred EXL3 modules have been materialized.

Deferred packed modules do not create a second copy of the 272.73 GB projection
payload. Their offload directories contain only small indices that reference
the already authenticated projection-checkpoint tensor ranges and bind each
range by SHA-256. The final streaming writer verifies those hashes while
copying the ranges into the publication shards. Projection checkpoints must
therefore remain in place until the artifact has been completely written and
validated; cleanup tooling enforces that ordering.

Every new v2 plan also carries a deterministic storage contract. Before model
loading, the runner groups output, rolling state, offload, and projection
requirements by the actual filesystem device and writes
`storage-preflight.json`. For the pinned source it budgets approximately
329.85 GB of tensor payload for the final artifact, 272.73 GB for durable K3
projection payloads, 128 GiB for bounded rolling state, 12 GiB for offload,
format overhead, and a 32 GiB post-completion free-space floor. Existing
completed projections and rolling files reduce only their corresponding
remaining requirement. Output and run state must share a filesystem because
publication is one atomic rename.

## Tests

The host-side corpus, plan, seed, boundary, and resume contracts are covered by:

```bash
PYTHONPATH=quantization pytest -q quantization/tests
```

Run these tests in the pinned image when the host does not have the locked
Python dependencies. Native Spark execution and numerical parity are qualified
separately after the artifact is complete; a successful quantization alone is
not permission to publish or replace the NVFP4 serving baseline.

After building the Spark native library on an SM120/SM121 host, validate the
production C ABI and every route-block regime against the pinned SparkInfer
implementation:

```bash
PYTHONPATH=third_party/sparkinfer \
python3 python/tools/validate_b12x_exl3_native.py \
  --native-library /path/to/libglmrt_native.so \
  --rows 1,3,129,257,513,1025,2049,2064
```

The validator uses the production GLM-5.2 TP4 geometry, passes inputs through
GLMRT's NVFP4 wire codec, requires a nonzero finite oracle, and reports the
compiled tile, register, spill, and numerical-difference evidence. It is a
runtime integration gate, not a substitute for held-out model-quality
qualification.

Before final model publication, the same validator must assemble all four TP
ranks directly from one complete calibrated layer in the resumable projection
store. Each pass authenticates the selected layer's 768 manifests and packed
payloads before loading them:

```bash
native_evidence_root=/path/to/native-evidence
mkdir -p "$native_evidence_root"
for tp_rank in 0 1 2 3; do
  PYTHONPATH=third_party/sparkinfer \
  python3 python/tools/validate_b12x_exl3_native.py \
    --native-library /path/to/libglmrt_native.so \
    --projection-checkpoint-dir /path/to/projection-checkpoints \
    --layer-id 3 --tp-rank "$tp_rank" \
    --rows 1,3,9,10,129,257,513,1025,2049,2064 \
    --output "$native_evidence_root/native-tp${tp_rank}.json"
done
```

This reads all 768 `trellis/suh/svh/mcg` projection checkpoints for that layer,
validates their exact namespace, shapes, manifest digests, and packed-payload
hashes, performs the production TP4 intermediate-axis slicing, and compares
the native ABI with the pinned SparkInfer implementation. The M=2,049 and
M=2,064 cases qualify the non-power-of-two bucket that retains a full prefill
wave's target/draft suffix. The selected quantizer global scale is already
folded inversely into `suh`; the runtime unit-scale buffer must remain 1.0.

## Accept and stage the completed artifact

The quantizer publishes locally with one same-filesystem atomic rename. Before
uploading or serving that directory, authenticate and summarize all projection
quality evidence. Final qualification must hash the packed payloads; the two
diagnostic flags shown below are only useful while a quantization is running:

```bash
quant_evidence_report="$state_root/glm52-exl3-quant-evidence.json"

python3 python/tools/validate_glm52_exl3_quant_evidence.py \
  --plan "$run_state/glmrt-gptqmodel-plan.json" \
  --projection-checkpoint-dir "$projection_root" \
  --error-journal "$run_state/.glmrt-exl3-error-journal.jsonl" \
  --output "$quant_evidence_report"

# In-progress inspection only; never use this report for release acceptance:
#   --allow-incomplete --skip-tensor-hashes
```

The default mode requires all 57,600 base routed-expert projections, exact
journal membership, content-bound plans/manifests/requests, correct K3 tensor
geometry, all finite and arithmetically consistent Hessian/reconstruction
metrics, and SHA-256 verification of every packed checkpoint. The signed report
also validates any complete execution-upgrade chain and accounts for
projection records produced by the parent plan and each upgraded executor. It
contains global, per-projection, and per-layer error distributions. It is
quantizer evidence, not an end-to-end model-quality result; held-out serving
qualification remains mandatory.

Then run the independent artifact validator:

The original v1 production run started before tokenizer files were included in
the immutable plan. Its one-time recovery evidence must use the retained
original container metadata and that exact pinned image:

```bash
docker inspect glmrt-exl3-k3-quant-safe-v6-full-1 \
  > "$state_root/original-container-inspect.json"

source_model_root="$(dirname "$(dirname "$source_snapshot")")"
repo_root="$(git rev-parse --show-toplevel)"
docker run --rm --network none --ipc=none \
  -v "$repo_root:$repo_root:ro" \
  -v "$source_model_root:$source_model_root:ro" \
  -v "$(dirname "$corpus_root"):$(dirname "$corpus_root"):ro" \
  -v "$(dirname "$state_root"):$(dirname "$state_root")" \
  "$quant_image" \
  python "$repo_root/python/tools/attest_glm52_quant_tokenizer.py" \
    --plan "$run_state/glmrt-gptqmodel-plan.json" \
    --snapshot "$source_snapshot" \
    --calibration-jsonl "$corpus_root/calibration.jsonl" \
    --container-inspect "$state_root/original-container-inspect.json" \
    --output "$state_root/tokenizer-attestation.json"
```

The attestation hashes the canonical tokenizer blobs and the ordered prepared
token/attention-mask stream, binds the original image/container/plan/corpus,
and proves both blobs predated container launch. This recovery step is not used
by new plans.

```bash
artifact_report="$state_root/glm52-exl3-artifact-validation.json"
tokenizer_attestation="$state_root/tokenizer-attestation.json"

python3 python/tools/validate_glm52_exl3_artifact.py \
  --artifact "$output_root" \
  --source-snapshot "$source_snapshot" \
  --projection-checkpoint-dir "$projection_root" \
  --tokenizer-attestation "$tokenizer_attestation" \
  --verify-artifact-file-hashes \
  --output "$artifact_report"
```

This proves the exact replacement of 57,600 base routed-expert projections by
230,400 K3 `trellis/suh/svh/mcg` tensors, rejects retained native copies of
those weights, checks the external and embedded EXL3 declarations, validates
all safetensors headers and bound run manifests, and byte-compares every
retained native tensor with the pinned source. It also byte-compares every
packed artifact tensor with its plan-bound calibrated projection checkpoint
and emits the same checkpoint-inventory identity as the independent quant
evidence validator. An upgraded run must expose the same active execution
upgrade in both reports. It streams data and does not materialize a model.
`--skip-retained-native-bytes` is diagnostic only and its report is not
accepted by the cache stager.

The `--tokenizer-attestation` argument is required for the in-progress v1
production run because its immutable plan predates direct tokenizer-file
binding. Capture the signed report from the retained original container inspect
record with `attest_glm52_quant_tokenizer.py`; new plans bind both tokenizer
files directly and must omit this legacy argument.

Stage the accepted internal artifact under its production model ID without
duplicating the tensor payload on the coordinator. This stage is for runtime
qualification; it still contains private recovery/provenance files and is not
the Hub upload tree:

```bash
python3 python/tools/stage_glm52_exl3_hf_snapshot.py \
  --artifact "$output_root" \
  --validation-report "$artifact_report" \
  --quant-evidence-report "$quant_evidence_report" \
  --model-id wrldsuksgo2mars/GLM-5.2-EXL3-K3-calibrated-v1 \
  --update-ref
```

The default hardlink mode requires the artifact and Hugging Face cache to be
on the same filesystem. It creates the standard blob/snapshot/symlink layout
with a content-derived local revision, so the candidate can be exercised by
the ordinary release and WIP paths before Hub publication. Use
`--link-mode copy` only when the extra artifact-sized storage is intentional.

Distribute the selected snapshot to all Sparks concurrently over RDMA and
verify every received blob before launch:

```bash
python3 python/tools/sync_glm52_exl3_hf_snapshot.py \
  --hosts ostrich,dodo,emu,kiwi \
  --output "$state_root/glm52-exl3-spark-sync.json"
```

### Retune the AOT buckets on the completed model

The source-time Spark sweep is provisional. Once the accepted EXL3 artifact is
staged on all four Sparks, perform one final model-backed tuning pass before
the paired serving qualification or publication. Use the final GLMRT and
SparkInfer build and the actual balanced-profile scheduler; do not infer this
profile from uniformly random routes or from a sweep that varies only global
`M`.

Capture the per-layer, per-rank live row count and complete expert route-count
vector for representative decode, speculative verify, concurrency, and
prefill requests. Retain request type and frequency so replay can minimize the
frequency-weighted full-system cost rather than choosing every isolated
microbenchmark minimum. The capture must include the transition bucket around
the 2,048-row prefill boundary and the target/draft suffix through 2,064 rows.

Use a unique capture identity because WIP process logs are append-only. The
main statistics trace is sufficient; the much larger per-row route trace is
not required. Start the final EXL3 deployment with the trace variables, run
the frozen tuning workload, then bind the captured records to that deployment:

```bash
route_capture_id=glm52-exl3-final-balanced-v1
route_capture_gate=/wip/run/glm52-exl3-final-balanced-v1.enabled
export GLMRT_PROTOCOL_V2_EXPERT_QUEUE_STATS=1
export GLMRT_PROTOCOL_V2_EXPERT_QUEUE_CAPTURE_ID="$route_capture_id"
export GLMRT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES_GATE_FILE="$route_capture_gate"
./run.sh --wip --wip-slot exl3-paired \
  --profile qualification-exl3.config --restart

# Open the gate only after startup/warmup, run the frozen tuning corpus, then
# close it before any unrelated requests can contaminate the frequency mix.
docker exec glrmt-coordinator-wip touch "$route_capture_gate"
# Run the decode/speculation/concurrency/prefill tuning corpus here.
docker exec glrmt-coordinator-wip \
  mv "$route_capture_gate" "$route_capture_gate.closed"

route_log="$state_root/glm52-exl3-route-capture.log"
docker exec glrmt-coordinator-wip \
  cat /wip/run/coordinator-8000.log >"$route_log"

python3 python/tools/analyze_glm52_exl3_route_profile.py \
  --log balanced="$route_log" \
  --deployment .glmrt-wip/run/deployment.json \
  --capture-id "$route_capture_id" \
  --output "$state_root/glm52-exl3-route-profile.json"

unset GLMRT_PROTOCOL_V2_EXPERT_QUEUE_STATS
unset GLMRT_PROTOCOL_V2_EXPERT_QUEUE_CAPTURE_ID
unset GLMRT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES_GATE_FILE
```

The analyzer accepts only the versioned trace contract, requires complete
layers 3–77 and all four replicated TP ranks, verifies each route-count vector
sums to `M * 8`, and rejects rank disagreement. Its report hashes the complete
input log and deployment evidence. Each sample records both M and the observed
expert-reuse distribution, including active experts, maximum reuse R, and the
padded work implied by every legal route-block width. Pass a selected fixture
directly to
`validate_b12x_exl3_native.py --route-profile ... --route-profile-sample N`;
the native validator authenticates the report and realizes the same expert
reuse degrees while alternating production and candidate capacities in one
process.

The benchmark-only native entry point also accepts `--compare-grid-x N`, with
an optional `--compare-capacity-rows M`, so each observed fixture can sweep
the persistent grid independently of its exported tile/capacity. Candidate
grids are bounded by that compiled bucket's safe cooperative-grid maximum;
serving never reads these controls. Capacity comparisons that cross a
SparkInfer route-block ABI boundary are rejected rather than timing stale
packed-route metadata.

Replay those observed route fixtures in one process against the production
capacity and each legal candidate capacity/tile/grid. In particular, recheck
the provisional exact-M=9 EXL3 specialization against its M=16 parent, every
large bucket used by the captures, and M=2,048 versus M=2,064. Alternate the
candidate order, report multiple medians, and include packing plus fused FC1,
activation, and FC2 time. A candidate wins only when its weighted improvement
survives repeated same-process measurement and it does not create a material
tail regression for a common route shape.

Bake only those measured winners into
`python/tools/_b12x_exl3_k3_profile.py`, the exported AOT regimes, and the
matching Rust/native capacity selector. The checkpoint-native trellis layout
remains the sole resident weight format; bucket tuning must not introduce dual
weight residency. Build the selected profile once, then rerun all-four-rank
calibrated native parity at every capacity boundary, including 2,049 and
2,064, before collecting the paired end-to-end results below. Preserve the
route corpus, raw timing reports, model revision, WIP deployment identity,
SparkInfer revision, power limits, and selected profile with the qualification
evidence so the final choices can be reproduced.

After staging, the remaining acceptance order is: Rust catalog validation,
all-four-rank preload and basic generation, the completed-model AOT retuning
above, held-out quality, matched native versus EXL3 decode/prefill
measurements, and the existing tool-call suite.
The general release benchmarks do not need to be regenerated. The following
is one fresh, paired qualification of this EXL3 artifact against NVFP4: both
arms must use the same GLMRT/SparkInfer build, balanced profile, 400 W power
limit, tokenizer, prompt seeds, and frozen prefill corpus. Use a new output
directory and run each command once against the NVFP4 deployment and once
against the EXL3 deployment:

Build one WIP slot only once, then launch both model arms from that same slot
with explicit complete configuration files. The files should be identical
except for `MODEL=luke` versus `MODEL=exl3`; do not rebuild or clone a second
slot between arms:

```bash
./wip.sh --slot exl3-paired
./run.sh --wip --wip-slot exl3-paired \
  --profile qualification-nvfp4.config --restart
# collect the NVFP4 arm, then:
./run.sh --wip --wip-slot exl3-paired \
  --profile qualification-exl3.config --restart
```

This makes the required coordinator/Spark slot fingerprints and SparkInfer
revision identical while preserving the distinct model revision and
model-sensitive expert-runtime fingerprint for each arm.

```bash
qualification_root="$state_root/serving-qualification"
mkdir -p "$qualification_root"

# Substitute nvfp4/exl3 and the corresponding model ID for each deployment.
arm=nvfp4
model=lukealonso/GLM-5.2-NVFP4
nonce_seed=2026082301
prefill_run_id=glm52-exl3-k3-paired-v1
frozen_corpus=/path/to/unchanged/prefill-corpus
tokenizer=/path/to/the/same/tokenizer.json

# Immediately after ./run.sh --wip reports ready for this arm, preserve its
# content-bound binary/model/runtime identity before launching the other arm.
cp .glmrt-wip/run/deployment.json \
  "$qualification_root/$arm-deployment.json"
expert_runtime_fingerprint="$(
  jq -r .fingerprints.expert_runtime \
    "$qualification_root/$arm-deployment.json"
)"

python3 python/tools/bench_real_full_mtp_acceptance.py \
  --model "$model" --suite weighted --repeats 5 \
  --nonce-seed "$nonce_seed" \
  > "$qualification_root/$arm-blended.jsonl"

python3 python/tools/bench_real_full_repeat_decode.py \
  --model "$model" --word orchid --count 100 --max-tokens 1500 \
  --warmups 1 --repeats 5 --nonce-seed "$nonce_seed" \
  --tokenizer "$tokenizer" \
  --output "$qualification_root/$arm-orchid.jsonl"

python3 python/tools/bench_release_prefill_matrix.py \
  --model "$model" --profile balanced --run-id "$prefill_run_id" \
  --tokenizer "$tokenizer" --corpus-root "$frozen_corpus" \
  > "$qualification_root/$arm-prefill.jsonl"

tool-eval-bench \
  --base-url http://127.0.0.1:8000/v1/ --parallel 1 \
  --model "$model" --json-file "$qualification_root/$arm-tool-eval.json"

# Preserve each append-only Spark log before switching arms. The analyzer
# selects the final `starting expert-*` segment and accepts a later orderly
# container stop, but copying while the WIP containers are live is simplest.
for host in ostrich dodo emu kiwi; do
  ssh "$host" docker exec glrmt-spark-expert-wip \
    cat /wip/run/expert-9100.log \
    > "$qualification_root/$arm-$host-expert.log"
done

weight_format=nvfp4  # use exl3 for the candidate arm
python3 python/tools/analyze_glm52_expert_startup.py \
  --model "$model" --weight-format "$weight_format" --cache-state cold \
  --expert-runtime-fingerprint "$expert_runtime_fingerprint" \
  --log ostrich="$qualification_root/$arm-ostrich-expert.log" \
  --log dodo="$qualification_root/$arm-dodo-expert.log" \
  --log emu="$qualification_root/$arm-emu-expert.log" \
  --log kiwi="$qualification_root/$arm-kiwi-expert.log" \
  --output "$qualification_root/$arm-startup.json"
```

`tool-eval-bench --version` must report
`2.3.2.dev3+g5df1e9e0c`, the build used for the accepted NVFP4 benchmark. The
benchmark files bind exact prompts independently of the selected model. The
startup analyzer also requires every Spark log to carry the exact expert
runtime fingerprint preserved in that arm's WIP deployment evidence. The
validator rejects different prompt sequences, tokenizer bytes, corpus bytes,
tool scenarios, sampling settings, runtime correctness failures, runtime
fingerprint mismatches, or runtime graph captures before comparing performance.
Every EXL3 candidate response must also pass the prompt-specific,
non-executing semantic contract (Python AST and requested assertions,
arithmetic result, word/bullet constraints, bare JSON edit shape, and
multilingual fork/page coverage as applicable). Baseline misses are retained
in the signed report instead of invalidating an otherwise usable paired
reference; they never excuse a candidate miss:

```bash
serving_qualification_report="$state_root/glm52-exl3-serving-qualification.json"

python3 python/tools/validate_glm52_exl3_serving_qualification.py \
  --artifact "$output_root" \
  --artifact-validation "$artifact_report" \
  --quant-evidence "$quant_evidence_report" \
  --baseline-blended "$qualification_root/nvfp4-blended.jsonl" \
  --candidate-blended "$qualification_root/exl3-blended.jsonl" \
  --baseline-repeat "$qualification_root/nvfp4-orchid.jsonl" \
  --candidate-repeat "$qualification_root/exl3-orchid.jsonl" \
  --baseline-prefill "$qualification_root/nvfp4-prefill.jsonl" \
  --candidate-prefill "$qualification_root/exl3-prefill.jsonl" \
  --baseline-tool-eval "$qualification_root/nvfp4-tool-eval.json" \
  --candidate-tool-eval "$qualification_root/exl3-tool-eval.json" \
  --baseline-startup "$qualification_root/nvfp4-startup.json" \
  --candidate-startup "$qualification_root/exl3-startup.json" \
  --baseline-deployment "$qualification_root/nvfp4-deployment.json" \
  --candidate-deployment "$qualification_root/exl3-deployment.json" \
  --candidate-native-validation "$native_evidence_root/native-tp0.json" \
  --candidate-native-validation "$native_evidence_root/native-tp1.json" \
  --candidate-native-validation "$native_evidence_root/native-tp2.json" \
  --candidate-native-validation "$native_evidence_root/native-tp3.json" \
  --output "$serving_qualification_report"
```

Keep the four native reports available through model-card rendering and public
snapshot preparation. The serving qualifier records their exact byte hashes,
TP-rank coverage, calibrated-layer inventory, tested native-library identity,
and mandatory row regimes. Both downstream publication tools re-read those
reports and reject a missing, modified, duplicated-rank, or summary-only native
gate; setting `native_kernel_parity` by hand is not sufficient.

The default release thresholds require no blended or orchid decode regression,
at least 95% of NVFP4 speculative acceptance, at least 95% of NVFP4 prefill in
every measured cell, and at least 98% of NVFP4 tool-evaluation points. Any
EXL3 expert resident preload and full expert service handoff must also be no
slower than matched NVFP4 startup. Any deliberate tradeoff must be made
explicit by changing the corresponding threshold argument; the selected
threshold and the losing cells remain in the content-bound report.

The calibrated-v1 publication used an explicit decode-optimized policy after
also preserving the default-policy rejection report. Its floors are 94% of
NVFP4 speculative acceptance and 79% of NVFP4 throughput in every prefill
cell. The measured acceptance ratio was 94.84%; geometric-mean prefill was
98.09%, but the worst 1K-suffix cells at 32K and 131K context were about
79.6%. This tradeoff accompanied 1.062x weighted decode, 1.111x Orchid repeat,
91 versus 87 tool score, 35/35 candidate semantic contracts, and 0.709x expert
startup. These relaxed floors are model/publication-specific; the validator's
95% defaults remain unchanged.

The deployment records additionally require both arms to use the same exact
coordinator and Spark WIP artifacts, SparkInfer revision, profile, dSpark mode,
and coordinator power limit. They bind each arm to its exact staged model
revision and model-sensitive expert-runtime fingerprint, avoiding the weaker
and incorrect practice of labeling a dirty WIP build with only `git rev-parse
HEAD`.

The orchid workload is a low-entropy performance probe, not a counting-quality
test. GLM-5.2 can overshoot the requested count even in the NVFP4 baseline, so
qualification records exactness but accepts a word-occurrence count from 80%
through 125% of the request. Prompt identity, tokenization, zero runtime graph
capture, and matched-arm decode timing remain mandatory. Semantic instruction
following is gated separately by the weighted contracts and tool evaluation.

After those gates pass, render the model card mechanically from the exact
accepted reports, then prepare a standard-only publication tree:

```bash
public_root="$output_root-publication"
publication_report="$state_root/glm52-exl3-publication.json"
rendered_model_card="$state_root/GLM52-EXL3-README.md"

python3 python/tools/render_glm52_exl3_model_card.py \
  --template quantization/GLM52_EXL3_MODEL_CARD.md \
  --artifact-validation "$artifact_report" \
  --quant-evidence "$quant_evidence_report" \
  --serving-qualification "$serving_qualification_report" \
  --output "$rendered_model_card"

python3 python/tools/prepare_glm52_exl3_hf_publication.py \
  --artifact "$output_root" \
  --source-snapshot "$source_snapshot" \
  --validation-report "$artifact_report" \
  --quant-evidence-report "$quant_evidence_report" \
  --serving-qualification-report "$serving_qualification_report" \
  --readme "$rendered_model_card" \
  --output "$public_root" \
  --report "$publication_report"
```

The renderer refuses unsigned, incomplete, rejected, or cross-bound evidence.
The publication builder additionally requires the rendered card to cite the
exact serving-report hash, so qualification numbers cannot be copied from a
different artifact or run.

The public tree hardlinks only the weight shards, copies standard tokenizer
and license files, emits compact public configuration, and excludes local
plans, recovery journals, quantization logs, and hardware paths. Re-stage and
re-sync that exact tree for a final generation smoke test:

```bash
python3 python/tools/stage_glm52_exl3_hf_snapshot.py \
  --artifact "$public_root" \
  --publication-report "$publication_report" \
  --update-ref

python3 python/tools/sync_glm52_exl3_hf_snapshot.py \
  --hosts ostrich,dodo,emu,kiwi
```

Upload only `$public_root`, then resolve and verify the exact public Hub
revision. The verifier force-downloads and hashes every file up to 64 MiB
(including the compact configuration and model card), checks the remote LFS
SHA-256 for larger shards, and rejects any missing or unexpected file:

```bash
hf upload wrldsuksgo2mars/GLM-5.2-EXL3-K3-calibrated-v1 \
  "$public_root" . --no-private \
  --commit-message "Publish calibrated GLM-5.2 EXL3 K3"

hub_verification_report="$state_root/glm52-exl3-hub-verification.json"
python3 python/tools/verify_glm52_exl3_hub_publication.py \
  --publication-report "$publication_report" \
  --revision main \
  --output "$hub_verification_report"

hub_revision="$(jq -r .resolved_revision "$hub_verification_report")"
```

Retain `hub_revision` with the final reports. The earlier standard-tree staging
and all-four-Spark smoke test prove the same publication bytes load in GLMRT;
the remote verifier proves that exact byte inventory is what the public Hub
revision exposes without downloading another full copy of every shard.

Keep the projection checkpoint store until the published model has passed the
complete serving and quality ladder. Once the complete artifact and quant
evidence are accepted, the rolling layer boundary, capture frontier, active
source, post-quant replay, JIT, and offload state are regenerable and may be
removed, but retain the immutable plan, artifact/run manifests, compact error
evidence, validation reports, and final model. Projection-checkpoint deletion
remains an explicit user decision.

Use the content-bound cleanup planner instead of deleting run directories by
hand. It is dry-run by default. After the complete artifact and quant-evidence
reports are accepted, review and then release only regenerable transient state:

```bash
transient_cleanup_report="$state_root/glm52-exl3-transient-cleanup.json"
python3 python/tools/cleanup_glm52_exl3_quant_state.py \
  --artifact-validation "$artifact_report" \
  --quant-evidence "$quant_evidence_report" \
  --output "$transient_cleanup_report"

# After reviewing the exact targets in the planned report:
python3 python/tools/cleanup_glm52_exl3_quant_state.py \
  --artifact-validation "$artifact_report" \
  --quant-evidence "$quant_evidence_report" \
  --execute \
  --output "$transient_cleanup_report"
```

Only after the remote Hub verifier accepts the exact public revision may the
current run's projection store be released:

```bash
checkpoint_cleanup_report="$state_root/glm52-exl3-checkpoint-cleanup.json"
python3 python/tools/cleanup_glm52_exl3_quant_state.py \
  --artifact-validation "$artifact_report" \
  --quant-evidence "$quant_evidence_report" \
  --publication-report "$publication_report" \
  --hub-verification "$hub_verification_report" \
  --release-projection-checkpoints \
  --output "$checkpoint_cleanup_report"

# Review first, then repeat with --execute.
```

The utility derives every target from the signed plan, refuses broad or
overlapping roots, never follows symlinks, and does not remove the immutable
plan/reports/final model. A separately supplied projection seed is deliberately
excluded because another run may still reference it. If root-owned state cannot
be removed as the host user, rerun only the reviewed `--execute` command with
`sudo`.
