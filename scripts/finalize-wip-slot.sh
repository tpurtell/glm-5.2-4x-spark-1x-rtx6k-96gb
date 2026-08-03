#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: finalize-wip-slot.sh SOURCE_DIR ROLE SLOT BUILD_OUTPUT IMAGE IMAGE_ID" >&2
  exit 2
fi

source_dir="$(realpath "$1")"
role="$2"
slot="$3"
build_output="$(realpath "$4")"
image="$5"
image_id="$6"
[[ "$role" == coordinator || "$role" == spark-expert ]] || {
  echo "invalid WIP role: $role" >&2
  exit 2
}
[[ "$slot" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || {
  echo "invalid WIP slot: $slot" >&2
  exit 2
}
[[ -x "$build_output/glmrt" && -s "$build_output/libglmrt_native.so" ]] || {
  echo "WIP build output is incomplete: $build_output" >&2
  exit 2
}
(
  cd "$build_output"
  sha256sum -c ARTIFACT_SHA256SUMS
)

slot_root="/wip/slots/$slot/$role"
incoming="/wip/incoming/${slot}.${role}.$$"
rm -rf "$incoming"
mkdir -p "$incoming/workspace/.glmrt-wip"
cp -a "$source_dir/." "$incoming/workspace/"
install -m 0755 "$build_output/glmrt" "$incoming/workspace/.glmrt-wip/glmrt"
install -m 0755 \
  "$build_output/libglmrt_native.so" \
  "$incoming/workspace/.glmrt-wip/libglmrt_native.so"
install -m 0644 \
  "$build_output/ARTIFACT_SHA256SUMS" \
  "$incoming/workspace/.glmrt-wip/ARTIFACT_SHA256SUMS"

python3 "$source_dir/scripts/verify-release-source-manifest.py" \
  --source "$incoming/workspace" \
  --write "$incoming/SOURCE_SHA256SUMS"
source_manifest_sha256="$(sha256sum "$incoming/SOURCE_SHA256SUMS" | awk '{print $1}')"
artifact_sha256="$(sha256sum "$incoming/workspace/.glmrt-wip/ARTIFACT_SHA256SUMS" | awk '{print $1}')"
sparkinfer_revision="$(
  python3 "$source_dir/scripts/verify-sparkinfer-source.py" \
    --source "$source_dir/third_party/sparkinfer" \
    --lock "$source_dir/third_party/sparkinfer.lock.json" \
    --print-revision
)"

python3 - "$incoming/META.json" <<PY
import json
import pathlib
import time

path = pathlib.Path(__import__("sys").argv[1])
metadata = {
    "schema": 1,
    "slot": ${slot@Q},
    "role": ${role@Q},
    "base_image": ${image@Q},
    "base_image_id": ${image_id@Q},
    "source_manifest_sha256": ${source_manifest_sha256@Q},
    "artifact_manifest_sha256": ${artifact_sha256@Q},
    "sparkinfer_revision": ${sparkinfer_revision@Q},
    "built_unix": int(time.time()),
}
path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY
sha256sum "$incoming/META.json" | awk '{print $1}' >"$incoming/FINGERPRINT"

mkdir -p "$(dirname "$slot_root")"
rm -rf "$slot_root"
mv "$incoming" "$slot_root"
echo "WIP slot finalized: slot=$slot role=$role fingerprint=$(<"$slot_root/FINGERPRINT")"
