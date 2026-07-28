#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/configure-docker-nvidia-runtime.sh [--verify-only]

Installs and configures the NVIDIA Container Toolkit for Docker on the local
host, then verifies Docker GPU passthrough with the existing glmrt coordinator
image. Run configuration mode with sudo.

Environment:
  GLMRT_DOCKER_GPU_VERIFY_IMAGE       default: glmrt-dev:oliver
  GLMRT_NVIDIA_TOOLKIT_SETUP_APT_REPO auto, 0, or 1; default: auto
  NVIDIA_CONTAINER_TOOLKIT_VERSION    optional exact package version
EOF
}

mode="configure"
case "${1:-}" in
  "")
    ;;
  --verify-only)
    mode="verify"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

image="${GLMRT_DOCKER_GPU_VERIFY_IMAGE:-glmrt-dev:oliver}"
setup_apt_repo="${GLMRT_NVIDIA_TOOLKIT_SETUP_APT_REPO:-auto}"

case "$setup_apt_repo" in
  auto|0|1)
    ;;
  *)
    echo "GLMRT_NVIDIA_TOOLKIT_SETUP_APT_REPO must be auto, 0, or 1; got: $setup_apt_repo" >&2
    exit 2
    ;;
esac

toolkit_candidate_available() {
  local candidate
  candidate="$(apt-cache policy nvidia-container-toolkit 2>/dev/null | awk '/Candidate:/ {print $2; exit}')"
  [[ -n "$candidate" && "$candidate" != "(none)" ]]
}

toolkit_installed() {
  dpkg-query -W -f='${Status}\n' nvidia-container-toolkit 2>/dev/null | grep -qx 'install ok installed'
}

setup_nvidia_apt_repo() {
  apt-get update
  apt-get install -y --no-install-recommends ca-certificates curl gnupg2

  install -d -m 0755 /usr/share/keyrings /etc/apt/sources.list.d
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
}

install_toolkit() {
  if toolkit_installed; then
    echo "nvidia-container-toolkit is already installed."
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "This helper currently supports Debian/Ubuntu hosts with apt-get." >&2
    exit 1
  fi

  if [[ "$setup_apt_repo" == "1" ]] || { [[ "$setup_apt_repo" == "auto" ]] && ! toolkit_candidate_available; }; then
    setup_nvidia_apt_repo
  fi

  if ! toolkit_candidate_available; then
    echo "nvidia-container-toolkit has no apt candidate; set up NVIDIA's apt repo or set GLMRT_NVIDIA_TOOLKIT_SETUP_APT_REPO=1." >&2
    exit 1
  fi

  if [[ -n "${NVIDIA_CONTAINER_TOOLKIT_VERSION:-}" ]]; then
    apt-get install -y \
      "nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
      "nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
      "libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
      "libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}"
  else
    apt-get install -y nvidia-container-toolkit
  fi
}

restart_docker() {
  if command -v systemctl >/dev/null 2>&1; then
    systemctl restart docker
  elif command -v service >/dev/null 2>&1; then
    service docker restart
  else
    echo "Cannot restart Docker automatically; restart the Docker daemon before verifying." >&2
    exit 1
  fi
}

verify_gpu_passthrough() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not on PATH." >&2
    exit 1
  fi

  local runtimes
  runtimes="$(docker info --format '{{range $name, $_ := .Runtimes}}{{println $name}}{{end}}' | sort | xargs)"
  echo "Docker runtimes: ${runtimes:-<none>}"
  if ! grep -qw 'nvidia' <<<"$runtimes"; then
    echo "Docker does not report an nvidia runtime yet; GPU verification is expected to fail until the toolkit is configured." >&2
  fi

  if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "Verification image '$image' is not present locally; build it first or set GLMRT_DOCKER_GPU_VERIFY_IMAGE." >&2
    exit 1
  fi

  docker run --rm --gpus all "$image" \
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
}

if [[ "$mode" == "verify" ]]; then
  verify_gpu_passthrough
  exit 0
fi

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run configuration mode as root, for example: sudo scripts/configure-docker-nvidia-runtime.sh" >&2
  exit 77
fi

install_toolkit

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  echo "nvidia-ctk is still not on PATH after installing nvidia-container-toolkit." >&2
  exit 1
fi

nvidia-ctk runtime configure --runtime=docker
restart_docker
verify_gpu_passthrough
