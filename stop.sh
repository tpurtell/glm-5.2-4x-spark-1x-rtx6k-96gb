#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_root/scripts/release-common.sh"

usage() {
  cat <<'EOF'
Usage: ./stop.sh [--profile FILE]

Gracefully stops and removes the coordinator plus GLMRT expert containers on
the four Sparks selected by glmrt.config. Current release and exact legacy
GLMRT expert names are covered; images, model caches, build artifacts, and
unrelated containers are left untouched.

Despite the option name, --profile FILE selects an entire alternate
configuration file.
EOF
}

config="$repo_root/glmrt.config"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile|--config)
      config="${2:?$1 requires a configuration file}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      release_die "unknown stop argument: $1"
      ;;
  esac
done

release_load_config "$config"
release_need docker
release_need ssh
release_need ss
release_need ps

docker info >/dev/null 2>&1 ||
  release_die "local Docker daemon is unavailable"

echo "== stopping GLMRT release services =="
release_stop_services \
  "$RELEASE_COORDINATOR_CONTAINER_NAME" \
  "$RELEASE_SPARK_CONTAINER_PREFIX" ||
  release_die "one or more remote GLMRT services could not be stopped"
echo "GLMRT release services are stopped."
