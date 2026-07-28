#!/usr/bin/env bash
set -euo pipefail

hosts_csv="${1:?hosts csv required}"
command_text="${2:?remote command required}"

IFS=',' read -r -a hosts <<< "$hosts_csv"
for host in "${hosts[@]}"; do
  echo "== $host =="
  ssh -o BatchMode=yes "$host" "$command_text"
done

