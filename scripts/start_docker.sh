#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker CLI not found in PATH." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  if [[ "$(uname -s)" == "Darwin" ]] && [[ -d "/Applications/Docker.app" ]]; then
    echo "Docker daemon is not running. Launching Docker Desktop..."
    open -a Docker
    for _ in $(seq 1 45); do
      if docker info >/dev/null 2>&1; then
        break
      fi
      sleep 2
    done
  fi
fi

if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker daemon is not reachable. Start Docker Desktop and retry." >&2
  exit 1
fi

port_open() {
  local port="$1"
  nc -z 127.0.0.1 "$port" >/dev/null 2>&1
}

container_running() {
  local cname="$1"
  docker ps --format '{{.Names}}' | grep -qx "$cname"
}

SERVICES_TO_START=()
check_service() {
  local svc="$1"
  local cname="$2"
  local port="$3"

  if container_running "$cname"; then
    echo "[ok] $svc container ($cname) already running on port $port"
  elif port_open "$port"; then
    echo "[skip] port $port already in use by local $svc process"
  else
    SERVICES_TO_START+=("$svc")
  fi
}

check_service "postgres" "nexusai-postgres" 5432
check_service "redis" "nexusai-redis" 6379
check_service "qdrant" "nexusai-qdrant" 6333
check_service "neo4j" "nexusai-neo4j" 7687

if (( ${#SERVICES_TO_START[@]} > 0 )); then
  echo "Starting Docker services: ${SERVICES_TO_START[*]}"
  docker compose up -d "${SERVICES_TO_START[@]}"
fi

wait_for_port() {
  local name="$1"
  local port="$2"
  local attempts=30
  for _ in $(seq 1 "$attempts"); do
    if port_open "$port"; then
      echo "[ready] $name is listening on port $port"
      return 0
    fi
    sleep 2
  done
  echo "Warning: $name did not become reachable on port $port within $((attempts * 2))s" >&2
  return 1
}

wait_for_port "PostgreSQL" 5432
wait_for_port "Redis" 6379
wait_for_port "Qdrant" 6333
wait_for_port "Neo4j" 7687
