#!/usr/bin/env bash
#
# docker-publish.sh — build and push the CheetahClaws image to Docker Hub.
#
# Reads the version from pyproject.toml and tags the image with both that
# version and `latest`. Defaults to a multi-arch build (amd64 + arm64) pushed
# straight to the registry.
#
# Usage:
#   DOCKERHUB_USERNAME=youruser ./scripts/docker-publish.sh
#   ./scripts/docker-publish.sh youruser            # username as 1st arg
#
# Options (env vars):
#   DOCKERHUB_USERNAME   Docker Hub account/namespace (required)
#   IMAGE_NAME           Repository name         (default: cheetahclaws)
#   PLATFORMS            buildx target platforms (default: linux/amd64,linux/arm64)
#   SINGLE_ARCH=1        Build only the host arch with plain `docker build`
#                        + `docker push` instead of multi-arch buildx.
#   PUSH_LATEST=0        Skip the extra `latest` tag (push only the version).
#   DRY_RUN=1            Print the commands instead of running them.
#
set -euo pipefail

# --- Locate the repo root (this script lives in <repo>/scripts) ------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- Resolve inputs --------------------------------------------------------
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-${1:-}}"
IMAGE_NAME="${IMAGE_NAME:-cheetahclaws}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
SINGLE_ARCH="${SINGLE_ARCH:-0}"
PUSH_LATEST="${PUSH_LATEST:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${DOCKERHUB_USERNAME}" ]]; then
  echo "error: Docker Hub username not set." >&2
  echo "  set DOCKERHUB_USERNAME=<user> or pass it as the first argument." >&2
  exit 1
fi

# --- Read the version from pyproject.toml ----------------------------------
# Match:  version = "3.5.84"  (first occurrence under [project]).
VERSION="$(grep -m1 -E '^version[[:space:]]*=' pyproject.toml \
  | sed -E 's/^version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/')"

if [[ -z "${VERSION}" ]]; then
  echo "error: could not read version from pyproject.toml" >&2
  exit 1
fi

REPO="${DOCKERHUB_USERNAME}/${IMAGE_NAME}"
VERSION_TAG="${REPO}:${VERSION}"
LATEST_TAG="${REPO}:latest"

echo "==> Repository : ${REPO}"
echo "==> Version    : ${VERSION}"
echo "==> Platforms  : $([[ "${SINGLE_ARCH}" == "1" ]] && echo "host arch only" || echo "${PLATFORMS}")"
echo "==> Tags       : ${VERSION_TAG}$([[ "${PUSH_LATEST}" == "1" ]] && echo " , ${LATEST_TAG}")"
echo

run() {
  echo "+ $*"
  [[ "${DRY_RUN}" == "1" ]] || "$@"
}

# --- Check daemon + login (best effort; skipped in dry-run) ----------------
if [[ "${DRY_RUN}" != "1" ]]; then
  if ! docker info >/dev/null 2>&1; then
    echo "error: docker daemon not reachable (is Docker running?)." >&2
    exit 1
  fi
  if ! docker system info 2>/dev/null | grep -qi 'Username:'; then
    echo "note: you may not be logged in — run 'docker login' first if push fails." >&2
    echo
  fi
fi

# --- Build tag args --------------------------------------------------------
TAG_ARGS=(-t "${VERSION_TAG}")
[[ "${PUSH_LATEST}" == "1" ]] && TAG_ARGS+=(-t "${LATEST_TAG}")

if [[ "${SINGLE_ARCH}" == "1" ]]; then
  # Plain build for the host architecture, then push each tag.
  run docker build "${TAG_ARGS[@]}" .
  run docker push "${VERSION_TAG}"
  [[ "${PUSH_LATEST}" == "1" ]] && run docker push "${LATEST_TAG}"
else
  # Multi-arch build+push in one step (requires buildx; images can't be
  # `docker load`-ed locally, so --push is mandatory here).
  if ! docker buildx inspect cheetah-multiarch >/dev/null 2>&1; then
    run docker buildx create --name cheetah-multiarch --use
  else
    run docker buildx use cheetah-multiarch
  fi
  run docker buildx build \
    --platform "${PLATFORMS}" \
    "${TAG_ARGS[@]}" \
    --push .
fi

echo
echo "Done. Published:"
echo "  ${VERSION_TAG}"
[[ "${PUSH_LATEST}" == "1" ]] && echo "  ${LATEST_TAG}"
echo
echo "Pull with:  docker pull ${LATEST_TAG}"
