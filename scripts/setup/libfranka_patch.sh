#!/usr/bin/env bash
#
# Patch a droid checkout so polymetis' Franka client speaks the same FCI
# protocol version as the robot's control server.
#
# Symptom this fixes:
#   terminate called after throwing an instance of 'franka::IncompatibleVersionException'
#     what():  libfranka: Incompatible library version (server version: 9, library version: 6).
#
# The server version comes from the robot's firmware and is not something we
# control; the library version is decided by which libfranka release we build.
# Rebuilding libfranka alone is NOT enough -- polymetis' franka_panda_client
# links against libfranka's soname, so it must be relinked afterwards or it
# keeps reporting the old version.
#
# Usage:
#   ./patch_libfranka_fci.sh                  # target protocol 9 (default)
#   ./patch_libfranka_fci.sh --protocol 8     # target a different server version
#   ./patch_libfranka_fci.sh --version 0.16.1 # pin an exact libfranka release
#   ./patch_libfranka_fci.sh --check          # report state, change nothing
#   ./patch_libfranka_fci.sh --repo ~/droid   # explicit repo root
#
# Env overrides: CONDA_ENV (default polymetis-local), REPO_ROOT

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-polymetis-local}"
REPO_ROOT="${REPO_ROOT:-}"
TARGET_PROTOCOL=9
PIN_VERSION=""
CHECK_ONLY=0
PATCH_LAUNCH=1

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --protocol) TARGET_PROTOCOL="$2"; shift 2 ;;
    --version)  PIN_VERSION="$2";     shift 2 ;;
    --repo)     REPO_ROOT="$2";       shift 2 ;;
    --env)      CONDA_ENV="$2";       shift 2 ;;
    --check)    CHECK_ONLY=1;         shift   ;;
    --no-launch-patch) PATCH_LAUNCH=0; shift  ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

# ---------------------------------------------------------------------------
# FCI protocol version -> lowest libfranka release implementing it.
#
# Read off `research_interface::robot::kVersion` in each release (it lives in
# the `common` submodule from 0.11 onward). Lowest match is preferred: it is
# the closest API to what polymetis' client was written against (0.9/0.10).
# ---------------------------------------------------------------------------
protocol_to_libfranka() {
  case "$1" in
    6) echo "0.10.0" ;;
    7) echo "0.13.3" ;;
    8) echo "0.14.0" ;;
    9) echo "0.15.0" ;;
    10) echo "0.18.0" ;;
    *) die "no known libfranka release for FCI protocol version '$1'" ;;
  esac
}

# ---------------------------------------------------------------------------
# Locate the repo
# ---------------------------------------------------------------------------
if [ -z "$REPO_ROOT" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel 2>/dev/null || true)"
fi
[ -n "$REPO_ROOT" ] || die "could not determine repo root; pass --repo <path>"

POLYMETIS="$REPO_ROOT/droid/fairo/polymetis"
LIBFRANKA="$POLYMETIS/polymetis/src/clients/franka_panda_client/third_party/libfranka"
BUILD_DIR="$POLYMETIS/polymetis/build"
CLIENT_BIN="$BUILD_DIR/franka_panda_client"

[ -d "$LIBFRANKA" ] || die "libfranka submodule not found at $LIBFRANKA (run: git submodule update --init --recursive)"
[ -d "$BUILD_DIR" ] || die "polymetis build dir not found at $BUILD_DIR (build polymetis first)"

LIBFRANKA_VER="${PIN_VERSION:-$(protocol_to_libfranka "$TARGET_PROTOCOL")}"

log "repo root:        $REPO_ROOT"
log "conda env:        $CONDA_ENV"
log "target protocol:  $TARGET_PROTOCOL"
log "libfranka target: $LIBFRANKA_VER"

# ---------------------------------------------------------------------------
# Report current state
# ---------------------------------------------------------------------------
current_tag="$(git -C "$LIBFRANKA" describe --tags 2>/dev/null || echo '<unknown>')"
log "libfranka currently checked out: $current_tag"

if [ -x "$CLIENT_BIN" ]; then
  linked="$(ldd "$CLIENT_BIN" 2>/dev/null | grep -oE 'libfranka\.so\.[0-9.]+' | head -1 || true)"
  log "franka_panda_client links against: ${linked:-<none / unresolved>}"
else
  warn "franka_panda_client not built yet at $CLIENT_BIN"
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
  log "--check given; stopping without making changes."
  exit 0
fi

# ---------------------------------------------------------------------------
# Activate the conda env (cmake and the build deps live there, not in base)
# ---------------------------------------------------------------------------
if [ -n "${CONDA_EXE:-}" ]; then
  CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
else
  CONDA_BASE=""
  for c in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/mambaforge" "$HOME/miniforge3" /opt/conda; do
    [ -f "$c/etc/profile.d/conda.sh" ] && CONDA_BASE="$c" && break
  done
fi
[ -n "$CONDA_BASE" ] || die "could not locate a conda installation; set CONDA_EXE"

# conda's own scripts trip over `set -u`
set +u
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || die "could not activate conda env '$CONDA_ENV'"
set -u
log "activated $CONDA_ENV ($(python --version 2>&1))"

command -v cmake >/dev/null 2>&1 \
  || die "cmake not found inside '$CONDA_ENV'. Install it: conda install -c conda-forge cmake"
log "cmake: $(cmake --version | head -1)"

# ---------------------------------------------------------------------------
# libfranka >= 0.15 pulls in fmt, and its exported FrankaConfig.cmake does a
# find_package(fmt REQUIRED) -- so downstream (polymetis) needs it visible too.
# ---------------------------------------------------------------------------
lowest() { printf '%s\n%s\n' "$1" "$2" | sort -V | head -1; }

if [ "$(lowest "$LIBFRANKA_VER" 0.15.0)" = "0.15.0" ]; then
  if [ -z "$(find "$CONDA_PREFIX" -maxdepth 4 -iname 'fmt*config.cmake' 2>/dev/null | head -1)" ]; then
    log "libfranka $LIBFRANKA_VER needs fmt; installing into $CONDA_ENV"
    conda install -y -c conda-forge fmt --freeze-installed \
      || die "fmt install failed. Retry manually: conda install -c conda-forge fmt"
  else
    log "fmt already present in $CONDA_ENV"
  fi
fi

# ---------------------------------------------------------------------------
# Make sure the target tag exists locally (older clones predate 0.15.x)
# ---------------------------------------------------------------------------
if ! git -C "$LIBFRANKA" rev-parse -q --verify "refs/tags/$LIBFRANKA_VER" >/dev/null; then
  log "tag $LIBFRANKA_VER not in local clone; fetching tags from origin"
  git -C "$LIBFRANKA" fetch --tags origin \
    || warn "tag fetch reported errors (submodule fetch failures here are usually harmless)"
  git -C "$LIBFRANKA" rev-parse -q --verify "refs/tags/$LIBFRANKA_VER" >/dev/null \
    || die "tag $LIBFRANKA_VER still not available after fetch"
fi

# ---------------------------------------------------------------------------
# Build libfranka, then RELINK the polymetis client against it.
# ---------------------------------------------------------------------------
log "building libfranka $LIBFRANKA_VER (clean rebuild; this wipes its build/ dir)"
( cd "$POLYMETIS" && ./scripts/build_libfranka.sh "$LIBFRANKA_VER" )

built_so="$(find "$LIBFRANKA/build" -maxdepth 1 -name 'libfranka.so.*.*' | head -1)"
[ -n "$built_so" ] || die "libfranka build produced no shared library"
log "built $(basename "$built_so")"

log "relinking franka_panda_client against the new libfranka"
cmake --build "$BUILD_DIR" --target franka_panda_client \
  || die "franka_panda_client failed to build against libfranka $LIBFRANKA_VER.
       polymetis' client was written against libfranka 0.9/0.10 and newer
       releases restructured the API. Capture the compile errors above --
       the client source is at
       $POLYMETIS/polymetis/src/clients/franka_panda_client/franka_panda_client.cpp"

# ---------------------------------------------------------------------------
# Verify the relink actually took
# ---------------------------------------------------------------------------
expected_soname="libfranka.so.$(echo "$LIBFRANKA_VER" | cut -d. -f1,2)"
actual_soname="$(ldd "$CLIENT_BIN" | grep -oE 'libfranka\.so\.[0-9.]+' | head -1 || true)"
if [ "$actual_soname" = "$expected_soname" ]; then
  log "verified: franka_panda_client -> $actual_soname"
else
  die "relink did not take: expected $expected_soname, got ${actual_soname:-<none>}"
fi

# ---------------------------------------------------------------------------
# Unrelated but adjacent: polymetis starts run_server under sudo, so the
# plain `pkill -9 run_server` in launch_robot.sh cannot kill a stale server
# and the next launch dies on "Port unavailable".
# ---------------------------------------------------------------------------
LAUNCH="$REPO_ROOT/droid/franka/launch_robot.sh"
if [ "$PATCH_LAUNCH" -eq 1 ] && [ -f "$LAUNCH" ]; then
  if grep -qE '^\s*pkill -9 run_server' "$LAUNCH"; then
    log "patching launch_robot.sh: pkill -> sudo pkill (stale server is root-owned)"
    sed -i \
      -e 's|^\(\s*\)pkill -9 run_server|\1sudo pkill -9 run_server|' \
      -e 's|^\(\s*\)pkill -9 franka_panda_cl|\1sudo pkill -9 franka_panda_cl|' \
      "$LAUNCH"
    if ! grep -q 'sport = :50051' "$LAUNCH"; then
      sed -i '/sudo pkill -9 franka_panda_cl/a # wait for port 50051 to be released before relaunching\nfor _ in $(seq 20); do ss -ltn '"'"'sport = :50051'"'"' 2>/dev/null | grep -q :50051 || break; sleep 0.5; done' "$LAUNCH"
    fi
  else
    log "launch_robot.sh already patched (or uses a different kill sequence)"
  fi
fi

cat <<EOF

$(log "done")

  libfranka   $LIBFRANKA_VER  (FCI protocol $TARGET_PROTOCOL)
  client      $CLIENT_BIN -> $actual_soname

Next:
  cd $REPO_ROOT/droid/franka && ./launch_robot.sh

If the robot still reports a version mismatch, read the numbers off the new
exception and re-run with the server version it names:
  $0 --protocol <server version>
EOF
