#!/usr/bin/env bash
# Run after the administrator has installed the host driver and source prerequisites.
set -euo pipefail
revision=180d922b2112b2dc17725036b3bae71852b43b6f
checkout=${1:-"$HOME/p150-tt-metal"}
if [[ ! -e "$checkout" ]]; then
    git init "$checkout"
    git -C "$checkout" remote add origin https://github.com/tenstorrent/tt-metal.git
    git -C "$checkout" fetch --depth 1 origin "$revision"
    git -C "$checkout" checkout --detach FETCH_HEAD
fi
cd "$checkout"
if [[ "$(git rev-parse HEAD)" != "$revision" ]]; then
    echo "This path contains a different revision. Choose a new personal checkout path." >&2
    exit 1
fi
git submodule update --init --recursive
export TT_METAL_HOME="$PWD"
if [[ ${2:-} == --checkout-only ]]; then
    exit 0
fi
./build_metal.sh
./create_venv.sh --env-dir "$TT_METAL_HOME/python_env"
