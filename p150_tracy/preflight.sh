#!/usr/bin/env bash
set -eu
: "${TT_METAL_HOME:?Set TT_METAL_HOME and activate its python_env first}"
echo "Linux and PCI devices"
uname -srmo
lspci -nn -d 1e52:
ls -l /dev/tenstorrent/
echo "Hugepage mounts and free 1 GiB pages"
findmnt -t hugetlbfs
cat /sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages
tt-smi --version
echo "Python and profiler from the active environment"
python -c 'import sys, torch, transformers, ttnn, tracy; print(sys.executable, torch.__version__, transformers.__version__, ttnn.__file__, tracy.__file__, sep="\n")'
python -m tracy --help | grep -E -- 'output-folder|device-memory-profiler|dump-device-data-mid-run|no-web-server'
for name in tracy-capture tracy-csvexport; do
    path="$TT_METAL_HOME/build/tools/profiler/bin/$name"
    [[ -x "$path" ]] || { echo "Missing Tracy executable: $path" >&2; exit 1; }
done
git -C "$TT_METAL_HOME" rev-parse HEAD
python "$(dirname "$0")/check.py" "$@"
