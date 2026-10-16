#!/bin/sh
# Test a toy pipeline on four procedural images, CPU only: prepare the masters, run pipeline
# versions v1 and v2 on them, evaluate both, compare. Usage: sh examples/demo.sh [WORKDIR]
set -e
here=$(cd "$(dirname "$0")" && pwd)
d=${1:-provsurvive-demo}
[ -e "$d" ] && { echo "$d exists; remove it or pass another WORKDIR" >&2; exit 1; }
mkdir -p "$d" && cd "$d"
python "$here/toy_pipeline.py" images images
provsurvive prepare images --out case --no-lpips
for v in v1 v2; do
  python "$here/toy_pipeline.py" $v case/masters processed-$v
  provsurvive evaluate case processed-$v --out report-$v --note $v --no-lpips
done
provsurvive compare report-v1 report-v2
