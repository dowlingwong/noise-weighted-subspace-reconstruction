#!/bin/bash
W=/home/dwong/mlst-successor-closeout-20260917
T=$W/MLST_SUCCESSOR_TRANSFER_20260917.tar
PY=/home/dwong/mlst-successor-py312-20260913/bin/python
{
echo "start $(date -Is)"
cd /
nice -n 10 ionice -c3 tar --create --file "$T" --exclude='__pycache__' --format=pax \
  ceph/dwong/mlst-successor-runs-20260913 \
  var/tmp/mlst-t19-amendment-20260913/preflight-amended \
  var/tmp/mlst-t22-envrebuild-preflight-20260916/preflight-rebuilt-env \
  ceph/dwong/noise-weighted-subspace-reconstruction/experiments/mlst2026/successor_20260913/remote/MANIFEST.json \
  var/tmp/mlst-t19-amendment-20260913/exp/experiments/mlst2026/successor_20260913/remote/MANIFEST.json \
  var/tmp/mlst-t19-amendment-20260913/exp/experiments/mlst2026/successor_20260913/remote/MANIFEST_SUPPLIED_2026-09-13.json \
  var/tmp/mlst-t19-amendment-20260913/amendment.patch \
  var/tmp/mlst-t19-amendment-20260913/t19-transition \
  home/dwong/mlst-successor-audit-20260913.tar.gz \
  home/dwong/mlst-successor-audit-20260916-envloss \
  home/dwong/mlst-successor-closeout-20260917/bundle-staging
echo "TAR_EXIT=$?"
echo "tar done $(date -Is)"
PYTHONDONTWRITEBYTECODE=1 nice -n 10 $PY $W/verify_transfer.py
echo "VERIFY_EXIT=$?"
echo "end $(date -Is)"
} > $W/transfer_build.log 2>&1
touch $W/transfer.done
