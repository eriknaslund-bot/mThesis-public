#!/usr/bin/env python3
"""M sweep on the 3 extra AV2 logs (cheap pass, mirrors pass4_m_sweep
on the primary log). Output combined with results_m.csv gives 4-log
coverage for the §4.6 ablation table.

Output: eval/results_m_multilog.csv (rows tagged with both M and log).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_pipeline as ep

EVAL = ep.EVAL

if not ep.BIN.exists():
    raise SystemExit(f"binary not found: {ep.BIN} -- run `cd build && make` first")

all_rows = []
for log_path in ep.MULTILOG_DIRS:
    log_dir = ep.REPO / log_path
    if not log_dir.exists():
        print(f"  [skip] log dir missing: {log_dir}")
        continue
    log_tag = log_dir.name.split("-")[0]
    extra_log_flags = [
        "--calib",  str(log_dir / "calibration.json"),
        "--frames", str(log_dir / "frames.json"),
        "--lidar",  str(log_dir / "sensors" / "lidar"),
    ]
    for m in ep.M_SWEEP:
        if m == 1000:
            # Skip M=1000 on extra logs. On the primary log this value
            # produces garbage output (well-known ill-conditioning, kept
            # in tab:results_density as the failure point); on at least
            # one extra log it triggers a CUDA illegal-memory-access in
            # the warp kernel because the blown-up TPS weights produce
            # NaN coordinates. The §4.6 failure-point claim is already
            # made from the primary log; we don't need the crash data.
            print(f"  log={log_tag} M={m}: SKIP (known ill-conditioned)")
            continue
        tag = f"M_{m}__{log_tag}"
        try:
            rows = ep.run_metrics(
                tag,
                ["--max-ctrl-per-overlap", str(m), *extra_log_flags],
                ep.N_FRAMES_SWEEP,
            )
        except RuntimeError as e:
            print(f"  log={log_tag} M={m}: FAILED ({e}); continuing")
            continue
        for r in rows:
            r["M"] = str(m)
            r["log"] = log_tag
        print(f"  log={log_tag} M={m}: {len(rows)} frames")
        all_rows.extend(rows)

path = EVAL / "results_m_multilog.csv"
ep.write_csv(all_rows, path)
print(f"wrote {path} ({len(all_rows)} rows)")
