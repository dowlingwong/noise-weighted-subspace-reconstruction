# GWOSC stable summary (EXP-06)

Two SEPARATE metric classes -- do not merge into one status.

## Implementation identity (verified positive)
Confirms two independent code paths (explicit FFT convolution and GWpy FIR convolution) compute the SAME shared FIR statistic identically, for synthetic and real H1/L1 windows. This is an implementation-correctness check, NOT a calibration or detection-sensitivity result, and must never be reported alongside the calibration metrics below as a single combined 'passed' status.
All settings accepted: True

## Real-noise calibration (verified negative boundary result)
Real-noise calibration requires more than a correct PSD implementation and a predeclared local PSD model. Do not state that this method is calibrated on GWOSC data, and do not combine this status with the implementation-identity file above.
- Global PSD: H1 random median ratio 1.986, chronological 3.183; L1 random 1.263, chronological 2.964 -- all exceed the acceptance gate (ratio near 1 expected under correct calibration).
- Time-local PSD: Synthetic control (local_radius_64s) passed (accepted=True); the same predeclared primary local model failed on real H1 and L1 (accepted=False).
