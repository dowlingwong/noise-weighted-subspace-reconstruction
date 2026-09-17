"""R6 instrument readiness: fail-closed interfaces and preflight only.

DATA-BLOCKED.  This package prepares the CRESST raw-pulse robustness grid and the
matched TIDMAD whole-window/narrowband design, exposes the exposure inventory and
the block/refusal logic, and NEVER fabricates a cell or claims a real-data result.
Every data-dependent entry point fails closed with a clear refusal when its inputs
are absent or fail provenance.
"""
