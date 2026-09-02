"""ORACLE-Cov: the mechanism tier of the consequence-aware failure-diagnostics study.

Everything here is CPU/numpy.  The testbed is the planted-structure waveform
generator of ``experiments.synthetic.nfpa_confirmatory`` (exact separable
covariance, known forward map), so the pullback metric, the excited tangent
span and the Fisher matrix are all available in closed form or by finite
differences of the generator.
"""
