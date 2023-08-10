"""Tests for state-space factorisations (at least, for the ones not tested otherwise).

To ensure that all state-space model factorisations work as expected,
we need to consider extrapolation and correction separately.
We already know that extrapolation is correct (from the strategy tests).

To check that linearisation/correction are correct,
we run a simple simpluation for each recipe and compare RMSEs.
"""
