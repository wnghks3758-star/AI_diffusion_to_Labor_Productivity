# Validation

Python 3.12; pinned dependencies in requirements.txt.

The automated suite covers training-only and full analysis exclusions, industry fold separation, Gaussian versus kernel GPS routing, normalized kernel targets, repeated-prediction medians, cache reuse, nonpositive PS2 predictions, cross-dose and size-group covariance, strict/supported aggregation, and completed-output reuse. Real-panel integration checks use reduced-cost learners; they do not establish causal identification or validate statistical significance.

Run: `python -B -m unittest discover -s tests -v`
