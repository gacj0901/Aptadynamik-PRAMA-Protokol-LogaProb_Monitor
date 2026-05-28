# Method

PRAMA treats an LLM response as a generation trajectory. Token-level signals are aggregated into short windows and passed into the PRAMA core as dynamic and symbolic inputs.

The core evolves state variables for coherence, pressure, accumulated tension, permissivity, polarization, regime geometry, and constitutive dominance. The resulting time series is used to monitor structural viability across the trajectory.

The offline scenarios are deterministic checks for the core dynamics. The Gemini pipeline applies the same core to logprob-derived signals from live model output.
