# Logprob Mapping

The Gemini pipeline reads chosen-token logprobs and top-candidate logprobs when the API returns them.

For each token, it computes:

- `gap`: the distance between the top two candidate logprobs, when available.
- `entropy`: Shannon entropy over the returned top candidates.
- `top1_logprob`: the chosen token logprob.

Signals are aggregated in fixed-size windows. Narrower gaps increase the dynamic input. Lower entropy increases the symbolic input. Each window then advances the PRAMA core by one step.
