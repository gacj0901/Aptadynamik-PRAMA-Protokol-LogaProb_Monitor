# Logprob Mapping

The Gemini pipeline reads chosen-token logprobs and top-candidate logprobs when the API returns them.

For each token, it computes:

- `gap`: the distance between the top two candidate logprobs, when available.
- `entropy`: Shannon entropy over the returned top candidates.
- `top1_logprob`: the chosen token logprob.

Signals are aggregated in fixed-size windows. Narrower gaps increase the dynamic input. Lower entropy increases the symbolic input. Each window then advances the PRAMA core by one step.

## DeepSeek mapping

DeepSeek is accessed through the OpenAI SDK with:

- `base_url="https://api.deepseek.com"`
- `model="deepseek-chat"`
- `logprobs=True`
- `top_logprobs=5`

The response may resolve to a provider-side model name such as `deepseek-v4-flash`. The raw response exposes token logprob content in:

```text
choice.logprobs.content = [
  { token, bytes, logprob, top_logprobs: [...] }
]
```

PRAMA stores each DeepSeek token as:

- `token`: the emitted token string.
- `top1_logprob`: the chosen token logprob.
- `top_logprobs`: valid returned candidate logprobs, excluding sentinel values such as `-9999.0`.
- `gap`: the difference between the two highest valid logprobs, or `0.0` when fewer than two valid values are available.
- `entropy`: normalized softmax entropy over the filtered candidate logprobs, or `0.0` when fewer than two valid values are available.

If the chosen-token logprob is not present in `top_logprobs`, it is inserted before computing the filtered candidate list. This keeps `raw.json` compatible with PRAMA ProbLog Components while avoiding sentinel values that would distort gap or entropy calculations.
