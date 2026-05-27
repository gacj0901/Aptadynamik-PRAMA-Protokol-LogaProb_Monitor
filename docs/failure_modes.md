# Failure Modes

Known operational limits:

- Missing logprobs: some API responses may not include token logprobs, leaving the pipeline without trajectory signals.
- Provider schema changes: logprob fields can move or change shape across SDK versions.
- Short responses: very small token counts produce few windows and less stable trajectory summaries.
- Parameter sensitivity: the mapping from token uncertainty to PRAMA inputs can be tuned and should be reported with results.
- Rate limits: the Gemini demo sleeps between prompts to respect free-tier limits.
