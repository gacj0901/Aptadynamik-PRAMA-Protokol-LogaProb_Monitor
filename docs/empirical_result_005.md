@'

\# Empirical Result 005 - Pre-motor logprob geometry vs PRAMA dynamics



Date: 2026-05-28  

Pipeline: PRAMA v2 prompt-family pipeline  

Model: gpt-4o-mini  

Context: replicated prompt-family experiments with logprob-derived metrics  



\## Core finding



The discriminating signals in the current LLM prompt-family experiment are computed before the PRAMA motor.



The strongest metrics are:



\- rigidity index

\- uncertainty index

\- margin index

\- intra-window entropy standard deviation

\- intra-window entropy range



These are properties of the model's logprob distribution. They describe the local geometry of generation before the PRAMA dynamics are applied.



The PRAMA motor receives these signals and produces dynamic outputs:



\- xi per window

\- lambda

\- integrity

\- regime

\- anomaly



However, in this specific application, those outputs do not consistently amplify the discrimination already present in the pre-motor logprob geometry. In several replications, the PRAMA outputs either add noise or weaken the family separation.



\## Interpretation



For short LLM generations, the logprob geometry layer is currently more discriminative than the PRAMA dynamics layer.



This does not invalidate the PRAMA motor. It delimits its operational regime.



The motor was designed for systems where:



1\. Input channels are structurally distinct.

2\. Tension accumulates across sufficiently long trajectories.

3\. Regime changes have time to develop.

4\. Threshold contraction and constitutive rotation operate over enough steps.



In the current experiment, a response of 128 tokens divided into windows of 8 tokens yields only about 16 motor steps. That is too short for the full PRAMA dynamic machinery to add stable discriminatory power.



By contrast, the offline verification used longer synthetic trajectories with clearly separated input channels. In that regime, the PRAMA motor discriminated correctly.



\## Methodological correction



The repository should distinguish three layers:



\### 1. Logprob Geometry Layer



Primary layer for LLM prompt-family discrimination.



It measures:



\- rigidity

\- uncertainty

\- margin

\- entropy volatility

\- entropy range



This layer directly captures how the model distributes probability mass during generation.



\### 2. PRAMA Dynamics Layer



Secondary layer for temporal viability analysis.



It measures:



\- accumulated tension

\- permissivity

\- integrity

\- regime transition

\- anomaly



This layer should not be assumed to improve classification in short LLM traces unless experimentally verified.



\### 3. Verification Layer



Evaluates whether the PRAMA motor adds discriminative power beyond the raw geometry layer.



The relevant question is not only:



"Does PRAMA discriminate?"



but also:



"Does PRAMA add discrimination beyond the pre-motor logprob geometry?"



\## Corrected empirical claim



The current experiments support the following claim:



Logprob-derived generation geometry can empirically distinguish between canonical, fictional, contradictory, and saturation prompt families in GPT-4o-mini. In particular, structural prompt stress is expressed more robustly as intra-window entropy volatility than as accumulated PRAMA tension.



This is an empirical result based on real model logprobs.



\## Scope limitation



The current result should not be generalized to all PRAMA applications.



It applies specifically to:



\- short LLM responses

\- GPT-4o-mini

\- logprob windows of approximately 8 tokens

\- prompt-family discrimination

\- derived metrics computed from token probabilities



The PRAMA motor may remain appropriate for:



\- longer trajectories

\- synthetic verification

\- control systems

\- multi-channel dynamical systems

\- regimes where input streams are structurally independent

\- tasks where accumulation and threshold contraction matter



\## Working conclusion



For the LLM prompt-family experiment, the primary object of study should be called generative logprob geometry.



PRAMA dynamics should be treated as an optional secondary transformation, not as the primary discriminator.



The next technical step is to update the v2 tests so they evaluate pre-motor geometry first and PRAMA dynamic outputs second.

'@ | Set-Content docs/empirical\_result\_005.md -Encoding utf8



git add docs/empirical\_result\_005.md

git commit -m "Document pre-motor logprob geometry finding"

git push

