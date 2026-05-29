\# Empirical Result 006 — Prompt Pressure Ψ restores dynamic discrimination in PRAMA v2



\## Summary



This experiment integrates an independent Prompt Pressure Layer Ψ into the PRAMA v2 pipeline.



Unlike the previous implementation, where both dynamic and symbolic channels were derived from output logprob geometry, Ψ is now extracted directly from the input prompt before generation. This makes Ψ independent from Φ and allows PRAMA dynamics to operate over a real coupling between environmental demand and generative response.



\## Experimental setup



Model:



```text

gpt-4o-mini

\-----------------



Configuración:

MAX\_TOKENS=128

PRAMA\_PROMPT\_LIMIT=5

Full replications=5

Families=4

Prompts per family=5

Total prompts per run=20

-----------------

Familia de prompts

canonical

fictional / unsupported

contradictory constraint

saturation / overconstraint


---------------------

The pipeline now separates three layers:



1\. Prompt Pressure Layer Ψ

&#x20;  Extracted from the prompt before generation.



2\. Logprob Geometry Layer Φ

&#x20;  Extracted from the model output.



3\. PRAMA Dynamics Layer

&#x20;  Computed from the dynamic evolution of the system.

Aggregate results



Across five full replications:



Prompt pressure tests:   25/25

Logprob geometry tests:  25/25

PRAMA dynamics tests:    18/20



Prompt pressure stability:



P1\_psi\_contradictory\_gt\_low             5/5

P2\_psi\_saturation\_gt\_low                5/5

P3\_fictional\_pressure\_low               5/5

P4\_saturation\_load\_gt\_contradictory     5/5

P5\_contradiction\_load\_gt\_canonical      5/5



Logprob geometry stability:



G1\_entropy\_std\_saturation\_gt\_canonical  5/5

G2\_entropy\_range\_saturation\_gt\_canonical 5/5

G3\_structural\_entropy\_std\_gt\_semantic   5/5

G4\_structural\_entropy\_range\_gt\_semantic 5/5

G5\_canonical\_rigidity\_highest           5/5



PRAMA dynamics stability:



D1\_xi\_contradictory\_gt\_canonical        5/5

D2\_xi\_saturation\_gt\_canonical           4/5

D3\_xi\_structural\_gt\_semantic            4/5

D4\_lambda\_contradictory\_lt\_fictional    5/5

Interpretation



The result shows that prompt pressure and output geometry are strongly aligned.



The key correction is that Ψ is no longer derived from the same output geometry as Φ. Ψ is extracted independently from the input prompt, while Φ is computed from logprob geometry during generation.



This removes the previous circularity between channels and allows PRAMA dynamics to recover stronger discrimination.



Main empirical claim



PRAMA v2 recovers dynamic discrimination when Ψ is extracted independently from the prompt rather than derived from the same logprob geometry as Φ.



Strong formulation



With Ψ calculated from input pressure and Φ calculated from output logprob geometry, PRAMA v2 obtains strong but not absolute dynamic separation over short LLM trajectories.



In five full replications, Prompt Pressure and Logprob Geometry passed all tests, while PRAMA Dynamics passed 18/20 secondary tests.



Methodological consequence



The previous instability of PRAMA dynamics was not simply a failure of the PRAMA motor. It was partly caused by insufficient independence between its input channels.



Once environmental pressure Ψ is extracted from the prompt, PRAMA dynamics becomes meaningfully testable as a coupling between:



environmental demand

generative geometry

dynamic viability

Conservative conclusion



The primary discriminator remains the combination of Prompt Pressure and Logprob Geometry. However, the introduction of an independent Ψ channel substantially improves the stability of PRAMA dynamics.



This supports the interpretation that PRAMA requires structurally independent channels to operate as a dynamic viability framework rather than as a redundant transformation of logprob geometry.



