## Latent Altruism: Steering Cooperative Intent in Large Language Models via Activation Engineering

NeurIPSSubmitted: March 5, 2026

### Contents

* [Summary](https://paperreview.ai/review#summarySection)
* [Strengths](https://paperreview.ai/review#strengthsSection)
* [Weaknesses](https://paperreview.ai/review#weaknessesSection)
* [Detailed Comments](https://paperreview.ai/review#detailedSection)
* [Questions](https://paperreview.ai/review#questionsSection)
* [Overall Assessment](https://paperreview.ai/review#assessmentSection)

### Summary

This paper studies whether a generic notion of “cooperative intent” is linearly encoded in an instruction-tuned 32B LLM and can be elicited at inference time via activation engineering. The authors construct a cooperation steering vector from contrastive hidden states (AllC vs. AllD) in the Iterated Prisoner’s Dilemma (IPD), show strong behavioral shifts toward cooperation (e.g., from 2% to 100% vs. TFT in IPD), localize peak discriminability to layer 57, and report zero-shot transfer to Stag Hunt and Chicken. They also find a sharp failure mode: adversarial system prompts with competitive/betrayal/deception/dominance framings nullify the steering effect.

### Strengths

* Technical novelty and innovation
  * The paper connects activation engineering with multi-agent strategic behavior, going beyond prior persona/honesty/stance steering to a game-theoretic setting.
  * The layer-wise discriminability analysis and identification of a “strategic bottleneck” around layer 57 is an interesting mechanistic insight that could inspire follow-up interpretability work.
* Experimental rigor and validation
  * Evaluations span three classic 2x2 social dilemmas and multiple opponent policies, including repeated-play settings.
  * Cross-game transfer (extracted on IPD, tested on Stag Hunt and Chicken) suggests the vector captures a task-general concept beyond surface token associations.
  * Comparisons among SV, CAA, and RepE provide a starting point for method benchmarking.
* Clarity of presentation
  * The problem motivation is clear, and the experimental setup is mostly well-described (games, payoffs, opponents, T/R trials, α-sweeps).
  * The geometry visualizations (PCA/t-SNE) and layer-wise plots aid intuition about separability and where strategic intent may be represented.
* Significance of contributions
  * Demonstrating strong behavioral control in repeated games with minimal overhead is potentially impactful for agentic LLMs.
  * The adversarial “contextual override” result is a valuable, cautionary contribution that delimits where activation steering fails.

### Weaknesses

* Technical limitations or concerns
  * Central implementation details are underspecified: how actions are decoded from logits, the exact prompting templates for per-round decisions, and the corpus used to compute perplexity (especially for the “PPL ratio = 1.0” claim).
  * The steering vector is derived from prompts that explicitly instruct “always cooperate/defect,” raising concerns about lexical/format confounds; there are no control vectors (random, orthogonal, token-level swaps) to rule out superficial effects.
  * The study uses a single model (Qwen2.5-32B-Instruct), further quantized to 4-bit AWQ, which may distort activation scales; generality to other architectures/sizes or full-precision settings is not tested.
* Experimental gaps or methodological issues
  * Reported inconsistencies: the text claims SV achieves 100% cooperation “at all tested α,” but Table 3 shows 34% at α = 0.05; Section 4.3 text implies “93.6% mean cooperation across all opponents” in Stag Hunt, but that number corresponds to the AllD row in Table 2, not the mean across opponents.
  * The adversarial-prompt robustness uses α = 0.3 only; concluding a “0% recovery” boundary without α sweeps, layer sweeps (e.g., injecting at the identified layer 57), or multi-layer/attention-head interventions weakens the generality of that claim.
  * No ablations test whether injecting at layer 57 (the identified bottleneck) materially outperforms last-layer injection in behavior, despite the mechanistic claim.
  * The PPL evaluation is opaque: dataset, tokenization, context length, and how PPL is aggregated are not specified, making “exactly 1.0” across all α suspicious.
* Clarity or presentation issues
  * Minor internal contradictions (noted above) and some imprecise language about averages vs. per-opponent results detract from clarity.
  * The description of the Fisher Discriminability Index and how projections are computed would benefit from a clearer, fully specified procedure (e.g., per-sample projections vs. class means).
* Missing related work or comparisons
  * Recent activation-engineering advances and diagnostics are largely absent. Notably missing: continuous diagnostics and layer-site sensitivity (CBMAS, 2601.06109), standardized evaluation protocols for open-ended generations (2410.17245), sparse/SAE-based steering (SPARE 2410.15999, SRS 2503.16851), and semantic denoising/augmentation of steering vectors (SAE-RSV 2509.23799). Input-adaptive methods like SADI (2410.12299) are also unexplored.
  * These omissions matter because several of these methods address exactly the paper’s core questions (layer dependence, capability preservation, robustness, denoising/generalization).

### Detailed Comments

* Technical soundness evaluation
  * The linear difference-of-means vector is a sensible baseline and matches prior activation engineering practice. The layer-wise discriminability analysis is a useful addition, but the mechanistic claim (“strategic bottleneck”) would be stronger with interventions at the identified layer, causal patching, or head/MLP-level analyses to validate function.
  * Missing controls reduce causal confidence: random/orthogonal vectors, token-swap controls (e.g., “cooperate” vs. paraphrases), and prompt-template variants could disentangle semantics from lexical artifacts.
  * The “PPL ratio = 1.0” claim is not credible without detailed methodology. If PPL is computed on the same short prompts used to decide C vs. D, it is not an adequate capability proxy; a standard held-out corpus and clear aggregation are required.
* Experimental evaluation assessment
  * The behavioral results (IPD → 100% vs. TFT, transfer to Stag Hunt/Chicken) are compelling, but the small number of repetitions (R = 5) and single architecture limit statistical and external validity.
  * The adversarial-context finding is important, but testing only one α and a single injection site undercuts the conclusion that “activation steering cannot withstand hostile contexts.” Sweeping α, injecting at layer 57, trying multi-layer or attention-head-level steering (as suggested in the Discussion), and comparing to sparse/feature-level methods (SPARE/SRS/SAE-RSV) would clarify whether the phenomenon is fundamental or method-specific.
  * The method comparison (SV vs. CAA vs. RepE) is valuable, but more baselines are warranted given recent progress: SADI (input-adaptive steering), SPARE/SRS/SAE-RSV (sparse/SAE-based, often with better robustness and interpretability), and the standardized likelihood-based evaluation protocol (2410.17245).
* Comparison with related work (using the summaries provided)
  * CBMAS (2601.06109) recommends dense α- and layer-site sweeps, logit-lens readouts, and control vectors to diagnose tipping points and fluency preservation. Adopting these diagnostics would significantly strengthen the claims about layer 57, α sensitivity, and language quality preservation.
  * Sparse/SAE-based steering methods such as SPARE (2410.15999), SRS (2503.16851), and SAE-RSV (2509.23799) provide interpretable, feature-level control, often with stronger robustness and denoising. Given the “contextual override” result, it is crucial to test whether sparse/feature-targeted edits can resist adversarial framings better than dense DoM vectors.
  * SADI (2410.12299) shows that input-adaptive steering often outperforms fixed steering vectors; incorporating an adaptive variant might mitigate context sensitivity and increase robustness across tasks and prompts.
  * The standardized evaluation pipeline (2410.17245) argues that multiple-choice or sampled outputs can overstate steering success; applying likelihood-based, open-ended diagnostics would clarify whether the intervention truly promotes cooperative continuations rather than suppressing defections or exploiting prompt idiosyncrasies.
* Discussion of broader impact and significance
  * The work raises important questions about aligning agentic LLM behavior in social dilemmas. However, always-cooperating against AllD opponents can be exploited and may not reflect social welfare optimization; future studies could evaluate welfare-aware steering (e.g., reciprocity-aware or conditional cooperation) and consider safety–performance trade-offs.
  * The “contextual override” finding highlights a critical safety concern: latent-space edits can be nullified by framing. This underscores the need for defense-in-depth (activation edits + prompt governance + output filters) and possibly more localized/circuit-level interventions.

### Questions for Authors

1. How exactly are actions decoded from logits? Is the model asked to output “Cooperate/Defect,” “C/D,” or free-form justifications? Please provide the exact prompt templates and the tokenization/mapping rules used to determine C vs. D each round.
2. How is perplexity computed (dataset, number of tokens, context length, decoding settings)? What explains the exact 1.000 ratio for SV across α? Please report PPL on a standard held-out corpus and provide variance/error bars.
3. What is N (number of calibration samples) used to compute the steering vector? Did you vary prompt templates or paraphrases during calibration to reduce lexical confounds? Any ablations with random/orthogonal vectors or token-swap controls?
4. Given the identified peak discriminability at layer 57, what are the behavioral effects of injecting at layer 57 vs. the last layer (and multi-layer injections)? Do α sweeps at layer 57 change the adversarial robustness conclusions?
5. Did you test additional α values or stronger/weaker injections under adversarial prompts? Is “0% recovery” robust across α and injection layers?
6. How sensitive are results to 4-bit AWQ quantization? Do full-precision (or higher-bit) runs exhibit the same layer-57 peak and behavioral effects?
7. Can you compare against more recent baselines (SADI, SPARE, SRS, SAE-RSV) and/or adopt CBMAS-style diagnostics and the standardized likelihood-based evaluation (2410.17245) to strengthen claims about robustness and capability preservation?

### Overall Assessment

The paper is timely and interesting: it applies activation engineering to cooperative behavior in repeated games, presents clear behavioral gains and an informative mechanistic signal (a late-layer “strategic bottleneck”), and candidly surfaces a critical failure mode under adversarial framings. However, several methodological gaps and inconsistencies limit confidence: missing details on action decoding and PPL computation, lack of control vectors and standardized diagnostics, single-model evidence (with 4-bit quantization), and incomplete robustness sweeps for the adversarial setting. The related-work coverage omits several recent, directly relevant steering advances (adaptive and sparse/SAE-based methods) and evaluation best practices that would likely sharpen and sometimes challenge the current claims. With stronger controls, additional baselines (SADI, SPARE, SRS, SAE-RSV), CBMAS-style layer/α diagnostics, clarified PPL methodology, and targeted interventions at the identified layer 57 (including adversarial sweeps), the contribution could become a solid empirical advance. In its current form, I view it as promising but not yet meeting the rigor and breadth expected for NeurIPS.
