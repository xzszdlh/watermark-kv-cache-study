# Watermark detection and source grounding under KV-cache compression

A study of sentence-level watermark signals, decoding uncertainty, and source-memory compression in abstractive summarization.

## Research question

Conditional text watermarking introduces a tension between detectability and fidelity to the input. A green-list bias changes which tokens the model samples, while summarization requires those tokens to remain consistent with the source document. Fu, Xiong, and Dong's work on conditional watermarking motivated our initial question: **at a fixed watermark strength, can local watermark scores identify sentences that have lost source support?**

We first tested that diagnostic idea. After adjustment for sentence-level properties, the apparent association was close to zero. Examining this result shifted our attention to the generation distribution itself: the same watermark bias can have different effects depending on the probabilities assigned to candidate tokens. HalluKV's analysis of source-memory anchoring then suggested an intervention. If access to source memory is reduced, can watermark detection improve even as source support deteriorates?

## Experimental setup

| Component | Setting |
| --- | --- |
| Summarizer | `facebook/bart-large-cnn` |
| Dataset | CNN/DailyMail v3.0.0 |
| Calibration set | 200 validation documents |
| Evaluation set | 500 disjoint test documents |
| Generation seeds | 42, 43, 44 |
| Maximum source length | 1,024 BART tokens |
| Watermark | `simple_1` green-list scheme, green fraction $\gamma=0.25$ |
| Bias sweep | $\delta \in \{0,1,2,4\}$ |
| Decoding settings | Unrestricted sampling; nucleus $p=0.9$; top-$k=50$; nucleus sampling with a 256-step limit |
| Source-support evaluator | `cross-encoder/nli-deberta-v3-base` |

Experiment 1 contains 33,600 generated summaries across four decoding settings. The cache experiments and their sensitivity settings contain another 84,000, for **117,600 generation records across calibration and test sets**. Repeated seeds and conditions share the same 500 independent test documents.

Watermark detection is evaluated on rendered text re-tokenized by BART. Thresholds are calibrated using unwatermarked validation outputs at a target false-positive rate of 1%. Cache-policy comparisons use the matching full-cache threshold. Sentence-level analysis also retains the original sampled-token signal, aligned to sentence spans through byte-to-character offsets.

For each summary sentence, NLI compares the sentence against overlapping windows of the model-visible source, using a 256-token window and a stride of 192 NLI tokens. The highest entailment probability across windows gives sentence support; the lowest sentence score in a summary gives its document-level support score. ROUGE-L and pre-watermark next-token entropy are recorded alongside these measurements.

## Experiment 1: Can sentence-level watermark scores identify unsupported claims?

The initial hypothesis was that a sentence with an unusually high local watermark $z$-score might be a sentence in which the green-list bias had displaced source-consistent choices. A useful association would allow the existing watermark signal to help identify unsupported claims without an additional verification-model pass.

Under unrestricted sampling at $\delta=2$ and seed 42, the analysis included 2,063 sentences. The raw correlation between sampled-token watermark scores and NLI support was negative, as expected:

| Association | Pearson / partial $r$ | Document-bootstrap 95% CI |
| --- | ---: | --- |
| Raw sentence-level association | $-0.138$ | |
| Adjusted association | $-0.009$ | $[-0.048, +0.025]$ |

The adjustment includes mean pre-watermark entropy, log unique-bigram count, relative sentence position, source trigram overlap, and bigram repetition. Confidence intervals for the adjusted association span zero across all four decoding settings.

The raw trend therefore did not provide the independent factuality signal we had hoped to obtain. Watermark scores also reflect the distribution from which tokens are sampled and the amount of evidence accumulated within a sentence. This led to a different test: hold the bias fixed and intervene on the model's access to source information.

## Experiment 2: What changes when source memory is pruned?

HalluKV's distinction between source-document memory and generated-history memory motivated separate interventions on BART's two cache compartments. After encoding the full source, we compared four policies:

| Policy | Source cross-attention cache | Generated-history self-attention cache |
| --- | --- | --- |
| Full | 100% retained | 100% retained |
| Self-half | 100% retained | 50% retained |
| Cross-half | 50% retained | 100% retained |
| Both-half | 50% retained | 50% retained |

Source-cache pruning gathers evenly spaced projected key/value positions in each cross-attention layer, preserves endpoints, and updates attention masks. History-cache pruning retains the decoder-start state and recent history while preserving the true absolute decoder position. Cache tensors are physically compacted, with checks on storage size and reuse. The 50% ratios apply within each compartment; they are not equal-total-memory-budget comparisons.

### Main results

Nucleus sampling ($p=0.9$), $\delta=2$, 128-step generation limit; means over 500 test documents and three seeds.

| Cache policy | NLI support | ROUGE-L × 100 | Text watermark $z$ | Detection rate | Base entropy (nats) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 0.1274 | 18.78 | 6.03 | 97.53% | 3.07 |
| Self-half | 0.1218 | 17.22 | 5.86 | 96.13% | 3.20 |
| Cross-half | 0.0813 | 15.72 | 6.68 | 99.00% | 3.58 |
| Both-half | 0.0777 | 14.11 | 6.28 | 97.60% | 3.66 |

Removing half of the source cache reduced NLI support from 0.1274 to 0.0813, a **36.2% decrease**, while increasing text watermark $z$ by **0.6445** and detection by **1.47 percentage points**. The paired 95% intervals were $[-0.0562,-0.0361]$ for the NLI difference and $[+0.5103,+0.7783]$ for the watermark-score difference.

The history-cache control behaved differently. Its NLI difference from Full was $-0.0056$, with a 95% interval of $[-0.0161,+0.0048]$. Source-cache pruning thus produced a substantially larger loss of measured support than history-cache pruning in this setting.

### Sensitivity analyses

Each row compares source-cache pruning with the matching full-cache baseline at $\delta=2$. Paired contrasts average across seeds within each document; intervals use 50,000 document-cluster bootstrap resamples. Detection differences are in percentage points.

| Setting | Change in NLI support [95% CI] | Change in text $z$ [95% CI] | Change in detection | Change in entropy (nats) |
| --- | --- | --- | ---: | ---: |
| 75% source retention | $-0.0153$ $[-0.0249,-0.0057]$ | $+0.260$ $[+0.135,+0.386]$ | +1.00 pp | +0.23 |
| 25% source retention | $-0.0750$ $[-0.0853,-0.0649]$ | $+0.802$ $[+0.664,+0.940]$ | +1.87 pp | +0.93 |
| Top-$k=50$, 50% retention | $-0.0777$ $[-0.0905,-0.0651]$ | $+0.702$ $[+0.602,+0.803]$ | +9.33 pp | +0.53 |
| 256-step limit, 50% retention | $-0.0454$ $[-0.0554,-0.0355]$ | $+0.647$ $[+0.482,+0.812]$ | +1.07 pp | +0.50 |
| Random source positions, 50% retention | $-0.0432$ $[-0.0533,-0.0334]$ | $+0.421$ $[+0.293,+0.548]$ | +1.67 pp | +0.43 |

Under top-$k=50$, full-cache detection starts at 84.40%, leaving more room for a change than the nucleus baseline. Pruning half of the source cache raises detection to **93.73%** while reducing NLI support by 0.0777. Extending the generation limit and replacing evenly spaced retention with random interior positions preserve the same direction of change.

## Interpreting the entropy change

At fixed $\delta=2$, source-cache pruning increases mean pre-watermark entropy from 3.07 to 3.58 nats. This accompanies lower source support and stronger watermark detection.

For a base probability mass $p_G$ assigned to green-list tokens, adding a logit bias $\delta$ gives the following green-list mass before top-$k$ or top-$p$ filtering:

$$
p'_G = \frac{e^{\delta}p_G}{1+(e^{\delta}-1)p_G}.
$$

The response to a fixed bias depends on the underlying probability distribution. The observed entropy increase is consistent with source-memory loss changing how strongly the watermark affects sampling. Entropy alone does not determine $p_G$; the cache intervention directly tests the joint changes in source support, entropy, and detection.

## Blinded sentence audit

A three-LLM audit evaluates 260 sentences with experimental conditions hidden: 100 sentences from the initial watermark study and 160 from paired cache conditions. The panel produced 775 valid structured judgments out of 780 requests.

Judges assign one of four labels: supported, contradicted, not-in-source, or unclear. This distinguishes interpretable factual errors from fragmented text that cannot be judged reliably. On interpretable sentences, NLI AUROC against the individual judges ranges from **0.753 to 0.910**. These are model-assisted judgments, reported separately from the automatic NLI measurements.

## Conclusion

The sentence-level experiment did not support using local watermark scores as an independent factuality diagnostic. The cache intervention established a different result: **source support and watermark detection can move in opposite directions at an unchanged watermark bias**. Under source-cache compression, a more detectable watermark can coexist with a less source-supported summary. Provenance detection and contextual faithfulness therefore need to be measured separately in this setting.
