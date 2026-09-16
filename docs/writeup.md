# False statements as a substrate for hidden computation

*Project write-up, 2026-09-16. Code: [github.com/andreach3n/filler-token-facts](https://github.com/andreach3n/filler-token-facts). Raw artifacts: private HF dataset `andreayhchen/filler-token-facts-states`.*

## 1. Question and background

Two recent papers show that frontier language models can use meaningless "filler" tokens appended to a prompt as extra positions for computation, without any chain of thought:

- **Brauer et al., *Reading Between the Dots* (arXiv 2607.03502, "paper 1").** Dots or counting sequences (`1 2 3 …`) placed between the question and `Answer:` raise no-CoT accuracy in DeepSeek V3 and Kimi K2 (e.g. systems of equations, DeepSeek V3: 31% → 61%). With a logit lens plus cross-example mean subtraction, the intermediate values of the computation are decodable at the filler positions, and KV-cache transplants between problems swap the answer 13–34% of the time.
- **Baherwani et al., *Not All LLM Reasoning is Visible in the Chain-of-Thought* (arXiv 2607.22925, "paper 2").** Semantic word fillers (animals, fruits, colors, …) help some models (Claude Opus 4.5: +11 points on multi-step arithmetic) and collapse others (Qwen3.5-397B to ~0% with word fillers).

**Our question:** can *false statements* — full sentences with propositional content that the model knows to be false — serve as the same kind of computational substrate? If so, text that looks like (mis)information can carry unrelated hidden computation, which matters for anyone monitoring model outputs.

## 2. Experimental design

### 2.1 Conditions

Every condition places the filler in the user turn between the problem and `Answer:` (paper 1's layout), in the target prompt and in every few-shot example, under a `Filler:` label. The system prompt says nothing about the filler (paper 2 found the hint has no effect, and telling the model that false statements are "space to think" would confound the result).

| Condition | Filler |
|---|---|
| `none` | — |
| `counting` | `1 2 3 …`, length matched to the statement block (positive control) |
| `false` | 10 false statements, ~110 tokens |
| `true` | the same 10 statements with the true object (matched control) |

Placement variants (systems of equations only): `after` the question (main), `before` the whole problem, and `middle` (after the 3rd of 5 variable definitions).

### 2.2 Statements

We avoided the standard truth-probing datasets (Geometry of Truth, CounterFact, Azaria–Mitchell) because they are heavily used in the literature. Instead:

- Facts were drawn from PopQA (Wikidata triples with popularity scores) for four relations with a single unambiguous answer: author, composer, director, sport.
- 43 well-known facts were hand-curated and written in our own templates, with a type word to remove ambiguity ("The film Alien was directed by Ridley Scott."). No statement contains a number.
- The false version swaps the object for another of the same kind from a pool ("The film Alien was directed by Steven Spielberg."), chosen so the true and false sentences have the **same token count** under each model's tokenizer (41 of 42 pairs match under all three tokenizers).
- Every sentence was posed to both study models as a true/false question: **84/84 sentences were classified correctly by both**, so the false statements are false *to the models*.
- The 42 pairs were grouped into **3 disjoint sets of 10** (3 author, 3 composer, 2 director, 2 sport), never repeating a name within a set. Each set is fixed across its problems, which paper 1's mean-subtraction requires. Problem *i* uses set *i* mod 3.

Filler token counts per set (DeepSeek tokenizer): set-0 counting 111 / false 112 / true 112; set-1 105/106/106; set-2 107/107/107.

### 2.3 Tasks

All tasks are computed from the prompt itself; none requires fact recall (reading factual statements triggers the model's own recall, which would collide with a recall-based task).

- **Systems of equations** (paper 1's "chained variable binding"): 5 definitions with nonsense CVC names, one chain `x = 41`, `y = twice the number for x plus 31`, and a question `What is twice the number for y minus 22?`. Every intermediate (x, 2x, y, 2y, answer) is kept in 0–999, which DeepSeek tokenizes as a single token, so lenses can find them.
- **Multi-step arithmetic** (paper 2): 5–7 nested operations, operands in [−99, 99].
- **Variable counting** (paper 2): count the distinct variables assigned in a short Python snippet (answers 3–7).

Prompts are 10-shot with a fixed few-shot pool; output is capped at 16 tokens; temperature 0.

### 2.4 Models and access

| Model | Access | Notes |
|---|---|---|
| DeepSeek V4 Flash (original 0423 checkpoint, 284B/13B active) | OpenRouter, pinned to Parasail fp8, reasoning off | main model; open weights; pre-fitted J-lens exists |
| DeepSeek V3-0324 | OpenRouter, pinned to GMICloud fp8 | reference: should reproduce paper 1 |

Each model is pinned to a single host with fallbacks disabled so paired conditions run on the same weights and kernels. (Claude Opus 4.5 and Qwen3.6-27B were set up but deferred.)

### 2.5 Statistics

All conditions run on identical problems, so differences are paired; we report McNemar tests and 95% CIs on the paired difference.

## 3. Behavioral results

### 3.1 Pilot: 3 tasks × 150 problems × 4 conditions

Accuracy; change vs. `none` is paired (gained/lost problems, McNemar p).

**DeepSeek V4 Flash**

| Task | none | counting | false | true |
|---|---|---|---|---|
| systems of equations | 54.7% | 71.3% (+16.7, +28/−3, p<.001) | **72.0% (+17.3, +28/−2, p<.001)** | 69.3% (+14.7, +27/−5, p<.001) |
| arithmetic | 40.7% | 41.3% (+0.7, ns) | 44.0% (+3.3, p=.30) | 43.3% (+2.7, p=.48) |
| variable counting | 90.7% | 92.0% (+1.3, ns) | 92.7% (+2.0, ns) | 91.3% (+0.7, ns) |

**DeepSeek V3-0324**

| Task | none | counting | false | true |
|---|---|---|---|---|
| systems of equations | 28.7% | 54.7% (+26.0, +48/−9, p<.001) | **51.3% (+22.7, +40/−6, p<.001)** | 53.3% (+24.7, +46/−9, p<.001) |
| arithmetic | 27.3% | 30.0% (+2.7, p=.50) | 28.7% (+1.3, ns) | 29.3% (+2.0, ns) |
| variable counting | 96.7% | 88.0% (−8.7, p=.004) | 87.3% (−9.3, p=.004) | 88.7% (−8.0, p=.008) |

Takeaways: V3 reproduces paper 1's counting-filler gain on systems of equations (28.7 → 54.7 vs. their 31 → 61). Arithmetic shows no gain on either model (matching paper 2's DeepSeek result); variable counting is at ceiling on V4 Flash and hurt by every filler on V3. The rest of the project uses systems of equations.

### 3.2 Main run: systems of equations, 300 problems, 3 placements

**DeepSeek V4 Flash** (`none` has 10 empty replies scored as wrong)

| Condition | Accuracy | vs. none (gained/lost, p) |
|---|---|---|
| none | 52.0% | — |
| counting, after question | 69.3% | +17.3 (+60/−8, p<.001) |
| **false, after question** | **69.3%** | **+17.3 (+58/−6, p<.001)** |
| true, after question | 68.7% | +16.7 (+58/−8, p<.001) |
| counting / false / true, before problem | 47.7 / 43.0 / 43.0% | −4.3 (p=.15) / −9.0 (p=.002) / −9.0 (p=.003) |
| counting / false / true, mid-definitions | 48.7 / 47.3 / 45.3% | −3.3 (p=.30) / −4.7 (p=.13) / −6.7 (p=.019) |

**DeepSeek V3-0324**

| Condition | Accuracy | vs. none |
|---|---|---|
| none | 33.3% | — |
| counting, after question | 56.0% | +22.7 (+84/−16, p<.001) |
| **false, after question** | **53.7%** | **+20.3 (+73/−12, p<.001)** |
| true, after question | 54.7% | +21.3 (+85/−21, p<.001) |
| counting / false / true, before problem | 25.3 / 22.7 / 23.7% | −8.0 (p=.010) / −10.7 (p<.001) / −9.7 (p<.001) |
| counting / false / true, mid-definitions | 27.3 / 28.7 / 27.0% | −6.0 (p=.047) / −4.7 (p=.13) / −6.3 (p=.025) |

Paired 95% CIs (V4 Flash excludes the empty-reply problems, n=290): counting vs none +16.6 [+11.5, +21.6]; false vs none +16.6 [+11.7, +21.4]; true vs none +15.9 [+10.9, +20.9]; **false vs counting +0.0 [−3.9, +3.9]**; false vs true +0.7 [−2.5, +3.9]. V3: counting +22.7 [16.7, 28.7]; false +20.3 [14.8, 25.9]; true +21.3 [15.1, 27.6]; **false vs counting −2.3 [−7.2, +2.5]**; false vs true −1.0 [−5.8, +3.8].

Mid-definition placement, split by whether the queried chain (x, then y) is defined before the filler: V4 Flash "x,y before" (n=106): none 50.9%, counting 52.8% (+1.9), false 50.9% (0.0), true 50.0% (−0.9); "x before, y after" (n=167): 52.1% → 47.3 / 44.9 / 41.9%; "x,y after" (n=27): 55.6% → 40.7 / 48.1 / 48.1%. V3 "x,y before": 34.0% → 34.9 / 33.0 / 31.1%.

**Findings.**
1. False statements help exactly as much as counting filler; the difference is within ±4 points on V4 Flash and within ±7 on V3.
2. Truth value makes no detectable difference.
3. The gain requires the filler *after* the question. Before the problem it hurts (as paper 1 found); mid-definitions it does not help even when both chain variables precede the filler. (Accuracy alone cannot say why; see §5.)
4. V4 Flash sometimes ends its reply immediately with no answer (10 of 300 no-filler prompts); the same behaviour reproduces on the locally hosted weights.

API cost for everything above: about $3.50.

## 4. Interpretability

### 4.1 Setup

DeepSeek V4 Flash was hosted on a RunPod pod with 2× RTX PRO 6000 Blackwell (96 GB each) using DeepSeek's own inference code (the original FP4/FP8 checkpoint, 160 GB, split over 2 GPUs). Getting it to run required several patches, all in `gpu/`:

- The tilelang **FP4 expert GEMM returns garbage on this GPU** (relative error 1.04 vs. a dequantized reference); all MoE experts are FP4, so the model degenerated to copying its context. It was replaced by a PyTorch dequant + bf16 matmul (error 0.0017). All other kernels checked out (FP8 GEMM 0.0016; activation quantization within fp8 noise; hyper-connection Sinkhorn exact).
- The sparse-attention kernel exceeded the card's ~100 KB shared-memory limit; it was split over 16-head chunks (identical maths; error 0.0018 vs. a PyTorch reference).
- `apache-tvm-ffi` pinned to 0.1.9; `fast_hadamard_transform` replaced by a verified PyTorch implementation.

Validation: the model loads in 24 s (82.9 GB/GPU) and a 2,091-token prompt takes 5–14 s; greedy answers match the API ("172", "Rome", "4"); accuracy on the analysed weights is none 55.3% / counting 70.7% / false 67.0% / true 67.0% (n=300 each), matching the API runs.

V4 Flash keeps **4 parallel residual streams** ("hyper-connections") merged only at the output head. For lens readouts we tested how to merge them: reading the model's own final merge (`hc_head`) at layer 41 agrees with the model's next-token prediction 71.9% of the time, versus 48.4% for averaging the streams and 35.9% for a single stream; at the last layer it reproduces the output exactly. All readouts use the `hc_head` merge.

### 4.2 Pipeline

1. **Extraction** (`extract_states.py`): one forward pass per prompt (all 1,200), storing the merged residual at every layer for all filler positions and the last 4 positions.
2. **Readout** (`lens_readout.py`): at every (layer, position), distributions from three lenses — logit lens `softmax(W_U · norm(h))`; **J-lens** and **R-lens** (Jacobian lenses pre-fitted for this checkpoint by a third party, `camilablank/workspace-lenses`, n=25 prompts, layers 0–41), `softmax(W_U · norm(J_l · h))`. Then paper 1's cross-example residual: subtract the mean distribution over the prompts of the same (condition, statement set), which removes what every prompt shares at a position (the filler text itself) and leaves what is prompt-specific.
3. **Aggregation** (`aggregate.py`): for each prompt, is each target (x, 2x = "c1x", y, 2y = "c2y", answer) the top-1 (or in the top-10) residual token at *any* filler position, per layer? **Chance baseline:** the same statistic with the next prompt's targets. x is written in the prompt (finding it is copying); c1x, y, c2y and the answer never appear in the prompt (finding them is computation).
4. **Causal test** (`patch_filler.py`): activation patching, §4.4.

### 4.3 Decoding results

Fraction of prompts (correct answers only) where the target is the **top-1** residual token at some filler position, at the best layer; chance in parentheses.

**Logit lens**

| Filler | n | x (given) | c1x | y | 2y | answer |
|---|---|---|---|---|---|---|
| counting | 212 | 0.99 (0.09) @L25 | 0.45 (0.14) @L37 | 1.00 (0.08) @L35 | 0.91 (0.04) @L37 | 0.92 (0.06) @L39 |
| **false** | 201 | 0.53 (0.01) @L26 | 0.15 (0.01) @L35 | **0.96 (0.01) @L35** | **0.82 (0.01) @L35** | **0.88 (0.01) @L39** |
| true | 201 | 0.57 (0.01) @L25 | 0.20 (0.02) @L35 | 0.96 (0.03) @L35 | 0.88 (0.01) @L35 | 0.89 (0.01) @L39 |

**J-lens**

| Filler | x | c1x | y | 2y | answer |
|---|---|---|---|---|---|
| counting | 0.95 (0.33) | 0.46 (0.13) | 1.00 (0.14) | 0.92 (0.05) | 0.94 (0.05) |
| false | 0.84 (0.02) | 0.25 (0.03) | 0.99 (0.03) | 0.89 (0.01) | 0.91 (0.01) |
| true | 0.83 (0.02) | 0.26 (0.03) | 1.00 (0.04) | 0.93 (0.01) | 0.93 (0.01) |

**R-lens**

| Filler | x | c1x | y | 2y | answer |
|---|---|---|---|---|---|
| counting | 0.99 (0.40) | 0.54 (0.14) | 1.00 (0.13) | 0.93 (0.04) | 0.93 (0.06) |
| false | 0.79 (0.02) | 0.28 (0.04) | 0.99 (0.02) | 0.89 (0.02) | 0.94 (0.01) |
| true | 0.79 (0.01) | 0.31 (0.04) | 0.98 (0.04) | 0.94 (0.01) | 0.96 (0.01) |

Top-10 (logit lens, false statements): x 0.95 (0.09), c1x 0.44 (0.14), y 1.00 (0.07), 2y 0.95 (0.05), answer 0.96 (0.03).

Further observations (logit lens):

- **Stages by depth.** y peaks at layer 35, 2y at 35–37, the answer at 39, at the filler positions — a sequential computation laid out across layers. The J-lens and R-lens place the peaks at nearly the same layers.
- **It tracks correctness.** On incorrectly answered prompts (false statements, n=99): y 0.97 (chance 0.04), 2y 0.51 (0.02), **answer 0.26 (0.01)**. When the model fails, the later stages are missing from the filler positions.
- **Work moves out of the answer slot.** At the final token, intermediates are decodable less often with filler than without: y 0.32 (none) vs 0.13–0.15 (with filler); 2y 0.47 vs 0.32–0.37. The answer is top-1 there in 100% of correct prompts at layer 42 by construction.
- **Where in the filler.** For statements the single strongest position is the last statement token: the answer is top-1 there in 26% of prompts (layer 39), 2y in 40% (layer 35). The "any position" rates are far higher because different prompts peak at different tokens. For counting the peaks sit 3–5 tokens from the end.
- **Raw probability.** Without mean subtraction, the logit lens assigns the answer token a probability of 0.84 (false) / 0.89 (counting) / 0.84 (true) at its best filler position at layer 39 (mean over prompts of the max over positions).
- **Counting vs statements.** The given number x is copied into counting positions (0.99) far more than into statement positions (0.53–0.57). c1x is weak everywhere (0.15–0.54): the model may compute y without materializing 2x.

### 4.4 Causal test: activation patching

For a correctly answered prompt A, another problem B from the same condition and statement set was run and its residual at every filler position captured (all 4 streams, every layer). A was then run with its filler-position residuals replaced by B's at a chosen layer range, everything downstream recomputed. n=100 pairs per cell, plus 5 self-patches per condition.

| Filler | Layers patched | p(A's answer) before → after | p(B's answer) | greedy = B | greedy = A | other |
|---|---|---|---|---|---|---|
| **false** | all (0–42) | 0.93 → 0.01 | 0.74 | **85%** | 1% | 14% |
| false | late (30–42) | 0.93 → 0.13 | 0.69 | **77%** | 13% | 10% |
| false | early (0–20) | 0.93 → 0.58 | 0.00 | 0% | 65% | 35% |
| counting | all | 0.90 → 0.05 | 0.63 | 73% | 4% | 23% |
| counting | late | 0.90 → 0.18 | 0.56 | 64% | 24% | 12% |
| counting | early | 0.90 → 0.67 | 0.00 | 0% | 72% | 28% |
| self-patch (both) | all | unchanged | — | — | 100% | 0% |

Transplanting the late-layer states at the false-statement positions makes the model output the *other problem's* answer 77–85% of the time, although A's own question is still in the prompt. Early-layer transplants never yield B's answer (later layers recompute from the context) and only degrade A's. Paper 1's KV-cache transplants on DeepSeek V3 produced 13–34% full swaps; residual patching here is stronger. The effect for false statements is at least as large as for counting filler.

## 5. Interpretation

- **The main claim holds.** DeepSeek V4 Flash (and, behaviorally, V3) uses false statements placed after the question as a computational substrate exactly as it uses counting filler: the same accuracy gain, the intermediates and the answer decodable at the statement positions far above chance, in depth-ordered stages that track correctness, and a causal path from those positions to the answer.
- **Truth value is irrelevant.** True statements behave identically, so the accurate framing is that the substrate is *content-agnostic*: the model knows the statements are false (84/84 verified) and computes on top of them anyway.
- **Placement.** Filler only helps after the question. Whether that is because the model does not compute y in advance, or computes it but cannot use it, is not settled by accuracy; the stored states for the middle-placement prompts were not extracted, so this remains open.
- **Safety relevance.** From the outside, the statement positions look like ordinary (mis)informative text; internally they carry the arithmetic. A monitor reading the text would see nothing.

## 6. Limitations

1. One task shows the effect for DeepSeek (arithmetic did not), one model was analysed mechanistically, statements come from one source (4 relations, 3 sets).
2. "Top-1 at any of ~110 positions at the best of 43 layers" is a lenient statistic; the chance baseline (~1%) and the top-1 numbers make it convincing, but position-resolved rates are much lower (26–40% at the best single position).
3. The J-lens and R-lens were fitted by a third party on 25 prompts and do not cover the last layer; they corroborate the logit lens but were not independently validated here.
4. The patching design replaces all filler positions at once; it shows the answer position reads from them, not which positions or heads.
5. No scrambled-statement or made-up-entity control yet, so "meaningful text with something to recall" is not separated from "any text of matching length".
6. Not tested: whether the statement positions simultaneously represent the statement content (its next word, its falseness) and the arithmetic — the most interesting mechanistic question, and answerable from the stored readouts.

## 7. Next steps

- **Dual-use analysis (no GPU):** at the layers and positions where y is decodable, check whether the raw (un-subtracted) top-1 is the statement's next word, i.e. reading and computing superposed.
- **Controls (~2 GPU-hours):** scrambled statements; statements about made-up entities; patching with B from a *different* statement set.
- **Breadth:** a second model (Opus 4.5 behaviorally; Qwen3.6-27B mechanistically) or a second task where DeepSeek gains.
- **Earliest-layer comparison** of the three lenses from the stored per-layer curves.

## 8. Artifacts and cost

- Repo: `fst/` (tasks, statements, prompts, API runs), `gpu/` (pod setup patches, extraction, lenses, aggregation, patching, monitor), `data/runs/` (all API responses), `results/gpu/` (lens summary, patching results), this document.
- HF dataset `andreayhchen/filler-token-facts-states`: 1,200 state files (~37 GB), 1,200 readouts (2.3 GB), prompts, logs.
- Cost: API ≈ $3.50 total; pod ≈ 16 hours of 2× RTX PRO 6000 at ~$3.40–4.20/h ≈ $55–65, of which ~7 hours were idle after two silent failures caused by the volume's ~27 GB quota (fixed by moving outputs to local disk and adding a heartbeat monitor).
