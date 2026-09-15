# False statements as filler

Can language models use appended **false statements** the way they use meaningless filler tokens,
as extra positions for hidden, no-chain-of-thought computation?

Builds on:
- Brauer et al., [*Reading Between the Dots*](https://arxiv.org/abs/2607.03502): dots/counting filler improves no-CoT accuracy and the computation is decodable at filler positions ([code](https://github.com/kaleybrauer/filler-token-reasoning)).
- Baherwani et al., [*Not All LLM Reasoning is Visible in the Chain-of-Thought*](https://arxiv.org/abs/2607.22925): semantic word fillers help some models (Opus 4.5) and hurt others (Qwen3.5-397B).

## Conditions

| Condition | Filler between question and `Answer:` |
|---|---|
| `none` | — |
| `counting` | `1 2 3 …`, length matched to the statement block (positive control) |
| `false` | ~10 false statements (~110 tokens) |
| `true` | the same statements with the true object (matched control) |

Planned later: scrambled statements, false number facts, animal words, statements about made-up entities.

## Pipeline

```bash
pip install -r requirements.txt
cp .env.example .env               # then fill in ANTHROPIC_API_KEY and OPENROUTER_API_KEY

python -m fst.statements pairs     # token-matched true/false pairs from data/statements/curated_spec.json
python -m fst.verify               # keep pairs every model labels correctly (~$0.10)
python -m fst.statements sets      # 3 disjoint sets of 10 pairs from the verified pairs
python -m fst.pilot build          # write prompts, print cost estimate (free)
python -m fst.pilot run            # pilot: 3 tasks x 150 problems x 4 conditions per model (~$10 total)
python -m fst.pilot report         # accuracy, paired differences vs. none (McNemar), format failures
pytest
```

## Results so far

Systems of equations, 300 problems, 10-shot, via OpenRouter (`python -m fst.pilot report --run soe300`).
Accuracy by condition; the change vs. no filler is paired over the same problems.

| Condition | V4 Flash | V3-0324 |
|---|---|---|
| none | 52.0% | 33.3% |
| counting (after question) | 69.3% (+17.3) | 56.0% (+22.7) |
| **false (after question)** | **69.3% (+17.3)** | **53.7% (+20.3)** |
| true (after question) | 68.7% (+16.7) | 54.7% (+21.3) |
| counting / false / true, before the problem | 47.7 / 43.0 / 43.0% | 25.3 / 22.7 / 23.7% |
| counting / false / true, mid-definitions | 48.7 / 47.3 / 45.3% | 27.3 / 28.7 / 27.0% |

- False statements help as much as counting filler. The false-minus-counting difference is +0.0% for V4 Flash (95% CI -3.9 to +3.9) and -2.3% for V3 (-7.2 to +2.5).
- Truth value makes no detectable difference.
- The gain requires filler after the question. Before the problem it hurts, as paper 1 found for "before". Mid-definitions it doesn't help even when both chain variables precede the filler.
- V3 reproduces paper 1's counting-filler gain.
- V4 Flash sometimes ends its reply immediately with no answer. This happens in 10 of 300 no-filler prompts and a few mid-definition prompts; those replies are scored as wrong. Excluding them, the after-question gains are +16.6% (CI +11.5 to +21.6).
- In the 150-problem pilot, arithmetic showed no gain on either model, and variable counting was at ceiling on V4 Flash and hurt by all fillers on V3.

## Design decisions

- **Placement** follows paper 1: filler sits in the user turn after the question under a `Filler:` label, and every few-shot example carries the same filler. No assistant prefill.
- **No filler hint in the system prompt.** Paper 1 told the model the filler gives it space to think. Paper 2 found the hint has no effect, and giving it for false statements would confound the result.
- **Tasks need no fact recall.** Reading factual statements triggers the model's own recall at those positions, so recall-based tasks (paper 1's fact addition, letter position) are excluded.
  - `system_of_equations`: paper 1's task, with every intermediate kept in 0–999. DeepSeek tokenizes those numbers as single tokens, so the logit lens can find them.
  - `arithmetic`: paper 2's task. Integer division and modulo take only a non-negative left operand and a positive right operand, so rounding conventions can't disagree.
  - `variable_counting`: paper 2's task. Paper 2 published no generator, so this snippet design is ours.
- **Statements** are hand-curated facts drawn from PopQA (author, composer, director, sport), written in our own templates.
  - Titles get a type word ("the film Alien") to avoid ambiguity.
  - No numbers appear in any statement.
  - False objects come from same-kind pools and match the true object's token length under each open model's tokenizer.
  - Geometry of Truth and CounterFact were avoided because they're heavily used in truth-probing work.
- **Statement sets are fixed** within a condition, so paper 1's cross-example mean subtraction removes content shared by every problem.
- **Models:**
  - Opus 4.5, via the Anthropic Batch API.
  - DeepSeek V4 Flash (original 0423 checkpoint, matching the pre-fitted J-lens at `camilablank/workspace-lenses`), via OpenRouter with fp8 hosts pinned and reasoning off.
  - DeepSeek V3-0324, as a reference that should reproduce paper 1.
  - Qwen3.6-27B, as the fallback interpretability model.
