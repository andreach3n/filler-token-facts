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
