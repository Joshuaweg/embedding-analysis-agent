# Session Handoff — 2026-03-11

## What was accomplished

### KERNEL initialized
- `_meta/agentdb/agent.db`, `.claude/CLAUDE.md`, `.claude/rules/project.md`, `_meta/context/active.md`

### Research direction
**Question:** Can LLM word embeddings determine learned value systems? (AI alignment framing)
**Methodology:** WEAT on raw GPT-2 embeddings + topological analysis via Mapper graph
**Plan:** `/home/joshua/_meta/plans/lovely-waddling-thimble.md`
**Prior art:** No published work combines TDA/Mapper + MFT + transformer embeddings. Closest: Yan et al. arXiv 2507.18607 (Mapper on BERT, July 2025).

### Code on master (85 tests passing)
New: `value_lexicons.py`, `value_analysis.py`, `tests/test_value_lexicons.py`, `_meta/research/value_analysis_results.{json,txt}`
Modified: `agent_tools.py` (+3 tools), `topology_agent.py` (19 tools total), `system_prompts.py`, `tests/test_topology_agent.py`

PyTorch CPU installed. GPT-2 weights cached.
Embedding load pattern (matches embedding.py):
```python
from transformers import GPT2Model
model = GPT2Model.from_pretrained('gpt2')
matrix = model.get_input_embeddings().weight.detach().numpy()  # (50257, 768)
```

### Fast-mode results summary
**Exp 1 — Pole Separation:** All 5 MFT dimensions ratio > 1 (range 1.64–2.57). Negative poles tighter than positive consistently (e.g. purity- intra=1.67, purity+ intra=10.0).

**Exp 2 — Community Coherence:** All poles exceed null baseline. purity_degradation.negative = 1.0 (8.2× null) — all 10 tokens in one community.

**Exp 3 — WEAT:** Standard gender/valence: d=-0.937. MFT gender/care-harm: d=-1.172 (+25%). Novel finding: MFT attributes more sensitive than generic valence.

**Exp 4 — Demographic projections:**
- Care/harm: female +0.020, male -0.106 (female more care-associated)
- Fairness: European +0.186, African +0.060 (gap)
- Purity: European +0.028, African **-0.152** (largest gap in dataset = 0.18)
- Loyalty: ALL groups negative — unexpected, needs investigation

## Next session TODO
1. Run full `value_analysis.py` (drop --fast) for paper-quality results
2. Investigate loyalty/betrayal axis — all demographics lean betrayal-negative (lexicon or axis issue?)
3. Verify WEAT sign convention: negative d = female more care-associated?
4. fairness_cheating.positive intra-distance = 9.67 — check sampling stability in full run
5. Add religion/nationality demographic sets to Exp 3+4
6. Begin paper structure (Intro → Related Work → Methods → Results → Discussion)

## Run commands
```bash
.venv/bin/python value_analysis.py            # full paper-quality run
.venv/bin/python value_analysis.py --fast     # quick validation
.venv/bin/python value_analysis.py --skip-embeddings  # graph-only
.venv/bin/python -m pytest tests/             # 85 tests
```
