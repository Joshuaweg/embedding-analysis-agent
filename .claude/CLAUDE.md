# Embedding Analysis Agent

## Tech Stack
- Language: Python 3 (venv at `.venv/`)
- Framework: pydantic-ai 1.67.0 + Ollama (OpenAI-compat endpoint)
- Graph: networkx, TDA/Mapper algorithm
- ML: transformers (tokenizers only), numpy, scipy, scikit-learn, umap-learn
- Testing: pytest + pytest-asyncio
- Package manager: pip (use `.venv/bin/pip` — system pip is locked)

## Project Structure
```
main.py                        # CLI entrypoint
topology_agent.py              # Agent factory (create_agent())
agent_tools.py                 # 16 PydanticAI tool functions (RunContext[Deps])
graph_structure.py             # TokenGraph + Node (DO NOT MODIFY)
system_prompts.py              # TOPOLOGY_AGENT_PROMPT
node_clusters_with_weights.json # Precomputed graph (~11MB, 9020 nodes, 36331 edges)
tests/                         # Unit + integration tests
_meta/                         # KERNEL state (agentdb, context, plans, etc.)
```

## KERNEL Integration

**Always start with `/ingest`** (or `/kernel:ingest` in terminal)

Routing:
- Tier 1 (1-2 files): Execute directly
- Tier 2 (3-5 files): Spawn surgeon agent
- Tier 3 (6+ files): Surgeon + adversary

**Run `/handoff` before closing** to save progress.

## Critical Rules
- `graph_structure.py` is stable — do NOT modify it
- Always use `.venv/bin/python` / `.venv/bin/pip`, never system python
- Run tests with: `.venv/bin/python -m pytest tests/` (integration excluded by default)
- pydantic-ai API: `output_type=` (not `result_type=`), `result.output` (not `result.data`)
- `OpenAIModel` deprecated → use `OpenAIChatModel` with `OpenAIProvider`

## Commands

| Command | What It Does |
|---------|--------------|
| `/ingest` | Start any task |
| `/validate` | Pre-commit checks |
| `/handoff` | Save progress |
| `/tearitapart` | Review plan before implementing |

## Running the Agent
```bash
# Single query (requires Ollama at localhost:11434)
.venv/bin/python main.py --query "Find path from 'king' to 'queen'"

# Interactive mode
.venv/bin/python main.py --interactive

# Test mode (no Ollama needed)
.venv/bin/python main.py --query "test" --test-mode
```
