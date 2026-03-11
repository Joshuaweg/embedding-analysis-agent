# Project Rules

## Tech Stack
- Python only — no JS/TS additions
- Use pydantic-ai 1.67.0 API (output_type, result.output, OpenAIChatModel + OpenAIProvider)
- All new tools must use `RunContext[Deps]` pattern matching agent_tools.py

## File Organization
- Core logic: top-level `.py` files
- Tests: `tests/` directory, mark integration tests with `@pytest.mark.integration`
- Precomputed data: JSON at root level
- KERNEL state: `_meta/` (never delete)

## Code Style
- Type annotations on all new functions
- Pydantic models for structured inputs/outputs
- Async functions for all agent tools

## Testing
- Run: `.venv/bin/python -m pytest tests/`
- Unit tests must not require Ollama
- Use `TestModel(call_tools=[])` when tool execution isn't needed
- Add regression test for every bug fixed

## Never Do
- Never modify `graph_structure.py`
- Never commit `node_clusters_with_weights.json` changes (it's the precomputed graph)
- Never use system python/pip (externally managed)
- Never skip tests before committing
- Never commit `.env` or API keys
- Never delete `_meta/` folder
