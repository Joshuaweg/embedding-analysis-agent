# Embedding Analysis Agent

**Initialized**: 2026-03-09
**Tech Stack**: Python, pydantic-ai 1.67.0, networkx, Ollama, transformers
**Status**: Ready

## What This Project Is
TDA-based agent that maps semantic relationships in GPT-2 token embeddings.
Uses the Mapper algorithm to build a graph (~9,020 nodes, 36,331 edges) over 50,257 tokens (768-dim),
then exposes it via a pydantic-ai agent with 16 tools for graph traversal, similarity search, and topology analysis.

## Structure
- `main.py` — CLI entrypoint
- `topology_agent.py` — agent factory
- `agent_tools.py` — 16 tool functions
- `graph_structure.py` — graph model (stable, do not modify)
- `system_prompts.py` — system prompt
- `node_clusters_with_weights.json` — precomputed graph data
- `tests/` — 77 passing unit tests

## Current Focus
Newly initialized with KERNEL — ready for first task.
