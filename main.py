"""CLI entry point for the Embedding Analysis Agent.

Usage:
    python main.py --query "Find the path from 'king' to 'queen'"
    python main.py --query "Which tokens are most central?" --model qwen2.5:7b
    python main.py --interactive
    python main.py --graph custom_graph.json --query "Analyse bias"

Flags:
    --query TEXT      Single query to run and exit.
    --graph PATH      Path to graph JSON file (default: node_clusters_with_weights.json).
    --model TEXT      Ollama model name (default: qwen2.5:7b).
    --interactive     Start an interactive REPL session (graph loaded once).
    --test-mode       Use PydanticAI TestModel instead of Ollama (for testing/CI).
"""
import argparse
import sys
from pathlib import Path

from pydantic_ai.usage import UsageLimits

from agent_tools import Deps
from graph_structure import TokenGraph
from topology_agent import create_agent

DEFAULT_GRAPH = "node_clusters_with_weights.json"
DEFAULT_MODEL = "qwen2.5:7b"
MAX_REQUESTS = 20  # safety cap to prevent runaway tool loops


def load_graph(graph_path: str) -> TokenGraph:
    """Load the token graph from a JSON file."""
    path = Path(graph_path)
    if not path.exists():
        print(f"Error: graph file not found: {path.resolve()}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading graph from {path}...", file=sys.stderr)
    graph = TokenGraph.from_json(str(path))
    if not hasattr(graph, "edge_weights"):
        graph.edge_weights = {}
    print(f"Graph loaded: {len(graph.nodes)} nodes.", file=sys.stderr)
    return graph


def run_query(agent, deps: Deps, query: str, history: list | None = None) -> tuple[str, list]:
    """Run a single query and return (response_text, updated_message_history)."""
    result = agent.run_sync(
        query,
        deps=deps,
        message_history=history or [],
        usage_limits=UsageLimits(request_limit=MAX_REQUESTS),
    )
    return result.output, result.all_messages()


def run_single_query(graph_path: str, model_name: str, query: str, test_mode: bool) -> None:
    """Run one query and print the result."""
    graph = load_graph(graph_path)
    agent = create_agent(model_name)

    if test_mode:
        from pydantic_ai.models.test import TestModel
        with agent.override(model=TestModel(call_tools=[])):
            result = agent.run_sync(query, deps=Deps(graph=graph))
        print(result.output)
        return

    try:
        response, _ = run_query(agent, Deps(graph=graph), query)
        print(response)
    except (ConnectionRefusedError, OSError) as e:
        print(f"Error: cannot reach Ollama at localhost:11434 — is it running?\n{e}", file=sys.stderr)
        sys.exit(1)


def run_interactive(graph_path: str, model_name: str, test_mode: bool) -> None:
    """Start an interactive REPL. Graph is loaded once and reused."""
    graph = load_graph(graph_path)
    agent = create_agent(model_name)
    deps = Deps(graph=graph)
    history = []

    print("Embedding Analysis Agent — interactive mode")
    print("Graph: {:,} nodes | Model: {}".format(len(graph.nodes), model_name))
    print('Type "exit" or press Ctrl+C to quit.\n')

    while True:
        try:
            query = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        if test_mode:
            from pydantic_ai.models.test import TestModel
            with agent.override(model=TestModel()):
                result = agent.run_sync(query, deps=deps)
            print(result.output)
            continue

        try:
            response, history = run_query(agent, deps, query, history)
            print(response)
            print()
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embedding Analysis Agent — topology-aware token graph analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--query", "-q",
        metavar="TEXT",
        help="Single query to run.",
    )
    parser.add_argument(
        "--graph", "-g",
        metavar="PATH",
        default=DEFAULT_GRAPH,
        help=f"Path to graph JSON file (default: {DEFAULT_GRAPH}).",
    )
    parser.add_argument(
        "--model", "-m",
        metavar="NAME",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive REPL session.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use PydanticAI TestModel instead of Ollama (for testing/CI).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.query and not args.interactive:
        parser.print_usage(sys.stderr)
        print("Error: provide --query TEXT or --interactive", file=sys.stderr)
        sys.exit(1)

    if args.interactive:
        run_interactive(args.graph, args.model, args.test_mode)
    else:
        run_single_query(args.graph, args.model, args.query, args.test_mode)


if __name__ == "__main__":
    main()
