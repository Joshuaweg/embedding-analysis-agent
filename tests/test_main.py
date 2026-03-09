"""Tests for main.py — CLI entry point behavior.

Tests use --test-mode flag which causes main.py to use PydanticAI's TestModel
instead of real Ollama, allowing CLI tests to run without Ollama running.

Integration tests (requiring real Ollama) are marked @pytest.mark.integration.
"""
import subprocess
import sys
import pytest
from pathlib import Path


def run_main(*args, timeout: int = 30) -> subprocess.CompletedProcess:
    """Helper: run main.py as a subprocess with the given args."""
    return subprocess.run(
        [sys.executable, "main.py", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Argument parsing — does not require Ollama
# ---------------------------------------------------------------------------

def test_cli_no_args_exits_nonzero():
    result = run_main(timeout=10)
    assert result.returncode != 0


def test_cli_missing_query_and_no_interactive_exits_nonzero(small_graph_json):
    result = run_main("--graph", str(small_graph_json), timeout=10)
    assert result.returncode != 0


def test_cli_missing_graph_file_exits_nonzero():
    result = run_main(
        "--graph", "nonexistent_graph_xyz.json",
        "--query", "test",
        timeout=10,
    )
    assert result.returncode != 0


def test_cli_missing_graph_error_message_is_clear():
    result = run_main(
        "--graph", "nonexistent_graph_xyz.json",
        "--query", "test",
        timeout=10,
    )
    stderr_lower = result.stderr.lower()
    assert "not found" in stderr_lower or "no such file" in stderr_lower or "error" in stderr_lower


def test_cli_unknown_flag_exits_nonzero():
    result = run_main("--not-a-real-flag", timeout=10)
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# Functional tests — use --test-mode to avoid Ollama dependency
# ---------------------------------------------------------------------------

def test_cli_query_with_test_mode_exits_zero(small_graph_json):
    result = run_main(
        "--graph", str(small_graph_json),
        "--query", "What nodes exist?",
        "--test-mode",
        timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"


def test_cli_query_produces_output(small_graph_json):
    result = run_main(
        "--graph", str(small_graph_json),
        "--query", "What nodes exist?",
        "--test-mode",
        timeout=30,
    )
    assert len(result.stdout.strip()) > 0


def test_cli_model_flag_accepted_and_forwarded(small_graph_json):
    """--model flag must be parsed by argparse and forwarded to agent creation without error."""
    result = run_main(
        "--graph", str(small_graph_json),
        "--model", "deepseek-r1:7b",
        "--query", "test query",
        "--test-mode",
        timeout=30,
    )
    assert "unrecognized" not in result.stderr.lower()
    assert "error: argument --model" not in result.stderr.lower()
    assert result.returncode == 0, f"stderr: {result.stderr}"


# ---------------------------------------------------------------------------
# Integration tests — require Ollama running locally
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_cli_query_with_real_ollama(small_graph_json):
    result = run_main(
        "--graph", str(small_graph_json),
        "--query", "What tokens are in node cube0_cluster0?",
        "--model", "llama3.1:8b",
        timeout=120,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert len(result.stdout.strip()) > 0
