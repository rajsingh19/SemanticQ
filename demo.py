#!/usr/bin/env python3
"""demo.py: Demonstration script for question-dedup project."""

import os
import sys
from pathlib import Path

# Ensure 'src' is available on python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from qdedup.llm import MockLLM, get_llm_client

load_dotenv()
console = Console()


def main() -> None:
    console.print(Panel("[bold green]question-dedup Demo[/bold green]", expand=False))

    api_key_set = bool(os.getenv("GEMINI_API_KEY"))
    console.print(f"GEMINI_API_KEY detected: [bold]{api_key_set}[/bold]")

    # Demonstrate MockLLM (runs without any API key)
    console.print("\n[bold cyan]1. Testing MockLLM (Deterministic / Offline):[/bold cyan]")
    mock = MockLLM()
    mock.register_response("duplicate check", "Duplicate found: Question ID #42")

    res1 = mock.generate("Run a duplicate check for: How do I sort a dict?")
    console.print(f"Mock Response 1: [green]{res1}[/green]")

    res2 = mock.generate("What is the meaning of life?")
    console.print(f"Mock Response 2: [green]{res2}[/green]")

    # Demonstrate client factory
    console.print("\n[bold cyan]2. Testing get_llm_client() fallback:[/bold cyan]")
    client = get_llm_client()
    console.print(f"Active client type: [yellow]{type(client).__name__}[/yellow]")
    client_res = client.generate("Hello world")
    console.print(f"Response: [green]{client_res}[/green]")


if __name__ == "__main__":
    main()
