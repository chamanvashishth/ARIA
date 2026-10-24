"""Command-line interface (CLI) for ARIA.

Provides commands to inspect system status, launch the local HTTP API,
run the Flask ChatGPT web interface, execute benchmarks, and interactively chat.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from aria import __version__
from aria.config import AriaConfig
from aria.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="aria",
        description="ARIA — Autonomous Research & Intelligence Architecture",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"ARIA {__version__}",
        help="Show ARIA version and exit",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: status (or default)
    status_parser = subparsers.add_parser(
        "status",
        help="Inspect system readiness, devices, and directories",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Output diagnostics in JSON format",
    )

    # Command: serve (HTTP API)
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the local dependency-free ARIA HTTP API server",
    )
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")

    # Command: web (Flask ChatGPT studio)
    web_parser = subparsers.add_parser(
        "web",
        help="Launch the ARIA ChatGPT Studio web interface",
    )
    web_parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    web_parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    web_parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")

    # Command: chat (Console SLM Chat)
    chat_parser = subparsers.add_parser(
        "chat",
        help="Interactive command-line chat with the ARIA neural SLM",
    )
    chat_parser.add_argument("--tokens", type=int, default=32, help="Max new tokens (default: 32)")

    # Command: benchmark
    subparsers.add_parser(
        "benchmark",
        help="Run ARIA compute backend benchmarks (VCPU, VGPU, VNPU)",
    )

    return parser


def cmd_status(args: argparse.Namespace, config: AriaConfig) -> None:
    """Display system diagnostics and readiness."""
    import json
    from aria.compute.backends import VirtualCPUBackend, VirtualGPUBackend, VirtualNPUBackend
    from aria.tokenizer import ByteTokenizer

    tok = ByteTokenizer()
    diagnostics = {
        "version": __version__,
        "status": "engineering foundation ready",
        "data_dir": str(config.data_dir),
        "tokenizer_vocab_size": tok.vocab_size(),
        "compute_backends": [
            VirtualCPUBackend.__name__,
            VirtualGPUBackend.__name__,
            VirtualNPUBackend.__name__,
        ],
    }

    if getattr(args, "json", False):
        print(json.dumps(diagnostics, indent=2))
    else:
        print(f"ARIA {__version__} — engineering foundation ready")
        print(f"  Data Directory : {config.data_dir}")
        print(f"  Tokenizer Vocab: {tok.vocab_size()} bytes")
        print(f"  Compute Devices: {', '.join(diagnostics['compute_backends'])}")


def cmd_serve(args: argparse.Namespace) -> None:
    """Launch the dependency-free local HTTP API server."""
    from aria.api import create_api_server
    from aria.brain import TransformerLanguageModel
    from aria.runtime import AriaRuntime

    print(f"[*] Initializing ARIA Neural Brain & Runtime...")
    model = TransformerLanguageModel(
        vocab_size=260,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        max_sequence_length=128,
        seed=42,
    )
    runtime = AriaRuntime(model)
    runtime.start()

    server = create_api_server(runtime, host=args.host, port=args.port)
    print(f"[+] ARIA API Server running at: http://{args.host}:{args.port}")
    print("    Endpoints: /health, /status, /generate (POST), / (UI)")
    print("    Press Ctrl+C to terminate.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down ARIA API Server...")
    finally:
        server.server_close()
        runtime.stop()


def cmd_web(args: argparse.Namespace) -> None:
    """Launch the Flask web application."""
    try:
        from app import create_app
    except ImportError:
        print("[-] Flask or scikit-learn is not installed. Install via: pip install .[web]")
        sys.exit(1)

    app = create_app()
    print("=" * 65)
    print(f"🤖 ARIA & ChatGPT Studio Running at: http://{args.host}:{args.port}")
    print("=" * 65)
    app.run(host=args.host, port=args.port, debug=args.debug)


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive command-line chat using the ARIA Transformer SLM."""
    from aria.brain import TransformerLanguageModel
    from aria.runtime import AriaRuntime, RuntimeConfig
    from aria.tokenizer import ByteTokenizer

    print(f"[*] Loading ARIA SLM Interactive Console...")
    tok = ByteTokenizer()
    model = TransformerLanguageModel(
        vocab_size=tok.vocab_size(),
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        max_sequence_length=128,
        seed=42,
    )
    runtime = AriaRuntime(model=model, tokenizer=tok)
    runtime.start()

    print("ARIA SLM Console Ready. Type 'quit' to exit.\n")
    while True:
        try:
            prompt = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ["quit", "exit", "q"]:
            print("Session ended.")
            break

        result = runtime.generate(prompt, config=RuntimeConfig(max_new_tokens=args.tokens))
        print(f"ARIA: {result.text}\n")

    runtime.stop()


def cmd_benchmark() -> None:
    """Run compute benchmarks."""
    import subprocess
    benchmark_script = AriaConfig().data_dir.parent / "scripts" / "benchmark_compute.py"
    if benchmark_script.exists():
        subprocess.run([sys.executable, str(benchmark_script)])
    else:
        print(f"[-] Benchmark script not found at: {benchmark_script}")


def main(argv: Sequence[str] | None = None) -> None:
    """CLI Entry point."""
    config = AriaConfig()
    config.ensure_directories()
    configure_logging(config.log_level)

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # Default behavior: prints version & foundation ready (ensuring backwards compatibility)
        print(f"ARIA {__version__} — engineering foundation ready")
    elif args.command == "status":
        cmd_status(args, config)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "web":
        cmd_web(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "benchmark":
        cmd_benchmark()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
