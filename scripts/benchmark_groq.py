#!/usr/bin/env python3
"""Projeto JOI - Groq API latency benchmark.

Measures key performance metrics for the Groq integration:
- TTFT (Time To First Token): critical for perceived latency
- Total response time
- Tokens per second (throughput)
- Stream chunk count and distribution

Usage:
    # Set your Groq API key first
    export GROQ_API_KEY="gsk_..."

    # Run benchmark with default prompts
    python scripts/benchmark_groq.py

    # Run with custom prompt
    python scripts/benchmark_groq.py --prompt "Explain quantum computing in 3 sentences"

    # Run multiple iterations for statistical significance
    python scripts/benchmark_groq.py --iterations 5

    # Compare models
    python scripts/benchmark_groq.py --model llama-3.1-8b-instant
    python scripts/benchmark_groq.py --model llama-3.1-70b-versatile

Output:
    Console table with per-iteration metrics + summary statistics.
    JSON report saved to scripts/benchmark_results.json for trend tracking.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logging import get_logger  # noqa: E402
from app.llm.base import LLMMessage, LLMRequest, LLMRole, TokenChunk  # noqa: E402
from app.llm.errors import LLMError  # noqa: E402
from app.llm.groq_client import GroqClient  # noqa: E402

logger = get_logger(__name__)


# ─── Default benchmark prompts (varied length/complexity) ────────────────

DEFAULT_PROMPTS = [
    "Olá, quem é você?",  # short, simple
    "Explique o conceito de memória vetorial em três frases.",  # medium, technical
    "Escreva um parágrafo curto sobre a importância da empatia em sistemas de IA conversacional, considerando tanto benefícios quanto riscos éticos.",  # long, complex
]


@dataclass
class IterationResult:
    """Metrics from a single benchmark iteration."""

    prompt: str
    model: str
    ttft_ms: float | None  # Time to first token
    total_ms: float  # Total request time
    chunk_count: int  # Number of stream chunks received
    completion_tokens: int
    prompt_tokens: int
    tokens_per_second: float  # completion_tokens / (total_ms - ttft_ms) * 1000
    finish_reason: str
    error: str | None = None


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report."""

    timestamp: str
    model: str
    iterations: list[IterationResult] = field(default_factory=list)

    @property
    def successful(self) -> list[IterationResult]:
        return [r for r in self.iterations if r.error is None]

    def summary(self) -> dict[str, Any]:
        """Compute summary statistics over successful iterations."""
        if not self.successful:
            return {"error": "no successful iterations"}

        ttfts = [r.ttft_ms for r in self.successful if r.ttft_ms is not None]
        totals = [r.total_ms for r in self.successful]
        tps = [r.tokens_per_second for r in self.successful]

        return {
            "iterations_total": len(self.iterations),
            "iterations_successful": len(self.successful),
            "iterations_failed": len(self.iterations) - len(self.successful),
            "ttft_ms": {
                "mean": statistics.mean(ttfts) if ttfts else None,
                "median": statistics.median(ttfts) if ttfts else None,
                "min": min(ttfts) if ttfts else None,
                "max": max(ttfts) if ttfts else None,
                "stdev": statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0,
            },
            "total_ms": {
                "mean": statistics.mean(totals),
                "median": statistics.median(totals),
                "min": min(totals),
                "max": max(totals),
            },
            "tokens_per_second": {
                "mean": statistics.mean(tps),
                "median": statistics.median(tps),
                "min": min(tps),
                "max": max(tps),
            },
        }


# ─── Benchmark runner ────────────────────────────────────────────────────


async def benchmark_single(
    client: GroqClient,
    prompt: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> IterationResult:
    """Run a single benchmark iteration with streaming."""
    request = LLMRequest(
        messages=[LLMMessage(role=LLMRole.USER, content=prompt)],
        model=model,
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
        request_id=f"bench-{int(time.monotonic() * 1000)}",
    )

    start = time.monotonic()
    ttft_ms: float | None = None
    chunk_count = 0
    completion_tokens = 0
    prompt_tokens = 0
    finish_reason = "stop"

    try:
        async for chunk in client.stream(request):
            chunk_count += 1
            if ttft_ms is None:
                ttft_ms = (time.monotonic() - start) * 1000

            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason

        total_ms = (time.monotonic() - start) * 1000

        # Tokens per second (excluding TTFT, since that's setup time)
        generation_ms = total_ms - (ttft_ms or 0)
        tps = (completion_tokens / generation_ms * 1000) if generation_ms > 0 else 0.0

        return IterationResult(
            prompt=prompt,
            model=model,
            ttft_ms=round(ttft_ms, 2) if ttft_ms else None,
            total_ms=round(total_ms, 2),
            chunk_count=chunk_count,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            tokens_per_second=round(tps, 2),
            finish_reason=finish_reason,
        )

    except LLMError as exc:
        total_ms = (time.monotonic() - start) * 1000
        return IterationResult(
            prompt=prompt,
            model=model,
            ttft_ms=ttft_ms,
            total_ms=round(total_ms, 2),
            chunk_count=chunk_count,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            tokens_per_second=0.0,
            finish_reason="error",
            error=f"{type(exc).__name__}: {exc.message}",
        )


async def run_benchmark(
    prompts: list[str],
    model: str,
    iterations: int,
    temperature: float,
    max_tokens: int,
) -> BenchmarkReport:
    """Run the benchmark across all prompts and iterations."""
    report = BenchmarkReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model,
    )

    try:
        client = GroqClient(default_model=model)
    except LLMError as exc:
        print(f"❌ Failed to initialize Groq client: {exc.message}")
        print("   Make sure GROQ_API_KEY is set in your environment.")
        sys.exit(1)

    try:
        for prompt_idx, prompt in enumerate(prompts, 1):
            print(f"\n{'─' * 70}")
            print(f"  Prompt {prompt_idx}/{len(prompts)}: {prompt[:60]}...")
            print(f"{'─' * 70}")

            for it in range(1, iterations + 1):
                print(f"  [{it}/{iterations}] ", end="", flush=True)
                result = await benchmark_single(
                    client=client,
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                report.iterations.append(result)

                if result.error:
                    print(f"❌ ERROR: {result.error}")
                else:
                    print(
                        f"✅ TTFT={result.ttft_ms}ms  "
                        f"Total={result.total_ms}ms  "
                        f"Tokens={result.completion_tokens}  "
                        f"TPS={result.tokens_per_second}"
                    )

        # Close the client cleanly
        await client.close()

    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user.")

    return report


# ─── Display ──────────────────────────────────────────────────────────────


def print_report(report: BenchmarkReport) -> None:
    """Print a formatted summary table to the console."""
    summary = report.summary()

    print(f"\n{'═' * 70}")
    print(f"  BENCHMARK SUMMARY — Model: {report.model}")
    print(f"  Timestamp: {report.timestamp}")
    print(f"{'═' * 70}")

    if "error" in summary:
        print(f"  ❌ {summary['error']}")
        return

    print(f"  Iterations:  {summary['iterations_successful']}/{summary['iterations_total']} successful")
    print()

    def fmt_stats(stats: dict[str, float | None]) -> str:
        if stats.get("mean") is None:
            return "N/A"
        parts = [f"mean={stats['mean']:.1f}"]
        if "median" in stats:
            parts.append(f"median={stats['median']:.1f}")
        parts.append(f"min={stats['min']:.1f}")
        parts.append(f"max={stats['max']:.1f}")
        if "stdev" in stats and stats["stdev"]:
            parts.append(f"σ={stats['stdev']:.1f}")
        return "  ".join(parts)

    print(f"  TTFT (ms):           {fmt_stats(summary['ttft_ms'])}")
    print(f"  Total time (ms):     {fmt_stats(summary['total_ms'])}")
    print(f"  Tokens/second:       {fmt_stats(summary['tokens_per_second'])}")

    print(f"\n{'═' * 70}")
    print("  Interpretation guide:")
    print("  • TTFT < 400ms   = excellent (real-time chat feel)")
    print("  • TTFT 400-800ms = good (acceptable for chat)")
    print("  • TTFT > 800ms   = poor (users perceive lag)")
    print("  • TPS > 80       = good throughput for streaming")
    print(f"{'═' * 70}")


def save_report(report: BenchmarkReport, output_path: Path) -> None:
    """Save the report as JSON for trend tracking."""
    data = {
        "timestamp": report.timestamp,
        "model": report.model,
        "iterations": [asdict(r) for r in report.iterations],
        "summary": report.summary(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Report saved to: {output_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Groq API latency for Projeto JOI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run with default prompts
  %(prog)s --prompt "Hello world"             # Custom prompt
  %(prog)s --iterations 5                     # 5 iterations per prompt
  %(prog)s --model llama-3.1-8b-instant       # Use faster model
  %(prog)s --max-tokens 512                   # Allow longer responses
        """,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt (overrides default prompts)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-70b-versatile",
        help="Groq model ID (default: llama-3.1-70b-versatile)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Iterations per prompt (default: 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/benchmark_results.json"),
        help="Output JSON file for the report",
    )

    args = parser.parse_args()

    # Verify API key is set
    if not os.environ.get("GROQ_API_KEY") and not os.path.exists(PROJECT_ROOT / ".env"):
        print("❌ GROQ_API_KEY not found.")
        print("   Set it via:  export GROQ_API_KEY='gsk_...'")
        print("   Or add it to: .env file in project root")
        sys.exit(1)

    # Determine prompts
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    print(f"\n🚀 Projeto JOI — Groq API Benchmark")
    print(f"   Model:       {args.model}")
    print(f"   Prompts:     {len(prompts)}")
    print(f"   Iterations:  {args.iterations} per prompt")
    print(f"   Total runs:  {len(prompts) * args.iterations}")

    # Run benchmark
    report = asyncio.run(
        run_benchmark(
            prompts=prompts,
            model=args.model,
            iterations=args.iterations,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )

    # Display and save results
    print_report(report)
    save_report(report, args.output)


if __name__ == "__main__":
    main()
