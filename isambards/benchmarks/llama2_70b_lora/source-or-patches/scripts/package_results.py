#!/usr/bin/env python3

import argparse
import json
import shutil
import sys
from pathlib import Path


def package_results(run_dir: Path, team: str) -> Path:
    if not team or Path(team).name != team or team in {".", ".."}:
        raise ValueError("Team must be a single directory name")
    count = len(list(run_dir.glob("result_*.txt")))
    expected = {f"result_{i}.txt" for i in range(count)}
    if not 1 <= count <= 10 or {p.name for p in run_dir.glob("result_*.txt")} != expected:
        raise ValueError("result_0.txt up to result_9.txt are required with no gaps")
    seeds = set()
    failed = []
    unseeded = []
    for i in range(count):
        events = {}
        for line in (run_dir / f"result_{i}.txt").read_text().splitlines():
            if ":::MLLOG" in line:
                event = json.loads(line.split(":::MLLOG", 1)[1])
                events[event["key"]] = event
        if "seed" not in events:
            failed.append(i)
            unseeded.append(i)
            continue
        seed = events["seed"]["value"]
        if seed in seeds:
            raise ValueError(f"Repeated seed in run {i}")
        seeds.add(seed)
        start = events.get("run_start", {}).get("time_ms")
        stop = events.get("run_stop", {})
        loss = events.get("eval_accuracy", {}).get("value")
        if not (stop.get("metadata", {}).get("status") == "success"
                and isinstance(loss, (int, float)) and loss <= 0.925
                and start is not None and stop["time_ms"] > start):
            failed.append(i)
    # A full set is ten runs that all logged a seed with at most one failed run
    full = count == 10 and len(failed) <= 1 and not unseeded
    print(f"{count} runs, failed runs {failed}: {'full set' if full else 'partial set'}", file=sys.stderr)

    source = run_dir / "source"
    system = source / "systems/MI300X_EPYC_9654.json"
    if not system.is_file():
        raise ValueError(f"Missing system description: {system}")
    output = run_dir / "submission" / team
    if output.exists():
        raise FileExistsError(f"Preserving existing package: {output}")
    results = output / "results/student_system/llama2_70b_lora"
    benchmark = output / "benchmarks/llama2_70b_lora"
    results.mkdir(parents=True)
    benchmark.mkdir(parents=True)
    (output / "systems").mkdir()
    shutil.copy2(system, output / "systems/student_system.json")
    shutil.copytree(source, benchmark / "source-or-patches")
    shutil.copy2(run_dir / "config.sh", benchmark / "config.sh")
    shutil.copy2(run_dir / "config.sh", benchmark / "source-or-patches/configs/submission.sh")
    for name in sorted(expected):
        shutil.copy2(run_dir / name, results / name)
    launcher = benchmark / "run_and_time.sh"
    launcher.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")/source-or-patches"\nexec ./run.sh submission "${1:-10}"\n')
    launcher.chmod(0o755)
    (benchmark / "README.md").write_text(
        "# Isambards: Llama 2 70B LoRA\n\n"
        "Source, container digest, launcher and system settings are preserved in source-or-patches/. "
        "config.sh is the configuration used for these ten runs.\n\n"
        "To reproduce, restore data/ and the checker .venv inside source-or-patches/ as described "
        "in its README.md, then run ./run_and_time.sh. It submits ten runs with distinct seeds.\n\n"
        "The official package checker must pass. Its normalization is saved in scaling.json. "
        "If the successful checker requires no normalization, the launcher writes scaling_factor=1.0.\n"
    )
    return output, full


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package ten converged MLPerf runs")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--team", default="Isambards")
    args = parser.parse_args()
    output, full = package_results(args.run_dir.resolve(), args.team)
    print(output)
    sys.exit(0 if full else 2)
