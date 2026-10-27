import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location("phase1_monitor", Path(__file__).resolve().parents[1] / "scripts/monitor_phase1.py")
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def campaign(tmp_path, rewards, frames=100_000_000):
    runs = []
    for seed, reward in enumerate(rewards):
        out = tmp_path / f"seed{seed}"
        out.mkdir()
        (out / "log.jsonl").write_text(json.dumps(dict(frames=frames, best_R=reward, fps=40000, rnd_loss=0.2)) + "\n")
        runs.append(dict(seed=seed, status="running", started=time.time(), out_dir=str(out)))
    for seed in range(len(rewards), 4):
        runs.append(dict(seed=seed, status="pending", out_dir=str(tmp_path / f"seed{seed}")))
    monitor.atomic_json(tmp_path / "state.json", dict(status="running", seeds=[0, 1, 2, 3], runs=runs))


def test_failed_pair_stops_instead_of_replacing_bad_seeds(tmp_path):
    campaign(tmp_path, [10000, 15000])
    report = monitor.inspect(tmp_path)
    assert any("100M" in p for p in report["problems"])
    assert (tmp_path / "STATUS.md").exists()


def test_one_bad_seed_does_not_stop_a_healthy_pair(tmp_path):
    campaign(tmp_path, [10000, 60000])
    assert not monitor.inspect(tmp_path)["problems"]


def test_gate_waits_for_both_seeds_at_same_budget(tmp_path):
    campaign(tmp_path, [10000, 15000], frames=90_000_000)
    assert not monitor.inspect(tmp_path)["problems"]


def test_non_finite_metrics_are_a_failure(tmp_path):
    campaign(tmp_path, [float("nan"), 60000])
    assert any("non-finite best_R" in p for p in monitor.inspect(tmp_path)["problems"])


def test_launcher_refuses_to_bypass_queue(tmp_path):
    runner = Path(__file__).resolve().parents[1] / "scripts/validate_phase1.py"
    result = subprocess.run([sys.executable, str(runner), "--output", str(tmp_path / "forbidden")],
                            env={k: v for k, v in os.environ.items() if k != "AGENT_JOB_GPU"},
                            capture_output=True, text=True)
    assert result.returncode != 0 and "Submit this command through the hub" in result.stderr
    assert not (tmp_path / "forbidden").exists()


def test_independent_monitor_requests_stop_on_bad_curves(tmp_path):
    campaign(tmp_path, [10000, 15000])
    runner = Path(__file__).resolve().parents[1] / "scripts/monitor_phase1.py"
    result = subprocess.run([sys.executable, str(runner), "--root", str(tmp_path), "--once"],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "stop_requested.json").exists()
    assert (tmp_path / "monitor.jsonl").exists()


def test_inspect_flags_the_100M_pair_gate_as_early_stop(tmp_path):
    campaign(tmp_path, [10000, 15000])
    report = monitor.inspect(tmp_path)
    assert report["early_stop"] is True
    assert any("100M" in p for p in report["problems"])


def test_inspect_no_early_stop_when_pair_is_healthy(tmp_path):
    campaign(tmp_path, [10000, 60000])
    assert monitor.inspect(tmp_path)["early_stop"] is False


def test_inspect_health_problem_is_not_an_early_stop(tmp_path):
    campaign(tmp_path, [float("nan"), 60000])
    report = monitor.inspect(tmp_path)
    assert report["early_stop"] is False
    assert report["problems"]


def retry_campaign(tmp_path, stop):
    root = tmp_path / "campaign"
    root.mkdir()
    monitor.atomic_json(root / "state.json",
                        dict(status="stopped", stop=stop, reason="seeds [0, 1]: mean reward 14050 at 100M is below 20000",
                             seeds=[0, 1, 2, 3], runs=[dict(seed=s, status="pending", out_dir=f"seed{s}") for s in range(4)]))
    runner = Path(__file__).resolve().parents[1] / "scripts/validate_phase1.py"
    return subprocess.run([sys.executable, str(runner), "--output", str(root)],
                          env={**os.environ, "AGENT_JOB_GPU": "1"}, capture_output=True, text=True, timeout=60)


def test_retry_of_designed_early_stop_exits_zero(tmp_path):
    # A gate stop is a designed terminal state: the report is the deliverable, not a job defect.
    result = retry_campaign(tmp_path, "early_stop")
    assert result.returncode == 0, result.stderr


def test_retry_of_failed_campaign_still_exits_nonzero(tmp_path):
    result = retry_campaign(tmp_path, "failure")
    assert result.returncode != 0
    assert "No automatic restart" in result.stderr


def test_two_seed_diagnostic_uses_declared_budget(tmp_path):
    campaign(tmp_path, [50000, 60000])
    state = json.loads((tmp_path / "state.json").read_text())
    state.update(seeds=[0, 1], runs=state["runs"][:2], frames_per_seed=100_000_000)
    monitor.atomic_json(tmp_path / "state.json", state)
    report = monitor.inspect(tmp_path)
    assert not report["problems"]
    assert all(r["remaining_hours"] == 0 for r in report["runs"])


def test_diagnostic_dry_run_records_seed_and_frame_budget():
    runner = Path(__file__).resolve().parents[1] / "scripts/validate_phase1.py"
    result = subprocess.run([sys.executable, str(runner), "--dry-run", "--seeds", "4", "5",
                             "--frames-per-seed", "100000000"], capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["seeds"] == [4, 5] and plan["frames_per_seed"] == 100_000_000


def test_bad_learning_does_not_mask_a_numerical_failure(tmp_path):
    campaign(tmp_path, [10000, 15000])
    log = tmp_path / "seed0" / "log.jsonl"
    rec = json.loads(log.read_text()); rec["global_U"] = float("inf")
    log.write_text(json.dumps(rec) + "\n")
    report = monitor.inspect(tmp_path)
    assert report["early_stop"] is False
    assert any("non-finite global_U" in p for p in report["problems"])
