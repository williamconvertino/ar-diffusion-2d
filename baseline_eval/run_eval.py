#!/usr/bin/env python3

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import EVAL_CONFIG, GPUS, MODELS, NUM_FEWSHOT_LIST, TASKS


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _make_logger(prefix: str = "") -> logging.Logger:
    fmt = f"%(asctime)s | {prefix}%(levelname)s | %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    name   = prefix.strip(" []|") or "main"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers  = [handler]
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Job definition
# ---------------------------------------------------------------------------

@dataclass
class Job:
    model_cfg:   dict
    num_fewshot: int
    tasks:       list
    cfg:         dict

    @property
    def label(self) -> str:
        return f"{self.model_cfg['name']} | {self.num_fewshot}-shot"


def result_path(output_dir: str, model_name: str, num_fewshot: int) -> Path:
    safe_name = model_name.replace(" ", "_").replace("/", "-")
    return Path(output_dir) / safe_name / f"{num_fewshot}shot" / "results.json"


# ---------------------------------------------------------------------------
# Worker process — owns exactly one GPU, runs jobs one at a time
# ---------------------------------------------------------------------------

def gpu_worker(gpu_id, job_queue, done_queue):
    """
    Pulls Job objects from job_queue sequentially.
    Sends (label, success, error_msg) to done_queue for each job.
    Exits on sentinel None.
    """
    device_label = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"
    prefix       = f"[GPU {gpu_id}] " if gpu_id is not None else "[CPU] "
    log          = _make_logger(prefix)

    # Pin this subprocess to a single physical GPU via CUDA_VISIBLE_DEVICES.
    # Inside the subprocess the device is always visible as cuda:0.
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device_str = "cuda:0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device_str = "cpu"

    # Import after env var is set so PyTorch picks up the right device.
    from lm_eval import evaluator  # noqa: PLC0415

    while True:
        job = job_queue.get()
        if job is None:
            log.info("No more jobs. Exiting.")
            break

        log.info("Starting: %s", job.label)
        out_path = result_path(job.cfg["output_dir"], job.model_cfg["name"], job.num_fewshot)

        if out_path.exists():
            log.info("Already done, skipping: %s", job.label)
            done_queue.put((job.label, True, None))
            continue

        try:
            model_kwargs = {
                "pretrained": job.model_cfg["model_id"],
                "dtype":      job.model_cfg.get("dtype", "bfloat16"),
                "device":     device_str,
                **job.model_cfg.get("extra_args", {}),
            }

            results = evaluator.simple_evaluate(
                model=job.model_cfg["model_type"],
                model_args=model_kwargs,
                tasks=job.tasks,
                num_fewshot=job.num_fewshot,
                batch_size=job.cfg["batch_size"],
                limit=job.cfg["limit"],
                log_samples=job.cfg["log_samples"],
            )

            results["_meta"] = {
                "model_name":  job.model_cfg["name"],
                "model_id":    job.model_cfg["model_id"],
                "num_fewshot": job.num_fewshot,
                "tasks":       job.tasks,
                "device":      device_label,
                "timestamp":   datetime.utcnow().isoformat() + "Z",
                "eval_config": {k: v for k, v in job.cfg.items() if k != "output_dir"},
            }

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as fh:
                json.dump(results, fh, indent=2, default=str)

            log.info("Saved → %s", out_path)
            done_queue.put((job.label, True, None))

        except Exception as exc:  # noqa: BLE001
            log.error("FAILED: %s\n%s", job.label, traceback.format_exc())
            done_queue.put((job.label, False, str(exc)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Multi-GPU LM Eval runner")
    p.add_argument("--models",    nargs="+", default=None, metavar="NAME")
    p.add_argument("--tasks",     nargs="+", default=None, metavar="TASK")
    p.add_argument("--fewshots",  nargs="+", type=int, default=None, metavar="N",
                   help="Few-shot counts (overrides NUM_FEWSHOT_LIST in config).")
    p.add_argument("--gpus",      nargs="+", type=int, default=None, metavar="ID",
                   help="GPU indices to use. Pass -1 for CPU-only.")
    p.add_argument("--batch_size", default=None)
    p.add_argument("--limit",      type=int, default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--log_samples", action="store_true")
    return p.parse_args()


def resolve_config(args):
    cfg = dict(EVAL_CONFIG)
    if args.batch_size is not None:
        try:
            cfg["batch_size"] = int(args.batch_size)
        except ValueError:
            cfg["batch_size"] = args.batch_size
    if args.limit       is not None: cfg["limit"]       = args.limit
    if args.output_dir  is not None: cfg["output_dir"]  = args.output_dir
    if args.log_samples:             cfg["log_samples"]  = True
    return cfg


def select_models(requested):
    if requested is None:
        return MODELS
    name_map = {m["name"]: m for m in MODELS}
    out = []
    for name in requested:
        if name not in name_map:
            raise ValueError(f"Unknown model '{name}'. Available: {list(name_map)}")
        out.append(name_map[name])
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 'spawn' gives each subprocess a clean CUDA context (critical on Linux).
    mp.set_start_method("spawn", force=True)

    args   = parse_args()
    cfg    = resolve_config(args)
    models = select_models(args.models)
    tasks  = args.tasks    if args.tasks    is not None else TASKS
    shots  = args.fewshots if args.fewshots is not None else NUM_FEWSHOT_LIST
    gpus   = args.gpus     if args.gpus     is not None else GPUS

    if gpus == [-1]:
        gpus = []   # CPU-only

    log = _make_logger()

    # ── Build job list ────────────────────────────────────────────────────────
    # Order: all shot settings for model[0], then model[1], …
    # This way a GPU assigned to model[0] works through its shots before
    # moving on, keeping the same model weights warm in memory longer.
    all_jobs = [
        Job(model_cfg=m, num_fewshot=n, tasks=tasks, cfg=cfg)
        for m in models
        for n in shots
    ]

    total      = len(all_jobs)
    worker_ids = gpus if gpus else [None]   # None → CPU worker
    n_workers  = len(worker_ids)

    log.info("━" * 60)
    log.info("Jobs      : %d  (%d model(s) × %d shot setting(s))",
             total, len(models), len(shots))
    log.info("Workers   : %s", [f"cuda:{g}" if g is not None else "cpu"
                                  for g in worker_ids])
    log.info("Tasks     : %s", tasks)
    log.info("Shot list : %s", shots)
    log.info("Output    : %s", cfg["output_dir"])
    log.info("━" * 60)

    # ── Set up queues and workers ─────────────────────────────────────────────
    job_queue  = mp.Queue()
    done_queue = mp.Queue()

    for job in all_jobs:
        job_queue.put(job)
    for _ in worker_ids:          # one sentinel per worker
        job_queue.put(None)

    processes = []
    for gpu_id in worker_ids:
        p = mp.Process(target=gpu_worker,
                       args=(gpu_id, job_queue, done_queue),
                       daemon=True)
        p.start()
        processes.append(p)
        label = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"
        log.info("Started worker %-10s  PID %d", label, p.pid)

    log.info("")

    # ── Collect completion notifications ──────────────────────────────────────
    finished, failed = 0, []
    while finished < total:
        label, success, err = done_queue.get()
        finished += 1
        status = "✓" if success else "✗"
        log.info("[%d/%d] %s  %s", finished, total, status, label)
        if not success:
            failed.append((label, err))

    for p in processes:
        p.join()

    log.info("")
    if failed:
        log.warning("%d job(s) failed:", len(failed))
        for label, err in failed:
            log.warning("  ✗ %s\n    %s", label, err)
    else:
        log.info("All %d jobs completed successfully.", total)
        log.info("Run  python parse_results.py  to generate the LaTeX table.")


if __name__ == "__main__":
    main()
