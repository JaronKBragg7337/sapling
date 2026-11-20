"""Resume the growth release with a compiled full-micro-batch speed test."""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import runpod_growth as G
import runpod_run as R


def main():
    stamp = "20261028-224312"
    tag = "growth-" + stamp
    base = f"https://github.com/{R.REPO}/releases/download/{tag}"
    prior_cost = 0.411
    run = R.Run(stamp, False)
    public_path = os.path.join(run.dir, "bundle.json")
    public = json.load(open(public_path, encoding="utf-8"))
    quote = R.price_quote()
    secure = float(quote.get("securePrice") or 3.99)
    deadline = G.deadline_tonight()
    run.log(step="resume", release=tag, prior_cost=prior_cost, deadline=deadline.isoformat(), benchmark="micro 64 + torch.compile")
    if R.list_pods():
        raise RuntimeError("refusing to resume while any pod exists")
    G.upload(run, tag, os.path.join(ROOT, "tools", "cloud_growth.py"))
    have = G.assets(tag)

    def missing_uploads():
        for name, size in public["files"].items():
            if name in public.get("urls", {}) or have.get(name) == size:
                continue
            if name in (public["core_file"], "reading_index.json"):
                path = os.path.join(run.dir, name)
            else:
                path = os.path.join(ROOT, "data", "reading", "shards", name)
            G.upload(run, tag, path)

    costs = [prior_cost]
    growth_ready = False
    st = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pod = None
        t_pod = None
        try:
            pod = G.create_pod(run, stamp, "bench64", base, 15, 0, 45, bench_micro=64, use_compile=True)
            t_pod = time.time()
            G.set_reaper(25)
            G.cloud_log(f"START | {pod['id']} | H100 SXM ({pod['cloud'].lower()}) | growth compiled speed test {stamp}; "
                        f"watchdog 15 min; reaper task | ${pod['cost_per_hr']:.2f}/hr")
            future = pool.submit(missing_uploads)
            st = R.watch(run, pod, t_pod + 13 * 60)
            if st.get("state") in ("done", "failed"):
                R.fetch(run, pod, names=("benchmark.json", "status.json", "run.txt"))
        finally:
            if pod:
                costs.append(G.terminate(run, pod, t_pod, f"growth compiled speed test {stamp}"))
        bench_path = os.path.join(run.dir, "benchmark.json")
        if not os.path.exists(bench_path) and st.get("benchmark"):
            json.dump(st["benchmark"], open(bench_path, "w"), indent=1)
        if st.get("state") != "done" or not os.path.exists(bench_path):
            raise RuntimeError(f"compiled speed test failed: {st}")
        bench = json.load(open(bench_path, encoding="utf-8"))
        run.log(step="speed measured compiled", **bench)
        future.result()
        have = G.assets(tag)
        for name, size in public["files"].items():
            if name not in public.get("urls", {}) and have.get(name) != size:
                raise RuntimeError(f"growth asset not verified after resume: {name}")
        rate = float(bench["tok_per_s"])
        target = int(bench["target_tokens"])
        train_h = target / rate / 3600
        estimated_growth = secure * (train_h + 20 / 60)
        estimated_total = sum(costs) + estimated_growth
        now = dt.datetime.now()
        max_by_cost = (G.WORKING_CAP - sum(costs)) / secure * 60
        max_by_deadline = (deadline - now).total_seconds() / 60 - 25
        max_minutes = min(max_by_cost, max_by_deadline)
        estimated_finish = now + dt.timedelta(minutes=train_h * 60 + 20)
        latest_finish = deadline - dt.timedelta(minutes=25)
        plan = {"measured_tok_per_s": rate, "params": bench["params"], "target_tokens": target,
                "training_hours": round(train_h, 3), "growth_cost_worst_case": round(estimated_growth, 2),
                "total_with_speed_tests_worst_case": round(estimated_total, 2),
                "estimated_finish": estimated_finish.isoformat(), "latest_finish_for_local_folds": latest_finish.isoformat(),
                "max_pod_minutes": round(max_minutes, 1)}
        run.log(step="growth dollar plan compiled", **plan)
        G.cloud_log(f"PLAN | - | H100 SXM | growth {stamp} compiled measured {rate:,} tok/s; target {target:,} tokens | "
                    f"growth ${estimated_growth:.2f} worst case, all pods ${estimated_total:.2f}; local-fold reserve 25 min")
        if estimated_total >= G.WORKING_CAP or estimated_finish > latest_finish or max_minutes < train_h * 60 + 20:
            raise RuntimeError(f"compiled plan does not fit cost/time limits: {plan}")
        train_minutes = max_minutes - 20
        pod = None
        t_pod = None
        try:
            pod = G.create_pod(run, stamp, "train", base, max_minutes, train_minutes, 45, use_compile=True)
            t_pod = time.time()
            G.set_reaper(max_minutes + 10)
            G.cloud_log(f"START | {pod['id']} | H100 SXM ({pod['cloud'].lower()}) | growth training {stamp}; "
                        f"target {target:,} tokens; watchdog {max_minutes:.1f} min; reaper task | ${pod['cost_per_hr']:.2f}/hr")
            st = R.watch(run, pod, t_pod + max_minutes * 60 - 120)
            if st.get("state") in ("done", "failed"):
                R.fetch(run, pod, names=("core_staged.pt", "log.jsonl", "status.json", "run.txt"))
        finally:
            if pod:
                costs.append(G.terminate(run, pod, t_pod, f"growth training {stamp}"))
        growth_ready = bool(st.get("state") == "done" and (st.get("run") or {}).get("target_reached") and
                            os.path.exists(os.path.join(run.dir, "core_staged.pt")))
        run.log(step="growth result", ready=growth_ready, status=st, total_cost=round(sum(costs), 2))
    left = [p.get("name") for p in R.list_pods()]
    if any(str(n).startswith(R.NAME_PREFIX) for n in left):
        G.reap()
        left = [p.get("name") for p in R.list_pods()]
    summary = {"stamp": stamp, "release": tag, "speed_test_compiled": bench, "growth_ready": growth_ready,
               "cost_all_pods": round(sum(costs), 2), "pods_left": left, "run_dir": os.path.relpath(run.dir, ROOT)}
    json.dump(summary, open(os.path.join(run.dir, "summary_compiled.json"), "w"), indent=1)
    G.cloud_log(f"FINISH | - | - | growth {stamp}; core ready={growth_ready} | all pods ${sum(costs):.2f}; pods left: {left or 'none'}")
    run.log(step="finished compiled", **summary)
    if not growth_ready:
        raise RuntimeError(f"growth did not produce a complete core: {st}")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            G.reap()
            left = [p.get("name") for p in R.list_pods()]
            G.cloud_log(f"FINISH | - | - | compiled growth run stopped on error | pods left: {left or 'none'}")
        finally:
            raise
