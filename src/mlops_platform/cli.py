"""``mlops`` command line: train, promote, rollback, drift, serve."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

import pandas as pd

from .data import CATEGORICAL, NUMERIC, load_credit, split_eval, synthetic_credit


def _data(args: argparse.Namespace) -> pd.DataFrame:
    if args.synthetic:
        return synthetic_credit(n=args.rows, seed=args.seed)
    return load_credit()


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")  # keep stdout machine readable
    parser = argparse.ArgumentParser(prog="mlops")
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "credit-risk-brl"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("train", "promote", "drift"):
        p = sub.add_parser(name)
        p.add_argument("--synthetic", action="store_true", help="use generated data instead of OpenML credit-g")
        p.add_argument("--rows", type=int, default=1000)
        p.add_argument("--seed", type=int, default=7)
    sub.choices["train"].add_argument("--experiment", default="credit-risk")
    sub.choices["promote"].add_argument("--version", required=True)
    sub.choices["promote"].add_argument("--dry-run", action="store_true")
    sub.choices["drift"].add_argument("--shift", type=float, default=0.0, help="simulate drift on the current data")
    sub.add_parser("rollback")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))

    args = parser.parse_args(argv)

    if args.cmd == "train":
        from .train import train

        pool, _ = split_eval(_data(args))
        result = train(pool, experiment=args.experiment, model_name=args.model_name)
        print(json.dumps(asdict(result), indent=2))
    elif args.cmd == "promote":
        from .promote import evaluate_and_promote

        _, evaluation = split_eval(_data(args))
        decision = evaluate_and_promote(args.model_name, args.version, evaluation, dry_run=args.dry_run)
        print(json.dumps(asdict(decision), indent=2))
        return 0 if decision.promoted else 2
    elif args.cmd == "rollback":
        from .promote import rollback

        champion, demoted = rollback(args.model_name)
        print(f"champion -> v{champion} (v{demoted} demoted)")
    elif args.cmd == "drift":
        from .drift import drift_report

        reference, current = split_eval(_data(args))
        if args.shift:
            current = synthetic_credit(n=len(current), seed=args.seed + 1, shift=args.shift)
        report = drift_report(reference, current, NUMERIC, CATEGORICAL)
        print(json.dumps([asdict(r) for r in report], indent=2))
        return 1 if any(r.severity == "major" for r in report) else 0
    elif args.cmd == "serve":
        import uvicorn

        from .serve import create_app

        uvicorn.run(create_app(), host="0.0.0.0", port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
