"""SmallModel CLI.

Usage:
    smallmodel serve [--teacher KEY] [--port PORT]
    smallmodel compress --teacher KEY [--max-mb MB] [--max-params N] [--min-layers N]
    smallmodel create --teacher KEY --layers 0,3,6,11 [--name NAME]
    smallmodel distill --teacher KEY --student PATH [--epochs N]
    smallmodel evaluate --teacher KEY --student PATH
    smallmodel list-teachers
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="smallmodel",
        description="SmallModel - Compress large embedding models into small, fast students.",
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # serve
    p_serve = sub.add_parser("serve", help="Launch interactive web UI")
    p_serve.add_argument("--teacher", default="gte", help="Initial teacher model key")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=7860)
    p_serve.add_argument("--output-dir", default="output")

    # compress
    p_compress = sub.add_parser("compress", help="Auto-compress a teacher model")
    p_compress.add_argument("--teacher", required=True)
    p_compress.add_argument("--max-mb", type=float, default=50.0)
    p_compress.add_argument("--max-params", type=int, default=20_000_000)
    p_compress.add_argument("--min-layers", type=int, default=4)
    p_compress.add_argument("--min-vocab", type=int, default=None)
    p_compress.add_argument("--pca", action="store_true",
                            help="Use PCA-based hidden dim reduction")
    p_compress.add_argument("--output-dir", default="output")

    # create
    p_create = sub.add_parser("create", help="Create student with specific layers")
    p_create.add_argument("--teacher", required=True)
    p_create.add_argument("--layers", required=True, help="Comma-separated layer indices")
    p_create.add_argument("--name", default=None)
    p_create.add_argument("--no-prune", action="store_true")
    p_create.add_argument("--output-dir", default="output")

    # distill
    p_distill = sub.add_parser("distill", help="Run knowledge distillation")
    p_distill.add_argument("--teacher", required=True)
    p_distill.add_argument("--student", required=True, help="Path to student model")
    p_distill.add_argument("--epochs", type=int, default=10)
    p_distill.add_argument("--batch-size", type=int, default=32)
    p_distill.add_argument("--patience", type=int, default=3)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run MTEB evaluation")
    p_eval.add_argument("--teacher", required=True)
    p_eval.add_argument("--student", required=True)
    p_eval.add_argument("--include-teacher", action="store_true")
    p_eval.add_argument("--output-dir", default="output")

    # list-teachers
    sub.add_parser("list-teachers", help="List available teacher models")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "list-teachers":
        cmd_list_teachers()
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "compress":
        cmd_compress(args)
    elif args.command == "create":
        cmd_create(args)
    elif args.command == "distill":
        cmd_distill(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


def cmd_list_teachers():
    from smallmodel.teachers import TEACHERS
    from smallmodel.sizing import estimate_for_teacher

    print(f"\n{'Key':<15} {'Short Name':<25} {'Layers':>6} {'Hidden':>7} {'Vocab':>10} {'FP32 MB':>8}")
    print("-" * 75)
    for key, t in TEACHERS.items():
        layers = list(range(t["num_layers"]))
        est = estimate_for_teacher(key, layers)
        print(f"{key:<15} {t['short_name']:<25} {t['num_layers']:>6} "
              f"{t['hidden_dim']:>7} {t['vocab_size']:>10,} {est['fp32_mb']:>7.1f}")
    print()


def cmd_serve(args):
    from smallmodel import SmallModel
    sm = SmallModel.from_teacher(args.teacher, output_dir=args.output_dir)
    sm.serve(host=args.host, port=args.port)


def cmd_compress(args):
    from smallmodel import SmallModel
    sm = SmallModel.from_teacher(args.teacher, output_dir=args.output_dir)
    path = sm.compress(
        max_params=args.max_params,
        max_fp32_mb=args.max_mb,
        min_layers=args.min_layers,
        min_vocab=args.min_vocab,
        use_pca=args.pca,
    )
    print(f"\nDone: {path}")


def cmd_create(args):
    from smallmodel import SmallModel
    layer_indices = [int(x.strip()) for x in args.layers.split(",")]
    sm = SmallModel.from_teacher(
        args.teacher, layer_indices=layer_indices, output_dir=args.output_dir,
    )
    path = sm.create(name=args.name, no_prune=args.no_prune)
    print(f"\nDone: {path}")


def cmd_distill(args):
    from smallmodel.distill import distill
    from smallmodel.teachers import TEACHERS
    t = TEACHERS[args.teacher]
    path = distill(
        teacher_name=t["model_id"],
        student_path=args.student,
        epochs=args.epochs,
        batch_size=args.batch_size,
        trust_remote_code=t["trust_remote_code"],
        patience=args.patience,
    )
    print(f"\nDone: {path}")


def cmd_evaluate(args):
    from smallmodel import SmallModel
    sm = SmallModel.from_teacher(args.teacher, output_dir=args.output_dir)
    sm._student_path = args.student
    sm.evaluate(include_teacher=args.include_teacher)


if __name__ == "__main__":
    main()
