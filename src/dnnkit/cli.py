import argparse
from dnnkit.train import train
from dnnkit.leaderboard import save_leaderboard
from dnnkit.paper import generate_report
from dnnkit.registry import get_model


def main():
    parser = argparse.ArgumentParser(prog="dnnkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--dataset", type=str, default="mnist")
    train_parser.add_argument("--model", type=str, default="mlp")

    subparsers.add_parser("leaderboard", help="Build leaderboard from experiment runs")

    report_parser = subparsers.add_parser("report", help="Generate paper/report summary")
    report_parser.add_argument("--outputs-dir", type=str, default="outputs")
    report_parser.add_argument("--output-file", type=str, default="paper/generated_report.md")

    args = parser.parse_args()

    if args.command == "train":
        train(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            dataset_name=args.dataset,
            model_name=args.model,
        )
    elif args.command == "leaderboard":
        save_leaderboard()
    elif args.command == "report":
        generate_report(outputs_dir=args.outputs_dir, output_file=args.output_file)

    # Support for model argument
    model = get_model(args.model)
    print(f"Model {args.model} created.")


if __name__ == "__main__":
    main()
