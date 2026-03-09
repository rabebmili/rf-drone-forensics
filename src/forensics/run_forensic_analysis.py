"""
Run forensic analysis on a signal file: classify segments, detect anomalies,
generate timeline and report.

Usage:
    python -m src.forensics.run_forensic_analysis --file "data/raw/DroneRF/AR drone/10100H_0.csv"
    python -m src.forensics.run_forensic_analysis --file "data/raw/DroneRF/AR drone/10100H_0.csv" --model resnet --task multiclass
"""

import argparse
from pathlib import Path

import torch

from src.models.cnn_spectrogram import SmallRFNet
from src.models.resnet_spectrogram import RFResNet
from src.models.transformer_spectrogram import RFTransformer
from src.forensics.timeline import (
    analyze_signal_file,
    generate_forensic_report,
    plot_forensic_timeline,
)


MODEL_REGISTRY = {
    "smallrf": SmallRFNet,
    "resnet": RFResNet,
    "transformer": RFTransformer,
}


def main():
    parser = argparse.ArgumentParser(description="Forensic analysis of RF signal file")
    parser.add_argument("--file", required=True, help="Path to raw signal CSV file")
    parser.add_argument("--model", default="smallrf", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--task", default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--weights", default=None, help="Path to model weights (auto-detected if not set)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--anomaly_threshold", type=float, default=0.7)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 2 if args.task == "binary" else 4
    class_names = (
        ["Background", "Drone"] if args.task == "binary"
        else ["Background", "AR Drone", "Bepop Drone", "Phantom Drone"]
    )

    # Load model
    if args.weights is None:
        args.weights = f"outputs/{args.model}_{args.task}/models/best_model.pt"

    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.weights, weights_only=True, map_location=device))
    model.eval()
    print(f"Loaded model: {args.model} ({args.task}) from {args.weights}")

    # Output directory
    if args.output_dir is None:
        file_stem = Path(args.file).stem
        args.output_dir = f"outputs/forensic_reports/{file_stem}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Analyze
    print(f"\nAnalyzing: {args.file}")
    timeline = analyze_signal_file(
        model, args.file, device,
        class_names=class_names,
        anomaly_threshold=args.anomaly_threshold
    )

    print(f"Processed {len(timeline)} segments")

    # Generate report
    report = generate_forensic_report(
        timeline, args.file,
        output_path=str(out_dir / "forensic_report.json"),
        class_names=class_names
    )

    # Plot timeline
    plot_forensic_timeline(
        timeline,
        output_path=str(out_dir / "forensic_timeline.png"),
        title=f"Forensic Timeline — {Path(args.file).stem}"
    )

    # Print summary
    summary = report["summary"]
    print(f"\n{'='*50}")
    print(f"  FORENSIC SUMMARY")
    print(f"{'='*50}")
    print(f"  Total segments:    {report['report_metadata']['total_segments']}")
    print(f"  Drone segments:    {summary['drone_segments_count']}")
    print(f"  Anomalous:         {summary['anomalous_segments_count']}")
    print(f"  Avg confidence:    {summary['average_confidence']:.4f}")
    print(f"  Class distribution:")
    for cls, count in summary["class_distribution"].items():
        print(f"    {cls}: {count}")
    if summary["anomalous_segment_ids"]:
        print(f"  Suspicious segments: {summary['anomalous_segment_ids']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
