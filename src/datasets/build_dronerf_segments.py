from pathlib import Path
import pandas as pd
from src.datasets.load_signal import load_dronerf_csv


WINDOW_SIZE = 131072
HOP_SIZE = 65536


def build_segment_index(
    metadata_csv="data/metadata/dronerf_metadata.csv",
    output_csv="data/metadata/dronerf_segments.csv",
    max_files=None
):
    df = pd.read_csv(metadata_csv)
    rows = []

    if max_files is not None:
        df = df.head(max_files)

    for i, row in df.iterrows():
        file_path = row["file_path"]
        print(f"[{i+1}/{len(df)}] Traitement : {file_path}")

        signal = load_dronerf_csv(file_path)
        n = len(signal)

        segment_id = 0
        for start in range(0, n - WINDOW_SIZE + 1, HOP_SIZE):
            end = start + WINDOW_SIZE

            rows.append({
                "file_path": file_path,
                "segment_id": segment_id,
                "start": start,
                "end": end,
                "label_binary": row["label_binary"],
                "label_multiclass": row["label_multiclass"],
                "label_class_name": row["label_class_name"],
                "activity_code": row["activity_code"],
                "sample_id": row["sample_id"],
                "dataset": row["dataset"]
            })

            segment_id += 1

    seg_df = pd.DataFrame(rows)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seg_df.to_csv(output_path, index=False)

    print(f"\nIndex des segments sauvegardé dans : {output_path}")
    print(f"Nombre total de segments : {len(seg_df)}")
    print("\nAperçu :")
    print(seg_df.head())


if __name__ == "__main__":
    build_segment_index(max_files=None)