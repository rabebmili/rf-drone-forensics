from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main(
    segments_csv="data/metadata/dronerf_segments.csv",
    output_csv="data/metadata/dronerf_segments_split.csv",
    test_size=0.15,
    val_size=0.15,
    random_state=42
):
    df = pd.read_csv(segments_csv)

    # On travaille sur la liste unique des fichiers
    file_df = df[["file_path", "label_binary", "label_multiclass", "label_class_name"]].drop_duplicates()

    # Split train+val / test au niveau fichier
    train_val_files, test_files = train_test_split(
        file_df,
        test_size=test_size,
        random_state=random_state,
        stratify=file_df["label_class_name"]
    )

    # Pour obtenir 15% val sur le total, on ajuste sur train_val
    val_ratio_adjusted = val_size / (1.0 - test_size)

    train_files, val_files = train_test_split(
        train_val_files,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=train_val_files["label_class_name"]
    )

    train_set = set(train_files["file_path"])
    val_set = set(val_files["file_path"])
    test_set = set(test_files["file_path"])

    def assign_split(file_path):
        if file_path in train_set:
            return "train"
        if file_path in val_set:
            return "val"
        if file_path in test_set:
            return "test"
        raise ValueError(f"Fichier non assigné: {file_path}")

    df["split"] = df["file_path"].apply(assign_split)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Fichier sauvegardé : {output_path}")
    print("\nRépartition des segments par split :")
    print(df["split"].value_counts())

    print("\nRépartition des fichiers par split :")
    files_with_split = df[["file_path", "split"]].drop_duplicates()
    print(files_with_split["split"].value_counts())

    print("\nRépartition des classes par split :")
    print(df.groupby(["split", "label_class_name"]).size())


if __name__ == "__main__":
    main()