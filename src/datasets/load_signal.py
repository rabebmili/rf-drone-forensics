import numpy as np
from pathlib import Path


def load_dronerf_csv(file_path, dtype=np.float32):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    if file_path.stat().st_size == 0:
        raise ValueError(f"Fichier vide : {file_path}")

    # Lecture brute du fichier texte
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"Aucune donnée trouvée dans : {file_path}")

    # Remplacer les retours ligne par des virgules pour uniformiser
    content = content.replace("\n", ",").replace("\r", ",")

    # Conversion rapide en tableau NumPy
    signal = np.fromstring(content, sep=",", dtype=dtype)

    if signal.size == 0:
        raise ValueError(
            f"Impossible de lire des valeurs numériques depuis : {file_path}"
        )

    return signal