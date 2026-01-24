import numpy as np
import pandas as pd
import json
from argparse import ArgumentParser

def normalize_levels(df):
    df = df.copy()
    df["noise_level"] = df["noise_level"].astype(str).str.strip()
    df["streak_level"] = df["streak_level"].astype(str).str.strip()
    return df

def sample_balanced(df, n_total, rng):
    groups = list(df.groupby(["noise_level", "streak_level"]))
    n_combos = len(groups)

    if n_combos == 0:
        return []

    if n_total % n_combos != 0:
        raise ValueError(f"n_total={n_total} non divisibile per n_combos={n_combos}")

    per_combo = n_total // n_combos
    sampled = []

    for (_, _), g in groups:
        idxs = g.index.to_numpy()
        if len(idxs) < per_combo:
            raise ValueError(
                "Impossibile campionare in modo perfettamente bilanciato: "
                "una combinazione ha meno immagini disponibili."
            )
        chosen = rng.choice(idxs, size=per_combo, replace=False)
        sampled.extend([int(i) for i in chosen])

    rng.shuffle(sampled)
    return sampled

def iterative_sampling_dict(
    map_csv_path,
    first_blocks=(12, 48, 240, 612),
    seed=None
):
    # se seed è None → random 
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    df_all = pd.read_csv(map_csv_path, index_col=0)
    df_all = normalize_levels(df_all)

    used = set()
    result = {}

    def clean_df():
        if not used:
            return df_all
        return df_all.drop(index=[i for i in used if i in df_all.index])

    for n in first_blocks:
        df_clean = clean_df()
        indices = sample_balanced(df_clean, n, rng)
        result[n] = indices
        used.update(indices)

        if n == first_blocks[-1]:
            break

    return result

def save_sampling_dict_json(sampling_dict, out_path):
    to_save = {str(k): v for k, v in sampling_dict.items()}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":

    parser = ArgumentParser(description="Estrazione indici bilanciati LDCTIQA")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed per il random (se omesso, campionamento completamente casuale)"
    )
    args = parser.parse_args()

    train_map = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAG2023_train/train_map.csv"

    sampling_dict = iterative_sampling_dict(
        map_csv_path=train_map,
        first_blocks=(12, 48, 240, 612),
        seed=args.seed
    )

    # Nome file che riflette il seed usato
    seed_tag = f"seed{args.seed}" if args.seed is not None else "noseed"
    out_json = f"sampling_dict_12_48_240_612_{seed_tag}.json"

    save_sampling_dict_json(sampling_dict, out_json)

    print(f"Dizionario di campionamento salvato in: {out_json}")
