import os
import glob
import numpy as np
import pandas as pd
import mne

STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Movement time": "UNK",
    "Sleep stage ?": "UNK",
}

def get_stage_for_time(hyp_df: pd.DataFrame, t_sec: float) -> str:
    hit = hyp_df[(hyp_df["start"] <= t_sec) & (hyp_df["end"] > t_sec)]
    if len(hit) == 0:
        return "UNK"
    return STAGE_MAP.get(hit["desc"].iloc[0], "UNK")

def convert_one_recording(psg_path: str, hyp_path: str,
                          epoch_sec: int = 30, target_sfreq: float = 100.0,
                          channels_keep=None, drop_unknown: bool = True) -> pd.DataFrame:
    if channels_keep is None:
        channels_keep = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"]

    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)

    available = [ch for ch in channels_keep if ch in raw.ch_names]
    if len(available) == 0:
        raise RuntimeError(f"No expected channels found in {os.path.basename(psg_path)}. "
                           f"Available: {raw.ch_names}")

    raw = raw.pick(available)

    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, npad="auto")

    sfreq = raw.info["sfreq"]
    data = raw.get_data()
    ch_names = raw.ch_names

    ann = mne.read_annotations(hyp_path)
    hyp_df = pd.DataFrame({"start": ann.onset, "dur": ann.duration, "desc": ann.description})
    hyp_df["end"] = hyp_df["start"] + hyp_df["dur"]

    total_sec = data.shape[1] / sfreq
    n_epochs = int(total_sec // epoch_sec)

    # recording id from PSG filename, e.g. SC4011E0
    recording_id = os.path.basename(psg_path).replace("-PSG.edf", "")

    rows = []
    for e in range(n_epochs):
        start = e * epoch_sec
        end = (e + 1) * epoch_sec
        mid = (start + end) / 2.0

        stage = get_stage_for_time(hyp_df, mid)
        if drop_unknown and stage == "UNK":
            continue

        s0 = int(start * sfreq)
        s1 = int(end * sfreq)
        epoch = data[:, s0:s1]

        feats = {}
        for i, ch in enumerate(ch_names):
            x = epoch[i]
            feats[f"{ch}_mean"] = float(np.mean(x))
            feats[f"{ch}_std"]  = float(np.std(x))
            feats[f"{ch}_rms"]  = float(np.sqrt(np.mean(x**2)))

        rows.append({
            "recording_id": recording_id,
            "epoch": e,
            "epoch_start_sec": start,
            "epoch_end_sec": end,
            "sleep_stage": stage,
            **feats
        })

    return pd.DataFrame(rows)

def find_matching_hypnogram(root_dir: str, psg_path: str) -> str | None:
    """
    PSG:        SC4011E0-PSG.edf
    Hypnogram:  SC4011EH-Hypnogram.edf  (last char differs)
    So we match by dropping the last char and wildcarding it:
    SC4011E*-Hypnogram.edf
    """
    base = os.path.basename(psg_path).replace("-PSG.edf", "")  # SC4011E0
    prefix = base[:-1]  # SC4011E
    pattern = os.path.join(root_dir, f"{prefix}*-Hypnogram.edf")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None

def build_sc_merged_table(root_dir: str,
                          out_csv: str = "sleepedf_SC_merged_epochs.csv",
                          epoch_sec: int = 30,
                          target_sfreq: float = 100.0,
                          drop_unknown: bool = True) -> pd.DataFrame:

    root_dir = os.path.expanduser(root_dir)

    psg_files = sorted(glob.glob(os.path.join(root_dir, "SC*-PSG.edf")))
    if len(psg_files) == 0:
        raise RuntimeError(f"No SC*-PSG.edf files found in: {root_dir}")

    all_dfs = []
    for psg_path in psg_files:
        hyp_path = find_matching_hypnogram(root_dir, psg_path)
        if hyp_path is None:
            print(f"SKIP (no hypnogram): {os.path.basename(psg_path)}")
            continue

        try:
            df_one = convert_one_recording(
                psg_path=psg_path,
                hyp_path=hyp_path,
                epoch_sec=epoch_sec,
                target_sfreq=target_sfreq,
                drop_unknown=drop_unknown
            )
            df_one["psg_file"] = os.path.basename(psg_path)
            df_one["hyp_file"] = os.path.basename(hyp_path)
            all_dfs.append(df_one)
            print(f"OK: {os.path.basename(psg_path)} -> epochs: {len(df_one)}")
        except Exception as ex:
            print(f"FAILED: {os.path.basename(psg_path)} | {ex}")

    if not all_dfs:
        raise RuntimeError("No recordings were converted successfully.")

    merged = pd.concat(all_dfs, ignore_index=True)
    merged.to_csv(out_csv, index=False)
    print(f"\nSaved merged table: {out_csv}")
    print("Rows:", len(merged), "| Columns:", len(merged.columns))
    return merged

if __name__ == "__main__":
    root_dir = "/Volumes/Neha_HD/sleep-edf/sleep-cassette"
    build_sc_merged_table(
        root_dir=root_dir,
        out_csv="sleepedf_SC_merged_epochs.csv",
        epoch_sec=30,
        target_sfreq=100.0,
        drop_unknown=True
    )
