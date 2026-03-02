import json
import os

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset, random_split

DATASET_DIR = "dataset/"
TEST_PROPORTION = 0.1

json_correspondance = {
    "M994_PAG": {
        "json_file": "M994_20191013_UMaze_SpikeRef.json",
        "prefix": "M994_PAG",
    },
    "M1162_MFB": {
        "json_file": "M1162_20200119_StimMFBWake.json",
        "prefix": "M1162_MFB",
    },
    "M1162_PAG": {
        "json_file": "M1162_20210121_UMaze.json",
        "prefix": "M1162_PAG",
    },
    "M1168_MFB": {
        "json_file": "M1168_20210121_StimMFBWake.json",
        "prefix": "M1168_MFB",
    },
    "M1182_PAG": {
        "json_file": "M1182_20210301_UMaze.json",
        "prefix": "M1182_PAG",
    },
    "M1199_PAG": {
        "json_file": "M1199_PAG.json",
        "prefix": "M1199_PAG",
    },
    "M1239_PAG": {
        "json_file": "M1239_20211110_UMazePAG.json",
        "prefix": "M1239_PAG",
    },
}


class SingleStrideWindowDataset(Dataset):
    def __init__(
        self,
        mouse_id,
        stride,
        window_size,
        use_speedMask=True,
        is_train=True,
        mid_split=False,
    ):
        self.mouse_id = mouse_id
        self.stride = stride
        self.window_size = window_size
        self.mid_split = mid_split

        self.json_file = os.path.join(
            DATASET_DIR, json_correspondance[mouse_id]["json_file"]
        )
        self.prefix = json_correspondance[mouse_id]["prefix"]
        self.parquet_file = os.path.join(
            DATASET_DIR,
            f"{self.prefix}_stride{self.stride}_win{self.window_size}_test.parquet",
        )

        with open(self.json_file, "r") as f:
            self.params = json.load(f)

        self.parquet_df = pd.read_parquet(self.parquet_file)

        self.nGroups = self.params["nGroups"]
        self.nChannelsPerGroup = [
            self.params[f"group{g}"]["nChannels"] for g in range(self.nGroups)
        ]

        if use_speedMask:
            mask = self.parquet_df["speedMask"].apply(
                lambda x: x[0] == 1 if len(x) > 0 else False
            )
            self.parquet_df = self.parquet_df[mask].reset_index(drop=True)

        if mid_split:
            test_mask = np.zeros(len(self.parquet_df), dtype=bool)
            mid_idx = int(len(self.parquet_df) * 0.5)
            test_mask[
                mid_idx : mid_idx + int(len(self.parquet_df) * TEST_PROPORTION)
            ] = True
            if is_train:
                self.parquet_df = self.parquet_df[~test_mask]
            else:
                self.parquet_df = self.parquet_df[test_mask]

        else:
            if is_train:
                self.parquet_df = self.parquet_df[
                    : int(len(self.parquet_df) * (1 - TEST_PROPORTION))
                ]
            else:
                self.parquet_df = self.parquet_df[
                    int(len(self.parquet_df) * (1 - TEST_PROPORTION)) :
                ]

    def __len__(self):
        return len(self.parquet_df)

    def __getitem__(self, idx):
        sample = self.parquet_df.iloc[idx]  # type: ignore

        # Target: (x, y) position
        pos = torch.tensor(sample["pos"][:2], dtype=torch.float32)

        # Get all spikes concatenated with group info
        groups = np.array(sample["groups"])

        length_val = sample["length"]
        if isinstance(length_val, (list, np.ndarray)):
            length = int(length_val[0])
        else:
            length = int(length_val)

        # Collect all spike waveforms
        all_spikes = []
        for g in range(self.nGroups):
            spikes_flat = np.array(sample[f"group{g}"])
            if len(spikes_flat) > 0:
                spikes = spikes_flat.reshape(-1, self.nChannelsPerGroup[g], 32)
                all_spikes.append(spikes)

        return {
            "groups": torch.tensor(groups[:length], dtype=torch.long),
            "pos": pos,
            "length": length,
            "spikes": all_spikes,  # list of arrays per group
        }


def gather_spikes(group: torch.Tensor, all_spikes: list[torch.Tensor]) -> torch.Tensor:
    """
    group: (length,) ints in [0, c-1]
    all_spikes[j]: (n_j, 5, 32), sum_j n_j == k
    returns: (k, 5, 32)
    """
    c = len(all_spikes)
    length = group.shape[0]
    # output buffer
    out = torch.zeros(
        (length, *all_spikes[0].shape[1:]),
        dtype=all_spikes[0].dtype,
        device=group.device,
    )

    for j in range(c):
        mask = group[:length] == j  # positions in output belonging to class j
        n = int(mask.sum().item())
        if n == 0:
            continue

        # Fill those positions with rows 0..n-1 from all_spikes[j]
        out[mask] = all_spikes[j][:n]

    return out


class SingleStrideWindowPaddedDataset(Dataset):
    def __init__(
        self,
        mouse_id,
        stride,
        window_size,
        use_speedMask=True,
        is_train=True,
        full_data=False,
        mid_split=False,
    ):
        self.mouse_id = mouse_id
        self.stride = stride
        self.window_size = window_size
        self.mid_split = mid_split

        self.json_file = os.path.join(
            DATASET_DIR, json_correspondance[mouse_id]["json_file"]
        )
        self.prefix = json_correspondance[mouse_id]["prefix"]
        self.parquet_file = os.path.join(
            DATASET_DIR,
            f"{self.prefix}_stride{self.stride}_win{self.window_size}_test.parquet",
        )

        with open(self.json_file, "r") as f:
            self.params = json.load(f)

        self.parquet_df = pd.read_parquet(self.parquet_file)

        self.nGroups = self.params["nGroups"]
        self.nChannelsPerGroup = [
            self.params[f"group{g}"]["nChannels"] for g in range(self.nGroups)
        ]

        if use_speedMask:
            mask = self.parquet_df["speedMask"].apply(
                lambda x: x[0] == 1 if len(x) > 0 else False
            )
            self.parquet_df = self.parquet_df[mask].reset_index(drop=True)

        if not full_data:
            if mid_split:
                test_mask = np.zeros(len(self.parquet_df), dtype=bool)
                mid_idx = int(len(self.parquet_df) * 0.5)
                test_mask[
                    mid_idx : mid_idx + int(len(self.parquet_df) * TEST_PROPORTION)
                ] = True
                if is_train:
                    self.parquet_df = self.parquet_df[~test_mask]
                else:
                    self.parquet_df = self.parquet_df[test_mask]
            else:
                if is_train:
                    self.parquet_df = self.parquet_df[
                        : int(len(self.parquet_df) * (1 - TEST_PROPORTION))
                    ]
                else:
                    self.parquet_df = self.parquet_df[
                        int(len(self.parquet_df) * (1 - TEST_PROPORTION)) :
                    ]

        self.max_length = max(self.parquet_df["length"])[0]
        self.max_channels = max(self.nChannelsPerGroup)

    def __len__(self):
        return len(self.parquet_df)

    def __getitem__(self, idx):
        sample = self.parquet_df.iloc[idx]  # type: ignore

        # Target: (x, y) position
        pos = torch.tensor(sample["pos"][:2], dtype=torch.float32)

        # Get all spikes concatenated with group info
        groups = np.array(sample["groups"])

        length_val = sample["length"]
        if isinstance(length_val, (list, np.ndarray)):
            length = int(length_val[0])
        else:
            length = int(length_val)

        # Collect all spike waveforms
        all_spikes = []

        for g in range(self.nGroups):
            spikes_flat = np.array(sample[f"group{g}"])
            spikes = torch.tensor(
                spikes_flat.reshape(-1, self.nChannelsPerGroup[g], 32)
            )

            spikes = pad(
                spikes,
                (0, 0, 0, self.max_channels - self.nChannelsPerGroup[g]),
            )
            all_spikes.append(spikes)

        groups = torch.tensor(groups[:length], dtype=torch.long)

        groups = pad(groups, (0, self.max_length - length), value=self.nGroups)
        gathered_spikes = (gather_spikes(groups, all_spikes) + 120) / 150

        is_not_padding = torch.zeros_like(groups, dtype=torch.bool)
        is_not_padding[:length] = True

        return (
            groups,
            pos,
            length,
            gathered_spikes,
            is_not_padding,
        )


def get_dataloader(
    mouse_id, stride, window_size, batch_size=256, split_type="temporal", shuffle=True
):
    assert split_type in ["temporal", "mid", "shuffled"]
    if split_type == "temporal":
        train_dataset = SingleStrideWindowPaddedDataset(
            mouse_id, stride, window_size, is_train=True, mid_split=False
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        test_dataset = SingleStrideWindowPaddedDataset(
            mouse_id, stride, window_size, is_train=False, mid_split=False
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, train_dataset.max_channels

    elif split_type == "mid":
        train_dataset = SingleStrideWindowPaddedDataset(
            mouse_id, stride, window_size, is_train=True, mid_split=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = SingleStrideWindowPaddedDataset(
            mouse_id, stride, window_size, is_train=False, mid_split=True
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, train_dataset.max_channels

    elif split_type == "shuffled":
        full_dataset = SingleStrideWindowPaddedDataset(
            mouse_id, stride, window_size, is_train=True, full_data=True
        )
        train_dataset, test_dataset = random_split(
            full_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, full_dataset.max_channels
    else:
        raise ValueError(f"Invalid split type: {split_type}")

    return train_loader, test_loader


if __name__ == "__main__":
    dataset = SingleStrideWindowPaddedDataset("M1182_PAG", 4, 108)

    for i in range(len(dataset)):
        (
            groups,
            pos,
            length,
            gathered_spikes,
            is_not_padding,
        ) = dataset[i]
        print("groups shape: ", groups.shape)
        print("gathered_spikes shape: ", gathered_spikes.shape)
        print("pos: ", pos)
        print("length: ", length)
        print("is_not_padding: ", is_not_padding)

        # print("min(gathered_spikes): ", gathered_spikes.min())
        # print("max(gathered_spikes): ", gathered_spikes.max())
        # print("mean(gathered_spikes): ", gathered_spikes.mean())
        # print("quantile(gathered_spikes): ", gathered_spikes.quantile(0.5))
        # print("quantile(gathered_spikes): ", gathered_spikes.quantile(0.75))

        break
