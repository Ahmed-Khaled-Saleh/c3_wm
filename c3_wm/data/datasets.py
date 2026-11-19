__all__ = ['TILE_NAMES', 'EGO', 'MultiAgentPOVDataset', 'FindGoalPOVDataset', 'MultiAgentWorldModelDataset',
           'FindGoalWorldModelDataset', 'MultiAgentPlanningDataset']

import os
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from fastcore import *
from fastcore.utils import *

class MultiAgentPOVDataset(Dataset):

    def __init__(self, h5_path, split="train", val_ratio=0.1, transform=None):
        assert split in ["train", "val"], "split must be 'train' or 'val'"
        self.h5_path = h5_path
        self.split = split
        self.transform = transform

        self.file = None
        self.index_map = []

        with h5py.File(self.h5_path, "r") as f:
            all_episodes = sorted(list(f.keys()))
            num_episodes = len(all_episodes)

            val_count = int(num_episodes * val_ratio)
            train_count = num_episodes - val_count

            if self.split == "train":
                active_episodes = all_episodes[:train_count]
            else:
                active_episodes = all_episodes[train_count:]

            for episode_key in active_episodes:
                episode_group = f[episode_key]
                num_frames = np.min(episode_group['agents_success_at']).item()

                for frame_idx in range(num_frames):
                    self.index_map.append((episode_key, frame_idx, "0_pov"))
                    self.index_map.append((episode_key, frame_idx, "1_pov"))

        print(
            f"Initialized {self.split} split with {len(active_episodes)} episodes ({len(self.index_map)} total agent frames)."
        )

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

        episode_key, frame_idx, agent_pov_key = self.index_map[idx]
        frame = self.file[episode_key][agent_pov_key][frame_idx]

        if self.transform:
            frame = self.transform(frame)

        return frame

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


TILE_NAMES = ('unseen', 'empty', 'wall', 'floor', 'door', 'key', 'ball', 'box', 'goal', 'lava', 'partner', 'ego')
EGO = TILE_NAMES.index('ego')

class FindGoalPOVDataset(Dataset):

    def __init__(self,
                 h5_path,
                 split: str = "train",
                 val_layouts=(200, 260),
                 transform=None,
                ):
        assert split in ["train", "val"], "split must be 'train' or 'val'"
        self.h5_path, self.transform, self.file = h5_path, transform, None
        with h5py.File(h5_path, "r") as f:
            layout, done = f["layout"][:, 0], f["terminated"][:]
        val = (layout >= val_layouts[0]) & (layout < val_layouts[1])
        rows = val if split == "val" else ~val
        self.index = np.argwhere(rows[:, None] & ~done)
        print(f"Initialized {split} split: {len(np.unique(layout[rows]))} layouts, {len(self.index)} agent frames.")

    def __len__(self):
        return len(self.index)

    def tiles(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")
        n, a = self.index[idx]
        t = self.file["image"][n, a, :, :, 0].T.astype(np.int64)
        t[-1, t.shape[1] // 2] = EGO
        return t

    def __getitem__(self, idx):
        tiles = self.tiles(idx)
        n, a = self.index[idx]
        frame = self.file["pixels"][n, a]
        if self.transform:
            frame = self.transform(frame)
        return frame, torch.from_numpy(tiles)

    def tile_frequencies(self, samples: int = 2000, seed: int = 0):
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(self), min(samples, len(self)), replace=False)
        counts = np.bincount(np.concatenate([self.tiles(i).ravel() for i in picks]), minlength=len(TILE_NAMES))
        return counts / counts.sum()

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

class MultiAgentWorldModelDataset(Dataset):
    def __init__(
        self,
        h5_path,
        split="train",
        val_ratio=0.1,
        history_size=3,
        num_preds= 1,
        receiver_transform=None,
        sender_transform=None,
    ):
        assert split in ["train", "val"], "split must be 'train' or 'val'"
        self.h5_path = h5_path
        self.split = split
        self.receiver_transform = receiver_transform
        self.sender_transform = sender_transform
        self.history_size = history_size
        self.num_preds = num_preds
        self.file = None
        self.index_map = []

        with h5py.File(self.h5_path, "r") as f:
            all_episodes = sorted(list(f.keys()))
            num_episodes = len(all_episodes)
            val_count = int(num_episodes * val_ratio)
            train_count = num_episodes - val_count

            if self.split == "train":
                active_episodes = all_episodes[:train_count]
            else:
                active_episodes = all_episodes[train_count:]

            for episode_key in active_episodes:
                episode_group = f[episode_key]
                num_frames = np.min(
                    episode_group['agents_success_at']
                ).item()

                for start_idx in range(num_frames - history_size - num_preds):
                    self.index_map.append((episode_key, start_idx, "0", "1"))
                    self.index_map.append((episode_key, start_idx, "1", "0"))

        print(
            f"Initialized {self.split} split with {len(active_episodes)} episodes "
            f"({len(self.index_map)} total sequence pairs)."
        )

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

        episode_key, start_idx, sender_id, receiver_id = self.index_map[idx]
        episode = self.file[episode_key]
        end_idx = start_idx + self.history_size

        receiver_pov_seq = episode[f"{receiver_id}_pov"][start_idx : end_idx + self.num_preds]
        receiver_act_seq = episode[f"{receiver_id}_act"][start_idx : end_idx]


        sender_pov_seq = episode[f"{sender_id}_pov"][start_idx : end_idx]
        sender_csi_seq = episode[f"{sender_id}_csi"][start_idx : end_idx]

        if self.receiver_transform:
            receiver_pov_seq = torch.stack(
                [self.receiver_transform(receiver_pov_seq[t]) for t in range(self.history_size + 1)]
            )

        else:
            receiver_pov_seq = torch.from_numpy(receiver_pov_seq).float() / 255.0

        if self.sender_transform:
            sender_pov = torch.stack(
                [self.sender_transform(sender_pov_seq[t]) for t in range(self.history_size)]
            )

        else:
            sender_pov = torch.from_numpy(sender_pov_seq).float() / 255.0

        return {
            "pixels": receiver_pov_seq,
            "action": torch.tensor(
                receiver_act_seq, dtype=torch.long
            ),

            "sender_pov": sender_pov,
            "sender_csi": torch.tensor(
                sender_csi_seq, dtype=torch.complex64
            ),
        }

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


def _relative_pose(sender_pos, sender_dir, receiver_pos, receiver_dir):
    vec = np.array(((1, 0), (0, 1), (-1, 0), (0, -1)))
    ahead = vec[receiver_dir]
    right = np.stack([-ahead[..., 1], ahead[..., 0]], -1)
    d = sender_pos - receiver_pos
    return np.stack([(d * ahead).sum(-1), (d * right).sum(-1), (sender_dir - receiver_dir) % 4], -1)

class FindGoalWorldModelDataset(Dataset):

    def __init__(self,
                 h5_path,
                 split: str = "train",
                 val_layouts=(200, 260),
                 history_size: int = 3,
                 num_preds: int = 1,
                 tokens=None,
                 max_age: int = 8,
                 transform=None,
                ):
        assert split in ["train", "val"], "split must be 'train' or 'val'"
        self.h5_path, self.transform, self.file = h5_path, transform, None
        self.history_size, self.num_preds, self.max_age = history_size, num_preds, max_age
        self.tokens = np.load(tokens, mmap_mode="r") if isinstance(tokens, (str, os.PathLike)) else tokens
        with h5py.File(h5_path, "r") as f:
            layout, done, action = f["layout"][:, 0], f["terminated"][:], f["action"][:]
            self.position, self.direction = f["position"][:], f["direction"][:]
            offset, length = f["ep_offset"][:], f["ep_len"][:]
        self.done = done
        self.episode_start = np.repeat(offset, length)
        val = (layout >= val_layouts[0]) & (layout < val_layouts[1])
        keep = val if split == "val" else ~val
        T, span = history_size, history_size + num_preds
        playing = ~done & np.isfinite(action)
        index = []
        for o, n in zip(offset, length):
            if n < span or not keep[o]:
                continue
            ok = playing[o:o + n]
            run = np.cumsum(np.vstack([np.zeros((1, ok.shape[1]), int), ~ok]), axis=0)
            starts = np.arange(n - span + 1)
            good = (run[starts + T] - run[starts]) == 0
            index += [(o + s, a) for s, a in np.argwhere(good)]
        self.index = np.array(index, dtype=np.int64).reshape(-1, 2)
        self.agents = done.shape[1]
        print(f"Initialized {split} split: {len(np.unique(layout[keep]))} layouts, {len(self.index)} windows.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")
        r, i = self.index[idx]
        T, span = self.history_size, self.history_size + self.num_preds
        pixels = self.file["pixels"][r:r + span, i]
        action = self.file["action"][r:r + T, i]
        frames = [self.transform(p) if self.transform else torch.from_numpy(p).permute(2, 0, 1).float() / 255.0
                  for p in pixels]
        out = {"pixels": torch.stack(frames), "action": torch.as_tensor(action, dtype=torch.long)}
        if self.tokens is None:
            return out

        senders = [j for j in range(self.agents) if j != i]
        rows = np.arange(r - self.max_age, r + T)
        inside = rows >= self.episode_start[r]
        rows = np.maximum(rows, self.episode_start[r])
        sent = inside[:, None] & ~self.done[rows][:, senders]
        sender = np.concatenate([self.position[rows][:, senders], self.direction[rows][:, senders, None]], -1)
        receiver = np.concatenate([self.position[rows, i], self.direction[rows, i][:, None]], -1)
        out.update(msg_tokens=torch.as_tensor(np.asarray(self.tokens[rows][:, senders]), dtype=torch.long),
                   msg_sent=torch.as_tensor(sent), msg_sender=torch.as_tensor(sender, dtype=torch.long),
                   msg_receiver=torch.as_tensor(receiver, dtype=torch.long))
        return out

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

class MultiAgentPlanningDataset(Dataset):

    def __init__(
        self,
        h5_path,
        split="train",
        val_ratio=0.1,
        agents=("0", "1"),
        jepa_transform=None,
        vqvae_transform=None,
        max_length=None,
        min_length= 1
    ):
        assert split in ["train", "val"], "split must be 'train' or 'val'"
        self.h5_path = h5_path
        self.split = split
        self.agents = list(agents)
        self.jepa_transform = jepa_transform
        self.vqvae_transform = vqvae_transform
        self.max_length = max_length
        self.file = None

        with h5py.File(self.h5_path, "r") as f:
            all_episodes = sorted(list(f.keys()))

            valid_episodes = [
                k for k in all_episodes
                if np.min(f[k]["agents_success_at"]) >= min_length and f[k]['agents_success_at'][0] > 0 and f[k]['agents_success_at'][1] > 0
            ]
            dropped = len(all_episodes) - len(valid_episodes)
            if dropped:
                print(f"[MultiAgentPlanningDataset] dropped {dropped} degenerate episode(s) "
                    f"with success_at < {min_length}")

            num_episodes = len(valid_episodes)
            val_count = int(num_episodes * val_ratio)
            train_count = num_episodes - val_count
            self.episode_keys = (
                valid_episodes[:train_count] if split == "train" else valid_episodes[train_count:]
            )

        print(f"Initialized {self.split} planning split with {len(self.episode_keys)} episodes.")

    def __len__(self):
        return len(self.episode_keys)

    def _load_agent(self, episode, agent_id, length):
        pov_seq = episode[f"{agent_id}_pov"][:length]
        act_seq = episode[f"{agent_id}_act"][:length]
        csi_seq = episode[f"{agent_id}_csi"][:length]
        pos_seq = episode[f"{agent_id}_pos"][:length]
        dir_seq = episode[f"{agent_id}_dir"][:length]

        if self.jepa_transform:
            pov_seq = torch.stack([self.jepa_transform(pov_seq[t]) for t in range(length)])

        if self.vqvae_transform:
            pov_seq_vqvae = torch.stack([self.vqvae_transform(pov_seq[t]) for t in range(length)])

        act_seq = torch.tensor(act_seq, dtype=torch.long)
        csi_seq = torch.tensor(csi_seq, dtype=torch.complex64)
        pos_seq = torch.tensor(pos_seq, dtype=torch.long)
        dir_seq = torch.tensor(dir_seq, dtype=torch.long)
        return pov_seq, act_seq, pov_seq_vqvae, csi_seq, pos_seq, dir_seq

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

        episode_key = self.episode_keys[idx]
        episode = self.file[episode_key]

        length = np.min(episode["agents_success_at"]).item()

        if self.max_length is not None:
            length = min(length, self.max_length)

        out = {
            "episode_key": episode_key,
            "length": length,
            "goal_pos": torch.tensor(episode["goal_pos"], dtype=torch.long),
            "goal_obs": torch.stack([
                self.jepa_transform(episode["goal_obs"][i]) for i in range(len(self.agents))
            ]),
        }

        for agent_id in self.agents:
            pixels, action, pov_seq_vqvae, csi_seq, pos_seq, dir_seq = self._load_agent(episode, agent_id, length)

            info = {
                        "pixels": pixels,
                        "action": action,
                        "pov_seq_vqvae": pov_seq_vqvae,
                        "csi": csi_seq,
                        "pos": pos_seq,
                        "dir": dir_seq,
                        "success_at": torch.tensor(episode['agents_success_at'][int(agent_id)], dtype=torch.long)
                    }

            out[f"agent_{agent_id}"] = info

        return out

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
