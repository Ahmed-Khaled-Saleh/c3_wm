__all__ = ['DataModule', 'VQDataModule', 'WMDataModule', 'planning_collate_fn', 'PlanningDataModule']

import os

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
import torch.nn.functional as F

import h5py

from .datasets import FindGoalPOVDataset, FindGoalWorldModelDataset, MultiAgentPOVDataset, MultiAgentWorldModelDataset, MultiAgentPlanningDataset
from .transforms import get_transforms

class DataModule:
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 64,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 persistent_workers: bool = False,
                 val_shuffle_seed: int | None = None):

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.val_shuffle_seed = val_shuffle_seed

    def setup(self):
        raise NotImplementedError("Subclasses should implement this method to load datasets.")

    def get_data_path(self):
        raise ValueError("data_dir is not set")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=self.sampler(self.train_dataset),
            collate_fn=self.collator(),
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=self.val_sampler(self.val_dataset),
            collate_fn=self.collator(),
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return None

    def sampler(self, dataset):
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if torch.distributed.is_initialized() else None
        return dist_sampler


    def val_sampler(self, dataset):
        sampler = self.sampler(dataset)
        if sampler is not None or self.val_shuffle_seed is None:
            return sampler
        return torch.randperm(len(dataset), generator=torch.Generator().manual_seed(self.val_shuffle_seed)).tolist()

    def collator(self):
        colate_fn= torch.utils.data.default_collate if torch.distributed.is_initialized() else None
        return colate_fn


class VQDataModule(DataModule):
    def __init__(self,
                 batch_size: int = 64,
                 img_size: int = 224,
                 data_dir: str | None = None,
                 val_layouts=(200, 260),
                 val_frames: int | None = None,
                 **kwargs):

        data_dir = data_dir or self.get_data_path()
        self.img_size = img_size
        self.val_layouts = tuple(val_layouts)
        self.val_frames = val_frames
        super().__init__(data_dir= data_dir, batch_size= batch_size, **kwargs)

    def setup(self):
        self.train_transforms, self.val_transforms = get_transforms(img_size= self.img_size)
        h5_path = os.path.join(self.data_dir, "dataset.h5")
        with h5py.File(h5_path, "r") as f:
            flat = "pixels" in f

        if flat:
            self.train_dataset = FindGoalPOVDataset(h5_path, "train", self.val_layouts, self.train_transforms)
            self.val_dataset = FindGoalPOVDataset(h5_path, "val", self.val_layouts, self.val_transforms)
            if self.val_frames and self.val_frames < len(self.val_dataset):
                picks = np.sort(np.random.default_rng(0).choice(len(self.val_dataset), self.val_frames, replace=False))
                self.val_dataset = torch.utils.data.Subset(self.val_dataset, picks.tolist())
        else:
            self.train_dataset  = MultiAgentPOVDataset(
                h5_path= h5_path, split= "train", transform= self.train_transforms
            )

            self.val_dataset  = MultiAgentPOVDataset(
                h5_path= h5_path, split= "val", transform= self.val_transforms
            )

class WMDataModule(DataModule):
    def __init__(self,
                 batch_size: int = 64,
                 img_size: int = 224,
                 history_size: int = 3,
                 data_dir: str | None = None,
                 val_layouts=(200, 260),
                 num_preds: int = 1,
                 max_age: int = 8,
                 tokens=None,
                 **kwargs):

        data_dir = data_dir or self.get_data_path()
        self.img_size = img_size
        self.history_size = history_size
        self.val_layouts, self.num_preds, self.max_age, self.tokens = tuple(val_layouts), num_preds, max_age, tokens
        super().__init__(data_dir= data_dir, batch_size= batch_size, **kwargs)


    def setup(self):

        self.train_sender_transform, self.val_sender_transform = get_transforms(img_size= self.img_size)
        self.train_receiver_transform, self.val_receiver_transform = get_transforms(img_size= self.img_size, model= "JEPA")
        h5_path = os.path.join(self.data_dir, "dataset.h5")
        with h5py.File(h5_path, "r") as f:
            flat = "pixels" in f

        if flat:
            common = dict(val_layouts=self.val_layouts, history_size=self.history_size, num_preds=self.num_preds,
                          tokens=self.tokens, max_age=self.max_age)
            self.train_dataset = FindGoalWorldModelDataset(h5_path, "train", transform=self.train_receiver_transform, **common)
            self.val_dataset = FindGoalWorldModelDataset(h5_path, "val", transform=self.val_receiver_transform, **common)
            return

        self.train_dataset  = MultiAgentWorldModelDataset(
            h5_path= h5_path,
            split= "train",
            history_size= self.history_size,
            receiver_transform= self.train_receiver_transform,
            sender_transform= self.train_sender_transform
        )

        self.val_dataset  = MultiAgentWorldModelDataset(
            h5_path= h5_path,
            split= "val",
            history_size= self.history_size,
            receiver_transform= self.val_receiver_transform,
            sender_transform= self.val_sender_transform
        )

def planning_collate_fn(batch):
    agent_keys = [k for k in batch[0].keys() if k.startswith("agent_")]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    T_max = lengths.max().item()

    out = {
        "episode_key": [item["episode_key"] for item in batch],
        "goal_pos": torch.stack([item["goal_pos"] for item in batch]),
        "goal_obs": torch.stack([item["goal_obs"] for item in batch]),
        "length": lengths,
    }
    for agent_key in agent_keys:
        pixels = _pad_stack([item[agent_key]["pixels"] for item in batch], T_max)
        action = _pad_stack([item[agent_key]["action"] for item in batch], T_max)
        pov_vqvae = _pad_stack([item[agent_key]["pov_seq_vqvae"] for item in batch], T_max)
        csi = _pad_stack([item[agent_key]["csi"] for item in batch], T_max)
        pos = _pad_stack([item[agent_key]["pos"] for item in batch], T_max)
        dir_ = _pad_stack([item[agent_key]["dir"] for item in batch], T_max)
        success_at = torch.stack([item[agent_key]["success_at"] for item in batch])

        mask = torch.arange(T_max).unsqueeze(0) < lengths.unsqueeze(1)

        out[agent_key] = {
            "pixels": pixels, "action": action, "pov_seq_vqvae": pov_vqvae,
            "csi": csi, "pos": pos, "dir": dir_, "mask": mask,
            "success_at": success_at,
        }
    return out


def _pad_stack(tensors, T_max):
    padded = []
    for t in tensors:
        pad_len = T_max - t.size(0)
        if pad_len > 0:
            pad_shape = (pad_len, *t.shape[1:])
            pad = torch.zeros(pad_shape, dtype=t.dtype)
            t = torch.cat([t, pad], dim=0)
        padded.append(t)
    return torch.stack(padded, dim=0)

class PlanningDataModule(DataModule):
    def __init__(self,
                 batch_size: int = 64,
                 img_size: int = 224,
                 max_length= 150,
                 history_size: int = 3,
                 goal_offset: int = 10,
                 data_dir: str | None = None,
                 **kwargs):

        data_dir = data_dir or self.get_data_path()
        self.img_size = img_size
        self.max_length = max_length
        self.history_size = history_size
        self.goal_offset = goal_offset
        super().__init__(data_dir= data_dir, batch_size= batch_size, **kwargs)

    def setup(self):

        self.train_sender_transform, self.val_sender_transform = get_transforms(img_size= self.img_size)
        self.train_receiver_transform, self.val_receiver_transform = get_transforms(img_size= self.img_size, model= "JEPA")

        self.train_dataset  = MultiAgentPlanningDataset(
            h5_path= os.path.join(self.data_dir, "dataset.h5"),
            split= "train",
            jepa_transform= self.train_receiver_transform,
            vqvae_transform= self.train_sender_transform,
            max_length= self.max_length,
            min_length= self.history_size + self.goal_offset + 1
        )

        self.val_dataset  = MultiAgentPlanningDataset(
            h5_path= os.path.join(self.data_dir, "dataset.h5"),
            split= "val",
            jepa_transform= self.val_receiver_transform,
            vqvae_transform= self.val_sender_transform,
            max_length= self.max_length,
            min_length= self.history_size + self.goal_offset + 1
        )

    def collator(self):
        return planning_collate_fn
