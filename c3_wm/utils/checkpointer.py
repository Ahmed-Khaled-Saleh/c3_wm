__all__ = ['BaseCheckpointer', 'RetrospectiveCheckpointer']

import os
import pathlib
from pathlib import Path
import heapq
import torch

class BaseCheckpointer:
    def __init__(self, project_name, ckp_dir, slurm_jobid, agent_id= None, rank=0):
        self.agent_id = agent_id if agent_id else 'default'
        self.rank = rank
        self.checkpoint_dir = ckp_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, state, current_acc, step):
        raise NotImplementedError("Must implement save_checkpoint in subclass")

    def get_checkpoint_path(self):
        raise NotImplementedError("Must implement get_checkpoint_path in subclass")

class RetrospectiveCheckpointer(BaseCheckpointer):
    def __init__(self, project_name, ckp_dir, slurm_jobid, agent_id= None, rank= 0, n_best=3):
        super().__init__(project_name, ckp_dir, slurm_jobid, agent_id, rank)

        self.n_best = n_best
        self.best_checkpoints = []
        self.last_path = os.path.join(self.checkpoint_dir, "last_state.pt")

        if self.rank == 0 and not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def save_checkpoint(self, state, current_acc, step):
        if self.rank != 0:
            return

        state['id'] = self.agent_id
        torch.save(state, self.last_path)

        filename = f"best_state_step_{step}_acc_{current_acc:.4f}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)

        if len(self.best_checkpoints) < self.n_best:
            torch.save(state, filepath)
            heapq.heappush(self.best_checkpoints, (current_acc, filename))
            print(f"[Rank 0] Saved new best: {filename}")

        elif current_acc > self.best_checkpoints[0][0]:
            old_acc, old_filename = heapq.heappop(self.best_checkpoints)
            old_filepath = os.path.join(self.checkpoint_dir, old_filename)

            if os.path.exists(old_filepath):
                os.remove(old_filepath)

            torch.save(state, filepath)
            heapq.heappush(self.best_checkpoints, (current_acc, filename))
            print(f"[Rank 0] Replaced {old_filename} with {filename}")

    def get_checkpoint_path(self):
        if not self.best_checkpoints:
            return None
        best_filename = max(self.best_checkpoints, key=lambda x: x[0])[1]
        return os.path.join(self.checkpoint_dir, best_filename)
