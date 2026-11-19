__all__ = ['JEPAGoalPlanner', 'DiscreteCEMPlanner', 'DiscreteCEMSolver']

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastcore.utils import patch
from loguru import logger

class JEPAGoalPlanner:
    def __init__(self, model, action_dim= 4, horizon=3, history_size=3,
                 pop_size=300, topk=30, opt_steps=30, device='cpu'):
        self.model = model
        self.action_dim = action_dim
        self.horizon = horizon
        self.history_size = history_size
        self.pop_size = pop_size
        self.topk = topk
        self.opt_steps = opt_steps
        self.device = device

    def _sample_actions(self, probs):
        B, H, A = probs.shape
        flat = probs.unsqueeze(1).expand(B, self.pop_size, H, A).reshape(-1, A)
        idx = torch.multinomial(flat, 1).view(B, self.pop_size, H)
        return idx.float()

    def _update_dist(self, cost, samples):
        B, S, H = samples.shape
        A = self.action_dim
        new_probs = torch.zeros(B, H, A, device=cost.device)
        for b in range(B):
            elite_idx = torch.topk(-cost[b], self.topk).indices
            elites = samples[b, elite_idx]
            for t in range(H):
                counts = torch.bincount(elites[:, t].long(), minlength=A).float()
                new_probs[b, t] = (counts + 1e-6) / (counts.sum() + 1e-6 * A)
        return new_probs

    @torch.no_grad()
    def plan(self, info_dict):
        device = self.device
        B = info_dict["pixels"].size(0)
        probs = torch.full((B, self.horizon, self.action_dim), 1.0 / self.action_dim, device=device)
        print("Starting planning with {} optimization steps...".format(self.opt_steps))
        for _ in range(self.opt_steps):
            samples = self._sample_actions(probs)

            hist_action = info_dict["action"].unsqueeze(1).expand(
                B, self.pop_size, *info_dict["action"].shape[1:]
            )
            action_candidates = torch.cat([hist_action, samples], dim=2).long().to(device)

            cand_info = {
                k: (v.unsqueeze(1).expand(B, self.pop_size, *v.shape[1:]).to(device)
                    if torch.is_tensor(v) else v)
                for k, v in info_dict.items()
            }

            cost = self.model.get_cost(cand_info, action_candidates)
            probs = self._update_dist(cost, samples)
            print("Optimization step completed.")

        plan = torch.argmax(probs, dim=-1)
        first_action = plan[:, 0]
        return first_action, plan

    @torch.no_grad()
    def eval_plan(self, info_dict, device, plan):

        final_action_seq = torch.cat(
        [info_dict["action"],plan],
        dim=1,
        ).unsqueeze(1).to(device)

        final_info = {
            k: (v.unsqueeze(1).to(device) if torch.is_tensor(v) else v)
            for k, v in info_dict.items()
        }
        final_cost = self.model.get_cost(final_info, final_action_seq).squeeze(1)

        return final_cost


class DiscreteCEMPlanner:

    def __init__(self, model, action_dim=4, horizon=3, history_size=3,
                 pop_size=300, topk=30, opt_steps=30,
                 smoothing=0.0, alpha=0.0, batch_size=None,
                 device='cpu', seed=0):
        self.model = model
        self.action_dim = action_dim
        self.horizon = horizon
        self.history_size = history_size
        self.pop_size = pop_size
        self.topk = topk
        self.opt_steps = opt_steps
        self.smoothing = smoothing
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)


@patch
def _sample_actions(self: DiscreteCEMPlanner, probs):
    B, H, A = probs.shape
    log_probs = probs.clamp_min(1e-10).log()
    log_probs = log_probs.unsqueeze(1).expand(B, self.pop_size, H, A)

    u = torch.rand(
        log_probs.shape, generator=self.torch_gen,
        device=self.device, dtype=probs.dtype,
    ).clamp_min(1e-10)
    gumbel = -(-u.log()).log()

    indices = (log_probs + gumbel).argmax(dim=-1)
    indices[:, 0] = probs.argmax(dim=-1)
    return indices


@patch
def _update_dist(self: DiscreteCEMPlanner, cost, samples):
    B, S, H = samples.shape
    A = self.action_dim

    _, topk_inds = torch.topk(cost, k=self.topk, dim=1, largest=False)
    batch_idx = torch.arange(B, device=cost.device).unsqueeze(1).expand(-1, self.topk)
    elites = samples[batch_idx, topk_inds]

    elite_onehot = F.one_hot(elites, num_classes=A).float()
    new_probs = elite_onehot.mean(dim=1)

    if self.smoothing > 0:
        new_probs = new_probs + self.smoothing
        new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)

    return new_probs

@patch
def _plan_chunk(self: DiscreteCEMPlanner, chunk_info, verbose=False, chunk_label=""):
    cur_bs = chunk_info["pixels"].size(0)
    probs = torch.full(
        (cur_bs, self.horizon, self.action_dim), 1.0 / self.action_dim, device=self.device
    )

    cand_info = {
        k: (v.unsqueeze(1).expand(cur_bs, self.pop_size, *v.shape[1:]).to(self.device)
            if torch.is_tensor(v) else v)
        for k, v in chunk_info.items()
    }
    hist_action = chunk_info["action"].unsqueeze(1).expand(
        cur_bs, self.pop_size, *chunk_info["action"].shape[1:]
    ).to(self.device)

    for step in range(self.opt_steps):
        logger.info(f"[DiscreteCEMPlanner]{chunk_label} step {step + 1}/{self.opt_steps} -- "
                    f"mean probs: {probs.mean(dim=(0,1)).cpu().numpy()}")

        samples = self._sample_actions(probs)
        action_candidates = torch.cat([hist_action, samples], dim=2).long()

        cost = self.model.get_cost(cand_info, action_candidates)
        new_probs = self._update_dist(cost, samples)

        probs = (
            self.alpha * probs + (1 - self.alpha) * new_probs
            if self.alpha > 0 else new_probs
        )

        if verbose:
            logger.info(
                f"[DiscreteCEMPlanner]{chunk_label} step {step + 1}/{self.opt_steps} "
                f"mean elite cost: {cost.mean().item():.4f}"
            )

    return probs


@patch
@torch.no_grad()
def plan(self: DiscreteCEMPlanner, info_dict, verbose=False):
    B = info_dict["pixels"].size(0)
    chunk_size = self.batch_size or B

    probs_full = torch.empty(B, self.horizon, self.action_dim, device=self.device)

    for start_idx in range(0, B, chunk_size):
        logger.info(f"Planning for chunk {start_idx}:{min(start_idx + chunk_size, B)}...")
        end_idx = min(start_idx + chunk_size, B)
        chunk_info = {
            k: (v[start_idx:end_idx] if torch.is_tensor(v) else v)
            for k, v in info_dict.items()
        }
        probs_full[start_idx:end_idx] = self._plan_chunk(
            chunk_info, verbose=verbose, chunk_label=f" chunk[{start_idx}:{end_idx}]"
        )

    plan = torch.argmax(probs_full, dim=-1)
    first_action = plan[:, 0]
    logger.info(f"Planning completed. Returning first actions and full plans.")
    logger.info(f"First actions: {first_action.cpu().numpy()}")
    logger.info(f"Full plans: {plan.cpu().numpy()}")
    return first_action, plan


@patch
@torch.no_grad()
def eval_plan(self: DiscreteCEMPlanner, info_dict, device, plan):
    final_action_seq = torch.cat(
        [info_dict["action"], plan], dim=1
    ).unsqueeze(1).to(device).long()

    final_info = {
        k: (v.unsqueeze(1).to(device) if torch.is_tensor(v) else v)
        for k, v in info_dict.items()
    }
    final_cost = self.model.get_cost(final_info, final_action_seq).squeeze(1)
    return final_cost


import numpy as np
from stable_marl.planning.solver import SolverBase, gather_agents

class DiscreteCEMSolver(SolverBase):
    discrete = True

    def __init__(self, cost, pop_size: int = 300, topk: int = 30, opt_steps: int = 30, smoothing: float = 0.0,
                 alpha: float = 0.0, batch_size: int | None = None, mode: str = 'per_agent', device='cpu', seed: int = 0,
                 verbose: bool = False):
        super().__init__(cost, batch_size=batch_size, num_samples=pop_size, mode=mode, device=device, seed=seed,
                         verbose=verbose)
        self.pop_size, self.topk, self.opt_steps, self.smoothing, self.alpha = pop_size, topk, opt_steps, smoothing, alpha

    def configure(self, *, action_space, n_envs: int, config):
        super().configure(action_space=action_space, n_envs=n_envs, config=config)
        if self.action_block != 1:
            raise ValueError("DiscreteCEMSolver plans one env step per planning step (action_block 1)")

    @torch.no_grad()
    def solve(self, info_dict: dict, init_action=None) -> dict:
        n = len(next(v for v in info_dict.values() if isinstance(v, (torch.Tensor, np.ndarray, list))))
        H, A, K, S = self.horizon, self.num_agents, self._num_actions, self.num_samples
        probs = torch.full((n, H, A, K), 1.0 / K, dtype=self.dtype, device=self.device)
        final_costs = []
        for start, end in self._batches(n):
            B, p, infos = end - start, probs[start:end], self._expand(info_dict, start, end)
            for _ in range(self.opt_steps):
                u = torch.rand((B, S, H, A, K), generator=self.torch_gen, device=self.device, dtype=self.dtype)
                idx = (p.clamp_min(1e-10).log().unsqueeze(1) - (-u.clamp_min(1e-10).log()).log()).argmax(-1)
                idx[:, 0] = p.argmax(-1)
                one_hot = F.one_hot(idx, num_classes=K).to(self.dtype)
                costs = self._costs(self.cost.get_cost(infos, one_hot), B, S)
                top = torch.topk(costs, k=self.topk, dim=1, largest=False).indices
                new = gather_agents(one_hot, top).mean(dim=1)
                if self.smoothing > 0:
                    new = (new + self.smoothing) / (new + self.smoothing).sum(-1, keepdim=True)
                p = self.alpha * p + (1 - self.alpha) * new if self.alpha > 0 else new
            final_costs += self._cost_list(costs.gather(1, top).mean(dim=1))
            probs[start:end] = p
        return {'actions': probs.argmax(-1).unsqueeze(-1).cpu(), 'costs': final_costs, 'probs': [probs.cpu()]}
