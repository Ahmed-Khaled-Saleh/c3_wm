from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from loguru import logger
from torch import nn

from comm_core import Fading
from stable_marl.comm.core import AgentNetwork, node_id
from stable_marl.comm.wrapper import CommunicationWrapper

from ..data.datasets import TILE_NAMES, _relative_pose
from ..comm.channel_data import _pair

MESSAGE_TOKENS = 49
POSE = 3

__all__ = ['MESSAGE_TOKENS', 'POSE', 'IMAGENET', 'HEADS', 'STARTS', 'DISTANCE_BINS', 'NO_COST', 'OnlineRicianFading',
           'C3WMCommunication', 'vq_tokens', 'c3wm_link', 'ActionSubset', 'jepa_transform', 'GoalHead', 'CostToGoHead',
           'CombinedHead', 'TileHead', 'MemoryCostHead', 'prior_mlp', 'PolicyPriorHead', 'RecurrentPriorHead',
           'load_head', 'load_heads', 'C3WMCost', 'MessageInfluence', 'TrajectoryRecorder', 'StartRecorder', 'Surprise',
           'GoalMemory', 'BeliefMap', 'LatentMemory', 'StuckEscape', 'save_trajectories', 'find_wrapper', 'by_start',
           'age_of_information', 'decision_time', 'fewest_actions', 'decision_rows', 'decision_quality', 'deploy',
           'merge_results']

class OnlineRicianFading(Fading):

    def __init__(
        self,
        cache,
        rho: float = 0.9,
        rician: bool = True,
    ):
        self.cache, self.rho, self.rician = cache, rho, rician
        self.node_positions, self.round, self.state = {}, 0, {}

    def reset(self, seed=None):
        super().reset(seed)
        self.node_positions, self.round, self.state = {}, 0, {}

    def update(self, time):
        self.round += 1

    def set_positions(self, positions):
        self.node_positions.update({k: tuple(int(v) for v in p) for k, p in positions.items()})

    def _draw(self) -> complex:
        return complex(self.rng.standard_normal(), self.rng.standard_normal()) / math.sqrt(2)

    def fade(self, a_node, b_node) -> float:
        first, second = sorted((a_node, b_node))
        cells = (self.node_positions[first], self.node_positions[second])
        s = self.state.get((first, second))
        if s is None or s['cells'] != cells:
            s = self.state[(first, second)] = dict(cells=cells, w=self._draw(), phase=self.rng.uniform(0, 2 * math.pi),
                                                    round=self.round)
        elif s['round'] < self.round:
            s['w'] = self.rho * s['w'] + math.sqrt(1 - self.rho ** 2) * self._draw()
            s['round'] = self.round
        k = _pair(*cells)
        d = min(self.cache.D.get(k, 0.0) / max(self.cache.P[k], 1e-30), 1.0)
        return abs(math.sqrt(d) * complex(math.cos(s['phase']), math.sin(s['phase'])) + math.sqrt(1 - d) * s['w']) ** 2

    def coefficient(self, transmission, link) -> complex:
        a, b = self.node_positions[transmission.sender], self.node_positions[transmission.receiver]
        if a == b:
            return math.inf
        gain = self.cache.gain(a, b, 'large_scale')
        if self.rician:
            gain *= self.fade(transmission.sender, transmission.receiver)
        return complex(math.sqrt(gain))

class C3WMCommunication(CommunicationWrapper):

    def __init__(
        self,
        env: gym.Env,
        tokens: Callable | None,
        link: Callable | None = None,
        max_age: int = 8,
        delivery: float = 1.0,
        withhold: tuple | None = None,
    ):
        super().__init__(env, message_fn=self._payload, message_dim=MESSAGE_TOKENS + POSE)
        self.tokens, self.link, self.max_age, self.delivery = tokens, link, max_age, float(delivery)
        self.withhold, self._round = withhold, 0
        self._erase = np.random.default_rng()
        self.records: list = []
        self.ages: list = []
        A = self.num_agents
        extra = {'msg_tokens': spaces.Box(0, np.iinfo(np.int64).max, (A, MESSAGE_TOKENS), np.int64),
                 'msg_received': spaces.MultiBinary(A),
                 'msg_age': spaces.Box(0, max_age, (A,), np.int64),
                 'msg_pose': spaces.Box(-np.inf, np.inf, (A, POSE), np.int64)}
        self.observation_space = spaces.Dict({a: spaces.Dict({**self.observation_space[a].spaces, **extra})
                                              for a in range(A)})
        self._payloads = {}
        self._clear()

    def _clear(self):
        A = self.num_agents
        self._memory = dict(tokens=np.zeros((A, A, MESSAGE_TOKENS), np.int64), received=np.zeros((A, A), bool),
                            age=np.zeros((A, A), np.int64), sender=np.zeros((A, A, POSE), np.int64))
        self._heard = np.full((A, A), -1, np.int64)

    def _payload(self, env, agent, obs):
        return self._payloads.get(agent)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, infos = self.env.reset(seed=seed, options=options)
        layout = int(np.asarray(infos[0].get('layout', -1)))
        options = self.link(layout, self.env.unwrapped) if self.link is not None else {}
        self.comm = AgentNetwork(self.num_agents, 'full_mesh', bits_per_element=8, **options)
        self.comm.reset(seed)
        self._erase = np.random.default_rng(None if seed is None else [int(seed), 11])
        self._clear()
        self._round = 0
        return self._communicate(obs, infos)

    def _communicate(self, obs: dict, infos: dict):
        A = self.num_agents
        done = [bool(agent.state.terminated) for agent in self.env.unwrapped.agents]
        sends = [not done[a] and not all(done[j] for j in range(A) if j != a) for a in range(A)]
        if self.tokens is None or not any(sends):
            self._payloads = {}
        else:
            toks = np.asarray(self.tokens(np.stack([obs[a]['pixels'] for a in range(A)])))
            self._payloads = {a: np.r_[toks[a], np.asarray(obs[a]['position']).reshape(-1), int(obs[a]['direction'])]
                              .astype(np.int64) for a in range(A) if sends[a]}
        obs, infos = super()._communicate(obs, infos)
        self._record()
        out = self._remember(obs)
        if self.tokens is not None:
            self.ages += [int(self._heard[a, j]) for a in range(A) for j in range(A) if j != a and not done[a] and not done[j]]
        return out, infos

    def _record(self):
        for r in self.comm.last_results:
            m = r.metadata or {}
            if 'channel_gain_db' in m and 'tx_power' in m:
                self.records.append((10 ** (m['channel_gain_db'] / 10), m['tx_power'], not m.get('outage', False),
                                     bool(r.success), m.get('wrong_tokens')))

    def _remember(self, obs: dict) -> dict:
        A, mem = self.num_agents, self._memory
        start, rounds = self.withhold or (None, None)
        withheld = start is not None and start <= self._round and (rounds is None or self._round < start + rounds)
        self._round += 1
        mem['age'] += 1
        self._heard = np.where(self._heard >= 0, self._heard + 1, -1)
        for a in range(A):
            arrived = np.flatnonzero(obs[a]['message_mask'])
            if self.delivery < 1:
                arrived = arrived[self._erase.random(len(arrived)) < self.delivery]
            if withheld:
                arrived = arrived[:0]
            for j in arrived:
                m = np.rint(obs[a]['messages'][j]).astype(np.int64)
                mem['tokens'][a, j] = m[:MESSAGE_TOKENS]
                mem['sender'][a, j] = m[MESSAGE_TOKENS:MESSAGE_TOKENS + POSE]
                mem['age'][a, j], mem['received'][a, j] = 0, True
                self._heard[a, j] = 0
        mem['received'] &= mem['age'] <= self.max_age
        mem['age'] = np.where(mem['received'], mem['age'], 0)
        out = dict(obs)
        for a in range(A):
            pose = _relative_pose(mem['sender'][a, :, :2], mem['sender'][a, :, 2], np.asarray(obs[a]['position']),
                                  int(obs[a]['direction']))
            out[a] = {**obs[a], 'msg_tokens': mem['tokens'][a].copy(), 'msg_received': mem['received'][a].copy(),
                      'msg_age': mem['age'][a].copy(), 'msg_pose': pose.astype(np.int64)}
        return out

def vq_tokens(vqvae, device="cpu") -> Callable:
    @torch.no_grad()
    def tokens(pixels):
        x = torch.as_tensor(pixels, device=device).permute(0, 3, 1, 2).float() / 127.5 - 1
        return vqvae.get_message_indices(x).flatten(1).cpu().numpy()
    return tokens

def c3wm_link(method: str, cache_for: Callable, saved: dict | None = None, crc: bool = True, rho: float = 0.9,
              rician: bool = True, policies: dict | None = None, policy: str | None = None, target: float = 0.95) -> Callable:
    from comm_core import ChannelInversion, FixedPower
    from c3_wm.comm.power_control import LearnedChannelInversion, c3wm_link as blocks
    from c3_wm.comm.evaluation import DistributionalChannelInversion
    if method == 'perfect channel':
        return lambda layout, env: {}
    if method not in ('perfect CSI', 'learned', 'lambda-policy', 'fixed power'):
        raise ValueError(f"method: 'perfect channel', 'perfect CSI', 'learned', 'lambda-policy' or 'fixed power', not {method!r}")
    if method == 'lambda-policy' and (policies is None or policy not in policies['policies']):
        raise ValueError(f"'lambda-policy' needs `policies` with {policy!r}: {list((policies or {}).get('policies', {}))}")
    b = policies['budget'] if method == 'lambda-policy' else saved['budget']

    def link(layout, env):
        cache = cache_for(layout)
        fading = OnlineRicianFading(cache, rho, rician)
        heads = lambda: {node_id(i): int(agent.state.dir) for i, agent in enumerate(env.agents)}
        if method == 'perfect CSI':
            control, tau = ChannelInversion(b.snr_target_db), 0.0
        elif method == 'fixed power':
            control, tau = FixedPower(b.max_power_w), math.inf
        elif method == 'lambda-policy':
            control = DistributionalChannelInversion.from_saved(policies, policy, target, cache.walls,
                                                                lambda: fading.node_positions, heads)
            tau = math.inf
        else:
            control = LearnedChannelInversion.from_saved(saved, cache.walls, lambda: fading.node_positions, heads)
            tau = saved['calibration']['tau_db']
        return dict(blocks=blocks(fading, control, b.max_power_w, tau, crc=crc), noise_power=b.noise_w)
    return link

class ActionSubset(gym.ActionWrapper):

    def __init__(self, env: gym.Env, n: int = 3):
        super().__init__(env)
        self.action_space = spaces.Dict({a: spaces.Discrete(n) for a in env.action_space.spaces})

    def action(self, action):
        return action

IMAGENET = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def jepa_transform(images: torch.Tensor) -> torch.Tensor:
    mean, std = (torch.tensor(v, device=images.device).view(1, 3, 1, 1) for v in IMAGENET)
    return (images.float() / 255 - mean) / std

class GoalHead(nn.Module):

    VIEW, AGENT, kind = 7, (3, 6), 'goal_location'

    def __init__(self, dim: int = 192, hidden: int = 256, absent: float = 12.0):
        super().__init__()
        self.config = dict(dim=dim, hidden=hidden, absent=absent)
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, self.VIEW ** 2 + 1))
        x, y = torch.meshgrid(torch.arange(self.VIEW), torch.arange(self.VIEW), indexing='ij')
        (ax, ay) = self.AGENT
        moves = (x - ax).abs() + (y - ay).abs() + (x != ax).long()
        self.register_buffer('moves', torch.cat([moves.flatten().float(), torch.tensor([float(absent)])]))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)

    def distance(self, emb: torch.Tensor, goal_emb: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward(emb).softmax(-1) @ self.moves

class CostToGoHead(nn.Module):

    kind = 'cost_to_go'

    def __init__(self, dim: int = 192, hidden: int = 256):
        super().__init__()
        self.config = dict(dim=dim, hidden=hidden)
        self.net = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, emb: torch.Tensor, goal_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb, goal_emb.expand_as(emb)], -1)
        return nn.functional.softplus(self.net(x)).squeeze(-1)

    def distance(self, emb: torch.Tensor, goal_emb: torch.Tensor) -> torch.Tensor:
        return self.forward(emb, goal_emb)

class CombinedHead(nn.Module):

    kind = 'combined'

    def __init__(self, location: GoalHead, cost_to_go: CostToGoHead):
        super().__init__()
        self.location, self.cost_to_go = location, cost_to_go

    def distance(self, emb: torch.Tensor, goal_emb: torch.Tensor) -> torch.Tensor:
        p = self.location(emb).softmax(-1)
        return p[..., :-1] @ self.location.moves[:-1] + p[..., -1] * self.cost_to_go(emb, goal_emb)

class TileHead(nn.Module):

    kind, AHEAD = 'tiles', 5 * 7 + 3

    def __init__(self, dim: int = 192, hidden: int = 256, kinds: int = len(TILE_NAMES)):
        super().__init__()
        self.config = dict(dim=dim, hidden=hidden, kinds=kinds)
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, MESSAGE_TOKENS * kinds))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb).unflatten(-1, (MESSAGE_TOKENS, self.config['kinds']))

    def wall_ahead(self, emb: torch.Tensor) -> torch.Tensor:
        return self.forward(emb)[..., self.AHEAD, :].softmax(-1)[..., TILE_NAMES.index('wall')]

class MemoryCostHead(nn.Module):

    kind, NO_ACTION, ACTIONS = 'memory_cost', 3, 3
    FORWARD = ((1, 0), (0, 1), (-1, 0), (0, -1))

    def __init__(self, dim: int = 192, hidden: int = 256, codes: int = 256, token_dim: int = 16, reads: str = 'embedding',
                 queries: int = 4, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if reads not in ('tokens', 'embedding'):
            raise ValueError(f"reads: tokens or embedding, not {reads!r}")
        self.config = dict(dim=dim, hidden=hidden, codes=codes, token_dim=token_dim, reads=reads, queries=queries, heads=heads)
        if dropout:
            self.config['dropout'] = dropout
        self.reads, self.dropout = reads, dropout
        self.frame = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden))
        self.action = nn.Embedding(self.ACTIONS + 1, hidden)
        self.token = nn.Embedding(codes, token_dim)
        if reads == 'tokens':
            k = torch.arange(MESSAGE_TOKENS)
            self.register_buffer('tile', torch.stack([6 - k // 7, k % 7 - 3], -1))
            self.register_buffer('forward_', torch.tensor(self.FORWARD))
            self.where = nn.Sequential(nn.Linear(2, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))
            self.query = nn.Parameter(0.02 * torch.randn(1, queries, token_dim))
            self.attend = nn.MultiheadAttention(token_dim, heads, batch_first=True)
            self.read = nn.Sequential(nn.LayerNorm(queries * token_dim), nn.Linear(queries * token_dim, hidden))
        else:
            self.message = nn.Sequential(nn.Linear(MESSAGE_TOKENS * token_dim + 2 + 4, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        drop = [nn.Dropout(dropout)] if dropout else []
        self.moves = nn.Sequential(nn.LayerNorm(dim + hidden), *drop, nn.Linear(dim + hidden, hidden), nn.GELU(), *drop,
                                   nn.Linear(hidden, hidden), nn.GELU(), *drop, nn.Linear(hidden, 1))
        self.goal = nn.Linear(hidden, 2)

    def cells(self, pose: torch.Tensor) -> torch.Tensor:
        f, r = self.forward_[pose[..., 2] % 4], self.forward_[(pose[..., 2] + 1) % 4]
        return pose[..., None, :2] + self.tile[..., :1] * f[..., None, :] + self.tile[..., 1:] * r[..., None, :]

    def _read(self, codes: torch.Tensor, pose: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        lead, V, d = codes.shape[:-2], codes.shape[-2], self.config['token_dim']
        feat = self.token(codes) + self.where(self.cells(pose).float() / 15)
        kv = feat.reshape(-1, V * MESSAGE_TOKENS, d)
        some = valid.reshape(-1, V).any(-1)
        ignore = ~valid.reshape(-1, V, 1).expand(-1, V, MESSAGE_TOKENS).reshape(-1, V * MESSAGE_TOKENS)
        ignore = ignore & some[:, None]
        out, _ = self.attend(self.query.expand(len(kv), -1, -1), kv, kv, key_padding_mask=ignore, need_weights=False)
        return (self.read(out.flatten(1)) * some[:, None]).reshape(*lead, -1)

    def inputs(self, emb: torch.Tensor, action: torch.Tensor, tokens=None, pose=None, new=None, own=None) -> torch.Tensor:
        x = self.frame(emb) + self.action(action)
        if self.reads == 'embedding':
            if tokens is not None:
                about = torch.cat([self.token(tokens).flatten(-2), pose[..., :2].float() / 15,
                                   nn.functional.one_hot(pose[..., 2] % 4, 4).float()], -1)
                x = x + (self.message(about) * new[..., None].float()).sum(-2)
            return x
        views, poses, valid = [], [], []
        if own is not None:
            views.append(own[..., None, :]), poses.append(torch.zeros(*own.shape[:-1], 1, 3, dtype=torch.long, device=own.device))
            valid.append(torch.ones(*own.shape[:-1], 1, dtype=torch.bool, device=own.device))
        if tokens is not None:
            views.append(tokens), poses.append(pose.long()), valid.append(new.bool())
        if views:
            x = x + self._read(torch.cat(views, -2), torch.cat(poses, -2), torch.cat(valid, -1))
        return x

    def cost(self, emb: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(self.moves(torch.cat([emb, memory], -1))).squeeze(-1)

    def observe(self, x: torch.Tensor, h: torch.Tensor | None = None, lengths: torch.Tensor | None = None):
        h0 = None if h is None else h[None].contiguous()
        if self.dropout and self.training:
            x = nn.functional.dropout(x, self.dropout)
        if lengths is None:
            out, hn = self.gru(x, h0)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, hn = self.gru(packed, h0)
            out = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])[0]
        return out, hn[0]

    def imagine(self, memory: torch.Tensor, imagined: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        lead = imagined.shape[:-2]
        out, _ = self.observe(self.inputs(imagined, actions).flatten(0, -3), memory.flatten(0, -2))
        return self.cost(imagined, out.unflatten(0, lead))

def prior_mlp(inputs: int, width: int, actions: int, dropout: float = 0.0) -> nn.Sequential:
    drop = [nn.Dropout(dropout)] if dropout else []
    return nn.Sequential(nn.LayerNorm(inputs), *drop, nn.Linear(inputs, width), nn.GELU(), *drop,
                         nn.Linear(width, width), nn.GELU(), *drop, nn.Linear(width, actions))

class PolicyPriorHead(nn.Module):

    kind = 'policy_prior'

    def __init__(self, dim: int = 192, hidden: int = 256, width: int = 256, actions: int = 3, dropout: float = 0.0):
        super().__init__()
        self.config = dict(dim=dim, hidden=hidden, width=width, actions=actions, dropout=dropout)
        self.net = prior_mlp(dim + hidden, width, actions, dropout)

    def forward(self, emb: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return nn.functional.log_softmax(self.net(torch.cat([emb, memory], -1)), -1)

class RecurrentPriorHead(MemoryCostHead):

    kind = 'recurrent_prior'

    def __init__(self, dim: int = 192, hidden: int = 256, codes: int = 256, token_dim: int = 16, reads: str = 'embedding',
                 queries: int = 4, heads: int = 4, width: int = 256, actions: int = 3, dropout: float = 0.0):
        super().__init__(dim, hidden, codes, token_dim, reads, queries, heads, dropout)
        self.config.update(width=width, actions=actions, dropout=dropout)
        self.policy = prior_mlp(dim + hidden, width, actions, dropout)

    def forward(self, emb: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return nn.functional.log_softmax(self.policy(torch.cat([emb, memory], -1)), -1)

HEADS = {h.kind: h for h in (GoalHead, CostToGoHead, TileHead, MemoryCostHead, PolicyPriorHead, RecurrentPriorHead)}

def load_head(path, map_location='cpu') -> nn.Module:
    saved = torch.load(path, map_location=map_location)
    head = HEADS[saved.get('kind', 'goal_location')](**saved['config'])
    head.load_state_dict(saved['state_dict'])
    return head.eval()

def load_heads(paths, map_location='cpu') -> nn.Module:
    if isinstance(paths, (str, bytes)) or hasattr(paths, '__fspath__'):
        return load_head(paths, map_location)
    heads = {h.kind: h for h in (load_head(p, map_location) for p in paths)}
    if len(paths) == 1:
        return next(iter(heads.values()))
    if len(paths) != 2 or set(heads) != {'goal_location', 'cost_to_go'}:
        raise ValueError(f"two heads combine as a goal_location and a cost_to_go one, not {[type(h).__name__ for h in heads.values()]}")
    return CombinedHead(heads['goal_location'], heads['cost_to_go']).eval()

class C3WMCost(nn.Module):

    COSTS = ('min', 'terminal', 'mean')
    GOAL_COSTS = ('straight', 'map')
    HEADINGS = ((1, 0), (0, 1), (-1, 0), (0, -1))

    def __init__(self, model, history_size: int = 3, messages: bool | str = True, cost: str = 'min', every: int = 1,
                 explore: bool = False, explore_below: float = 0.05, message_effect: bool = False,
                 goal_cost: str | None = None, goal_head: nn.Module | None = None, wall_head: nn.Module | None = None,
                 wall_above: float = 0.5, memory_head: nn.Module | None = None, move_messages: bool = False,
                 step_cost: float = 0.01, near_step_cost: float | None = None, near_below: float = 4.0,
                 novelty: str = 'history', explore_within: float | None = None, imagine_noise: float = 0.0,
                 cic: bool = False, prior_head: nn.Module | None = None, prior_weight: float = 1.0):
        super().__init__()
        if prior_head is not None and memory_head is None and getattr(prior_head, 'kind', None) != 'recurrent_prior':
            raise ValueError("prior_head reads the memory: with memory_head only (a RecurrentPriorHead reads its own)")
        if cost not in self.COSTS:
            raise ValueError(f"cost: one of {self.COSTS}, not {cost!r}")
        if goal_cost not in (None, *self.GOAL_COSTS):
            raise ValueError(f"goal_cost: None or one of {self.GOAL_COSTS}, not {goal_cost!r}")
        self.model, self.history_size, self.messages, self.cost = model, history_size, messages, cost
        self.gaps, self.spreads, self.every, self._calls = [], [], every, 0
        self.explore, self.explore_below, self.explored, self._flat = explore, explore_below, [], {}
        self.message_effect, self.heard, self.changed = message_effect, [], []
        self.goal_cost, self.known = goal_cost, []
        self.goal_head, self.wall_head, self.wall_above = goal_head, wall_head, wall_above
        self.memory_head, self.move_messages, self.step_cost = memory_head, move_messages, step_cost
        self.near_step_cost, self.near_below = near_step_cost, near_below
        if novelty not in ('history', 'episode'):
            raise ValueError(f"novelty: history or episode, not {novelty!r}")
        self.novelty, self.episodic = novelty, None
        self.explore_within = explore_within
        self.imagine_noise = imagine_noise
        self.cic, self.cic_values, self.silent_changed, self.content_changed = cic, [], [], []
        self.first_costs = {}
        self.resampled = []
        self.first_best = {}
        self.prior_head, self.prior_weight, self._now, self._prior = prior_head, prior_weight, {}, {}
        self._prior_key = 'prior_memory' if getattr(prior_head, 'kind', None) == 'recurrent_prior' else None
        self._escape, self.escaped = {}, []

    def _score(self, info: dict, actions: torch.Tensor | None = None, memory: torch.Tensor | None = None) -> torch.Tensor:
        pred = info['predicted_emb'][..., info.get('observed', self.history_size):, :]
        if memory is not None:
            moves = self.memory_head.imagine(memory.to(pred.device)[:, None].expand(-1, pred.shape[1], -1), pred, actions)
            step = torch.full(moves.shape[:1], self.step_cost, device=pred.device)
            if self.near_step_cost is not None:
                near = moves.flatten(1).min(1).values < self.near_below
                step = torch.where(near, torch.full_like(step, self.near_step_cost), step)
            d = moves + step[:, None, None] * torch.arange(1, pred.shape[-2] + 1, device=pred.device)
        elif self.goal_head is not None:
            d = self.goal_head.distance(pred, info['goal_emb'][..., -1:, :]) + \
                self.step_cost * torch.arange(1, pred.shape[-2] + 1, device=pred.device)
        else:
            d = (pred - info['goal_emb'][..., -1:, :]).pow(2).sum(-1)
        return {'min': d.min(-1).values, 'terminal': d[..., -1], 'mean': d.mean(-1)}[self.cost]

    @torch.no_grad()
    def prior_log(self, info_dict: dict) -> torch.Tensor:
        pixels = info_dict['pixels']
        out = []
        for a in range(pixels.shape[3]):
            now = self.model.encode({'pixels': pixels[:, 0, -1, a].unsqueeze(1)})['emb'][:, -1].float()
            out.append(self.prior_head(now, self._prior_memory(info_dict, a)))
        return torch.stack(out, 1)

    def _prior_memory(self, info_dict: dict, a: int) -> torch.Tensor:
        if self._prior_key is None:
            return self._memory(info_dict, a)
        if self._prior_key not in info_dict:
            raise KeyError("a RecurrentPriorHead reads its own memory: wrap the envs in LatentMemory(key='prior_memory')")
        return info_dict[self._prior_key][:, 0, -1, a].float()

    def _with_prior(self, cost: torch.Tensor, first: torch.Tensor, a: int, memory: torch.Tensor | None = None) -> torch.Tensor:
        if self.prior_head is None:
            return cost
        logp = self._prior[a] if memory is None else self.prior_head(self._now[a], memory.float())
        return cost - self.prior_weight * logp.gather(1, first)

    def _memory(self, info_dict: dict, a: int) -> torch.Tensor | None:
        if self.memory_head is None:
            return None
        if 'memory' not in info_dict:
            raise KeyError("memory_head plans from LatentMemory's observations (`memory`): wrap the envs in it")
        return info_dict['memory'][:, 0, -1, a].float()

    def _novelty(self, info: dict, a: int | None = None) -> torch.Tensor:
        emb = info['predicted_emb']
        seen = emb[..., :info.get('observed', self.history_size), :]
        near = (emb[..., -1:, :] - seen).pow(2).sum(-1).min(-1).values
        if self.novelty != 'episode' or self.episodic is None or a is None:
            return near
        episodes = self.episodic()
        if len(episodes) != emb.shape[0]:
            return near
        last = emb[..., -1, :]
        return torch.stack([torch.cdist(last[b].float(), torch.stack(episodes[b][a]).to(last.device).float()).pow(2).min(-1).values
                            if episodes[b][a] else near[b] for b in range(emb.shape[0])])

    def _moves_to_goal(self, info_dict: dict, actions: torch.Tensor, a: int, imagined: torch.Tensor | None = None) -> torch.Tensor:
        last = lambda k: info_dict[k][:, :, -1, a]
        pos, heading, goal = last('position').long(), last('direction').long(), last('goal_cell').long()
        vec = torch.tensor(self.HEADINGS, device=pos.device)
        if self.goal_cost == 'map':
            free, moves = last('free').bool().flatten(2), last('goal_moves').long().flatten(2)
            height = last('free').shape[-1]
            seen = last('seen').bool().flatten(2) if 'seen' in info_dict else torch.ones_like(free)
            imagined_wall = self.wall_head.wall_ahead(imagined) > self.wall_above \
                if self.wall_head is not None and imagined is not None else None
        steps = []
        for t in range(actions.shape[-1]):
            act = actions[..., t]
            heading = torch.where(act == 0, (heading - 1) % 4, torch.where(act == 1, (heading + 1) % 4, heading))
            nxt = pos + vec[heading] * (act == 2).unsqueeze(-1)
            if self.goal_cost == 'map':
                cell = lambda p: (p[..., 0] * height + p[..., 1]).unsqueeze(-1)
                open_ = free.gather(-1, cell(nxt))
                if imagined_wall is not None and t > 0:
                    unseen = ~seen.gather(-1, cell(nxt))
                    open_ = open_ & ~(unseen & imagined_wall[..., t - 1:t] & (act == 2).unsqueeze(-1))
                pos = torch.where(open_, nxt, pos)
                d = moves.gather(-1, cell(pos)).squeeze(-1)
                steps.append(torch.where(d < 0, torch.full_like(d, 1000), d))
            else:
                pos = nxt
                steps.append((pos - goal).abs().sum(-1))
        d = torch.stack(steps, -1).float()
        return (d + 0.01 * torch.arange(1, d.shape[-1] + 1, device=d.device)).min(-1).values

    @staticmethod
    def _per_first(costs: torch.Tensor, first: torch.Tensor, K: int) -> torch.Tensor:
        return torch.stack([torch.where(first == k, costs, torch.full_like(costs, float('inf'))).min(1).values
                            for k in range(K)], -1)

    @classmethod
    def _gap(cls, costs: torch.Tensor, first: torch.Tensor, K: int) -> torch.Tensor:
        best = cls._per_first(costs, first, K).sort(-1).values
        spread = costs.median(1).values - costs.min(1).values
        gap = (best[:, 1] - best[:, 0]) / spread.clamp_min(1e-8)
        return torch.where(torch.isfinite(best[:, 1]), gap, torch.full_like(gap, float('nan')))

    def _note(self, costs: torch.Tensor, gap: torch.Tensor):
        spread = costs.median(1).values - costs.min(1).values
        self.gaps += gap[torch.isfinite(gap)].tolist()
        self.spreads += (spread / costs.median(1).values.abs().clamp_min(1e-8)).tolist()

    def landscape(self) -> dict:
        return self.landscape_of(self.raw())

    def raw(self) -> dict:
        return dict(gaps=list(self.gaps), spreads=list(self.spreads), explored=list(self.explored), heard=list(self.heard),
                    changed=list(self.changed), known=list(self.known), cic=list(self.cic_values),
                    silent_changed=list(self.silent_changed), content_changed=list(self.content_changed),
                    resampled=list(self.resampled), escaped=list(self.escaped))

    @staticmethod
    def landscape_of(raw: dict) -> dict:
        g, s = np.array(raw['gaps']), np.array(raw['spreads'])
        explored, heard, changed, known = (np.array(raw[k], bool) for k in ('explored', 'heard', 'changed', 'known'))
        return {'first_action_gap': float(np.median(g)) if len(g) else None,
                'first_action_ties': float((g < 0.05).mean()) if len(g) else None,
                'cost_spread': float(np.median(s)) if len(s) else None,
                'explored': float(explored.mean()) if len(explored) else None,
                'heard': float(heard.mean()) if len(heard) else None,
                'message_changes': float(changed[heard].mean()) if heard.any() else None,
                'goal_known': float(known.mean()) if len(known) else None,
                'cic': float(np.mean(raw['cic'])) if raw.get('cic') else None,
                'memory_silent_changes': float(np.mean(raw['silent_changed'])) if raw.get('silent_changed') else None,
                'content_changes': float(np.mean(raw['content_changed'])) if raw.get('content_changed') else None,
                'cic_soft': float(np.mean(raw['cic_soft'])) if raw.get('cic_soft') else None,
                'resampled': float(np.mean(raw['resampled'])) if raw.get('resampled') else None,
                'escaped': float(np.mean(raw['escaped'])) if raw.get('escaped') else None}

    def _intervene(self, info: dict, actions: torch.Tensor, info_dict: dict, a: int, cost: torch.Tensor, K: int):
        heard = info_dict['memory_cf_heard'][:, 0, -1, a].reshape(-1) > 0
        if not heard.any():
            return
        cf = info_dict['memory_cf'][:, 0, -1, a].float()
        best = lambda c: self._per_first(c, actions[:, :, 0], K).argmin(-1)
        chosen = best(cost)
        key = f'{self._prior_key}_cf' if self._prior_key else None
        prior_cf = cf if key is None else info_dict[key][:, 0, -1, a].float() if key in info_dict else None
        under = torch.stack([best(self._with_prior(self._score(info, actions, cf[:, k]), actions[:, :, 0], a,
                                                   None if prior_cf is None else prior_cf[:, k]))
                             for k in range(cf.shape[1])], 1)
        for b in torch.nonzero(heard).flatten().tolist():
            self.silent_changed.append(bool(under[b, 0] != chosen[b]))
            others = under[b, 1:]
            self.content_changed.append(float((others != chosen[b]).float().mean()))
            acts = torch.cat([chosen[b:b + 1], others])
            p = torch.bincount(acts, minlength=K).float() / len(acts)
            p = p[p > 0]
            self.cic_values.append(float(-(p * p.log2()).sum()))

    def criterion(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        return self.get_cost(info_dict, action_candidates)

    @torch.no_grad()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        B, S, _, A = action_candidates.shape[:4]
        plan = action_candidates.argmax(-1)
        pixels, history = info_dict['pixels'], info_dict.get('action_history')
        T = pixels.shape[2]
        goal = info_dict['goal'].reshape(B, S, -1, A, *pixels.shape[-3:])[:, :, -1]
        costs, note = [], self._calls % self.every == 0
        self._calls += 1
        if note:
            self.first_best = {}
        for a in range(A):
            actions = plan[:, :, :, a]
            if history is not None and T > 1:
                past = torch.nan_to_num(history[:, :, :, a].reshape(B, S, -1).float(), nan=0.0).long()
                actions = torch.cat([past[:, :, -(T - 1):], actions], dim=2)
            info = {'pixels': pixels[:, :, :, a]}
            if self.messages and 'msg_tokens' in info_dict:
                heard = info_dict['msg_received'][:, :, :, a].bool()
                info['messages'] = {'tokens': info_dict['msg_tokens'][:, :, :, a].long(),
                                    'received': torch.zeros_like(heard) if self.messages == 'nothing' else heard,
                                    'age': info_dict['msg_age'][:, :, :, a].long(),
                                    'pose': info_dict['msg_pose'][:, :, :, a].long()}
            goal_emb = self.model.encode({'pixels': goal[:, 0, a].unsqueeze(1)})['emb']
            info['goal_emb'] = goal_emb.unsqueeze(1).expand(B, S, 1, -1)
            inputs = dict(info)
            info = self.model.rollout(info, actions, history_size=self.history_size, move_messages=self.move_messages,
                                      noise=self.imagine_noise)
            memory = self._memory(info_dict, a)
            cost = self._score(info, plan[:, :, :, a], memory)
            if self.prior_head is not None:
                if note or a not in self._prior:
                    self._now[a] = self.model.encode({'pixels': pixels[:, 0, -1, a].unsqueeze(1)})['emb'][:, -1].float()
                    self._prior[a] = self.prior_head(self._now[a], self._prior_memory(info_dict, a))
                cost = self._with_prior(cost, plan[:, :, 0, a], a)
            first = self._per_first(cost, plan[:, :, 0, a], action_candidates.shape[-1])
            self.first_best[a] = first if a not in self.first_best else torch.minimum(self.first_best[a], first)
            if note:
                self.first_costs[a] = self._per_first(cost, plan[:, :, 0, a], action_candidates.shape[-1]).cpu()
            known = info_dict['goal_known'][:, 0, -1, a].bool() if self.goal_cost else torch.zeros(B, dtype=torch.bool)
            if note:
                gap = self._gap(cost, plan[:, :, 0, a], action_candidates.shape[-1])
                self._note(cost, gap)
                if self.message_effect and 'messages' in inputs:
                    m = inputs['messages']
                    silent = self._score(self.model.rollout(
                        dict(inputs, messages={**m, 'received': torch.zeros_like(m['received'])}), actions,
                        history_size=self.history_size, move_messages=self.move_messages, noise=self.imagine_noise),
                        plan[:, :, :, a], memory)
                    silent = self._with_prior(silent, plan[:, :, 0, a], a)
                    best = lambda c: self._per_first(c, plan[:, :, 0, a], action_candidates.shape[-1]).argmin(-1)
                    self.heard += m['received'][:, 0, -1].flatten(1).any(-1).tolist()
                    self.changed += (best(cost) != best(silent)).tolist()
                if self.cic and memory is not None and 'memory_cf' in info_dict:
                    self._intervene(info, plan[:, :, :, a], info_dict, a, cost, action_candidates.shape[-1])
                if self.goal_cost:
                    self.known += known.tolist()
                if self.explore:
                    self._flat[a] = (gap < self.explore_below) & ~known.to(gap.device)
                    if 'escape' in info_dict:
                        self._escape[a] = info_dict['escape'][:, 0, -1, a, 0].to(gap.device) > 0.5
                        self._flat[a] = self._flat[a] | self._escape[a]
                        self.escaped += self._escape[a].tolist()
                    self.explored += self._flat[a].tolist()
            if self.explore:
                novel = everywhere = -self._novelty(info, a)
                if self.explore_within is not None:
                    best = cost.min(1, keepdim=True).values
                    novel = torch.where(cost <= best + self.explore_within, novel, novel.max() + 1 + cost)
                cost = torch.where(self._flat[a][:, None], novel, cost)
                if a in self._escape:
                    cost = torch.where(self._escape[a][:, None], everywhere, cost)
            if known.any():
                cost = torch.where(known.to(cost.device)[:, None], self._moves_to_goal(
                    info_dict, plan[:, :, :, a], a, info['predicted_emb'][..., info.get('observed', self.history_size):, :]), cost)
            costs.append(cost)
        return torch.stack(costs, dim=-1)

class MessageInfluence:

    def __init__(self):
        self.cic, self.cic_soft, self.silent_changed, self.content_changed = [], [], [], []

    def note(self, probs: torch.Tensor, silent: torch.Tensor):
        probs, silent = probs.float().cpu(), silent.float().cpu()
        best = probs.argmax(-1)
        self.silent_changed.append(bool(silent.argmax() != best[0]))
        self.content_changed.append(float((best[1:] != best[0]).float().mean()))
        p = torch.bincount(best, minlength=probs.shape[1]).float() / len(best)
        p = p[p > 0]
        self.cic.append(float(-(p * p.log2()).sum()))
        marginal = probs.mean(0, keepdim=True)
        terms = probs * (probs.clamp_min(1e-12).log2() - marginal.clamp_min(1e-12).log2())
        self.cic_soft.append(float(terms.sum(-1).mean()))

    def raw(self) -> dict:
        return dict(gaps=[], spreads=[], explored=[], heard=[], changed=[], known=[], cic=list(self.cic),
                    silent_changed=list(self.silent_changed), content_changed=list(self.content_changed),
                    cic_soft=list(self.cic_soft))

    def landscape(self) -> dict:
        return C3WMCost.landscape_of(self.raw())

def _over(terminated, truncated) -> bool:
    flags = lambda d: np.array(list(d.values()) if isinstance(d, dict) else d, bool)
    return bool((flags(terminated) | flags(truncated)).all())

class TrajectoryRecorder(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episodes, self._current = [], None

    def _state(self):
        u = self.env.unwrapped
        cells, dirs = np.full((len(u.agents), 2), -1, np.int16), np.full(len(u.agents), -1, np.int8)
        for i, agent in enumerate(u.agents):
            if not agent.state.terminated and tuple(agent.state.pos) != (-1, -1):
                cells[i], dirs[i] = agent.state.pos, int(agent.state.dir)
        return cells, dirs

    def reset(self, *, seed=None, options=None):
        from c3_wm.utils.env_utils import LayoutWrapper
        out = self.env.reset(seed=seed, options=options)
        layout = find_wrapper(self.env, LayoutWrapper)
        self._current = dict(seed=None if seed is None else int(seed), layout=None if layout is None else layout.layout,
                             steps=[self._state()])
        return out

    def step(self, actions):
        out = self.env.step(actions)
        if self._current is not None:
            if _over(out[2], out[3]):
                c = self._current
                self.episodes.append(dict(seed=c['seed'], layout=c['layout'], cells=np.stack([s[0] for s in c['steps']]),
                                          dirs=np.stack([s[1] for s in c['steps']])))
                self._current = None
            else:
                self._current['steps'].append(self._state())
        return out

def _goal_steps(u, start) -> int:
    from collections import deque
    goal, start = tuple(int(v) for v in u.goal_pos), tuple(int(v) for v in start)
    free = lambda x, y: 0 <= x < u.grid.width and 0 <= y < u.grid.height and \
        ((o := u.grid.get(x, y)) is None or o.can_overlap())
    seen, queue = {start: 0}, deque([start])
    while queue:
        x, y = cell = queue.popleft()
        if cell == goal:
            return seen[cell]
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nxt not in seen and free(*nxt):
                seen[nxt] = seen[cell] + 1
                queue.append(nxt)
    return -1

def _in_view(u, cell) -> np.ndarray:
    from stable_marl.envs.multigrid.obs import get_view_exts
    pos = np.array([a.state.pos for a in u.agents], np.int_)
    dirs = np.array([int(a.state.dir) for a in u.agents], np.int_)
    size = u.agents[0].view_size
    top_left, (gx, gy) = get_view_exts(dirs, pos, size), (int(v) for v in cell)
    return (top_left[:, 0] <= gx) & (gx < top_left[:, 0] + size) & (top_left[:, 1] <= gy) & (gy < top_left[:, 1] + size)

class StartRecorder(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.starts, self._seed, self._t = {}, None, 0

    def _in_view(self):
        return _in_view(self.env.unwrapped, self.env.unwrapped.goal_pos)

    def reset(self, *, seed=None, options=None):
        out = self.env.reset(seed=seed, options=options)
        self._seed, self._t = (None if seed is None else int(seed)), 0
        if seed is not None:
            u = self.env.unwrapped
            in_view = self._in_view()
            self.starts[int(seed)] = dict(distance=np.array([_goal_steps(u, a.state.pos) for a in u.agents]), in_view=in_view,
                                          seen_at=np.where(in_view, 0, -1), reached_at=np.full(len(u.agents), -1))
        return out

    def step(self, actions):
        out = self.env.step(actions)
        self._t += 1
        s = self.starts.get(self._seed)
        if s is not None:
            done = np.array([a.state.terminated for a in self.env.unwrapped.agents])
            rewarded = np.array(list(out[1].values()) if isinstance(out[1], dict) else out[1], float) > 0
            s['reached_at'] = np.where((s['reached_at'] < 0) & rewarded, self._t, s['reached_at'])
            s['seen_at'] = np.where((s['seen_at'] < 0) & self._in_view() & ~done, self._t, s['seen_at'])
        return out

def _moves_from(u, cell, extra=()) -> dict:
    from collections import deque
    blocked = set(extra)
    free = lambda x, y: (x, y) not in blocked and 0 <= x < u.grid.width and 0 <= y < u.grid.height and \
        ((o := u.grid.get(x, y)) is None or o.can_overlap())
    cell = tuple(int(v) for v in cell)
    seen, queue = {cell: 0}, deque([cell])
    while queue:
        x, y = queue.popleft()
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nxt not in seen and free(*nxt):
                seen[nxt] = seen[(x, y)] + 1
                queue.append(nxt)
    return seen

class Surprise(gym.Wrapper):

    def __init__(self, env: gym.Env, kind: str, when='seen', delay: int = 3, min_move: int = 5):
        super().__init__(env)
        if kind not in ('goal', 'wall'):
            raise ValueError(f"surprise kind: goal or wall, not {kind!r}")
        if when != 'seen' and not isinstance(when, int):
            raise ValueError(f"surprise when: 'seen' or a step number, not {when!r}")
        self.kind, self.when, self.delay, self.min_move = kind, when, int(delay), int(min_move)
        self.events, self._seed, self._t, self._seen, self._rng = {}, None, 0, -1, None

    def reset(self, *, seed=None, options=None):
        out = self.env.reset(seed=seed, options=options)
        self._seed, self._t, self._seen = (None if seed is None else int(seed)), 0, -1
        self._rng = np.random.default_rng(None if seed is None else [int(seed), 11])
        u = self.env.unwrapped
        if seed is not None:
            self.events[int(seed)] = dict(at=-1, size=0, seen_after=np.full(len(u.agents), -1))
        self._watch(u)
        return out

    def _acting(self, u) -> np.ndarray:
        return np.array([not a.state.terminated for a in u.agents])

    def _watch(self, u):
        view = _in_view(u, u.goal_pos) & self._acting(u)
        if self._seen < 0 and view.any():
            self._seen = self._t
        e = self.events.get(self._seed)
        if e is not None and e['at'] >= 0:
            e['seen_after'] = np.where((e['seen_after'] < 0) & view, self._t, e['seen_after'])

    def step(self, actions):
        u = self.env.unwrapped
        e = self.events.get(self._seed)
        due = self._t == self.when if self.when != 'seen' else self._seen >= 0 and self._t >= self._seen + self.delay
        if e is not None and e['at'] < 0 and due and self._acting(u).any():
            e['size'], e['at'] = self._move_goal(u) if self.kind == 'goal' else self._close_path(u), self._t
            self._watch(u)
        out = self.env.step(actions)
        self._t += 1
        self._watch(u)
        return out

    def _move_goal(self, u) -> int:
        from stable_marl.envs.multigrid.core.constants import Color
        from stable_marl.envs.multigrid.core.world_object import Goal
        old = tuple(int(v) for v in u.goal_pos)
        agents = {tuple(int(v) for v in a.state.pos) for a in u.agents}
        reach = _moves_from(u, old)
        cells = [c for c in reach if c != old and c not in agents and (u.grid.get(*c) is None)]
        far = [c for c in cells if abs(c[0] - old[0]) + abs(c[1] - old[1]) >= self.min_move]
        hidden = [c for c in far if not _in_view(u, c).any()]
        choice = hidden or far or cells
        new = choice[self._rng.integers(len(choice))]
        u.grid.set(*old, None)
        u.grid.set(*new, Goal(Color.from_index(int(u._variation('goal.color')))))
        u.goal_pos = np.asarray(new)
        u._goal_infos = u.goal_infos()
        return abs(new[0] - old[0]) + abs(new[1] - old[1])

    def _close_path(self, u) -> int:
        from stable_marl.envs.multigrid.core.constants import Color
        from stable_marl.envs.multigrid.core.world_object import Wall
        goal = tuple(int(v) for v in u.goal_pos)
        acting = self._acting(u)
        pos = [tuple(int(v) for v in a.state.pos) for a in u.agents]
        view = _in_view(u, goal) & acting
        dist = _moves_from(u, goal)
        who = int(np.flatnonzero(view)[0]) if view.any() else \
            min(np.flatnonzero(acting), key=lambda i: dist.get(pos[i], 10 ** 6))
        path, cell = [], pos[who]
        while cell != goal and cell in dist:
            cell = min(((cell[0] + 1, cell[1]), (cell[0] - 1, cell[1]), (cell[0], cell[1] + 1), (cell[0], cell[1] - 1)),
                       key=lambda c: dist.get(c, 10 ** 6))
            path.append(cell)
        taken = set(pos) | {goal}
        best, added = None, 0
        for k, c in enumerate(path[:-1]):
            nxt = path[k + 1]
            across = (0, 1) if nxt[0] != c[0] else (1, 0)
            for block in ([c], [c, (c[0] + across[0], c[1] + across[1]), (c[0] - across[0], c[1] - across[1])]):
                block = [b for b in block if b not in taken and u.grid.get(*b) is None]
                if c not in block:
                    continue
                after = _moves_from(u, goal, block)
                if any(acting[i] and pos[i] not in after for i in range(len(pos))):
                    continue
                more = after[pos[who]] - dist[pos[who]]
                if more > added:
                    best, added = block, more
        if best is None:
            return 0
        color = Color.from_index(int(u._variation('wall.color')))
        for b in best:
            u.grid.set(*b, Wall(color))
        return added

class GoalMemory(gym.Wrapper):

    def __init__(self, env: gym.Env, messages: bool = True):
        super().__init__(env)
        self.messages = messages
        u = env.unwrapped
        self._A, self._size = len(self.observation_space.spaces), (u.width, u.height)
        extra = {'goal_known': spaces.Discrete(2), 'goal_cell': spaces.Box(0, max(self._size), (2,), np.int64),
                 'free': spaces.MultiBinary(self._size), 'seen': spaces.MultiBinary(self._size),
                 'goal_moves': spaces.Box(-1, np.iinfo(np.int64).max, self._size, np.int64)}
        self.observation_space = spaces.Dict({a: spaces.Dict({**self.observation_space[a].spaces, **extra})
                                              for a in range(self._A)})

    def reset(self, *, seed=None, options=None):
        from c3_wm.utils.env_utils import goal_moves
        obs, infos = self.env.reset(seed=seed, options=options)
        u = self.env.unwrapped
        self._known, self._cell = np.zeros(self._A, bool), np.asarray(u.goal_pos, np.int64)
        self._moves = np.full(self._size, -1, np.int64)
        for (x, y), m in goal_moves(u).items():
            self._moves[x, y] = m
        return self._observe(obs), infos

    def step(self, actions):
        obs, *rest = self.env.step(actions)
        return (self._observe(obs), *rest)

    def _observe(self, obs: dict) -> dict:
        goal = TILE_NAMES.index('goal')
        sees = np.array([(np.asarray(obs[a]['image'])[..., 0] == goal).any() for a in range(self._A)])
        for a in range(self._A):
            heard = np.asarray(obs[a].get('message_mask', np.zeros(self._A)), bool) if self.messages else np.zeros(self._A, bool)
            self._known[a] |= sees[a] | (heard & sees).any()
        return {**obs, **{a: {**obs[a], 'goal_known': int(self._known[a]), 'goal_cell': self._cell.copy(),
                              'free': (self._moves >= 0).astype(np.int8), 'seen': np.ones(self._size, np.int8),
                              'goal_moves': self._moves.copy()}
                          for a in range(self._A)}}

def _moves_to(free: np.ndarray, cell) -> np.ndarray:
    from collections import deque
    moves = np.full(free.shape, -1, np.int64)
    moves[tuple(cell)] = 0
    queue = deque([tuple(int(v) for v in cell)])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < free.shape[0] and 0 <= ny < free.shape[1] and free[nx, ny] and moves[nx, ny] < 0:
                moves[nx, ny] = moves[x, y] + 1
                queue.append((nx, ny))
    return moves

class BeliefMap(gym.Wrapper):

    def __init__(self, env: gym.Env, tokens: Callable, table, messages: bool = True):
        super().__init__(env)
        self.tokens, self.table, self.messages = tokens, np.asarray(table, np.int64), messages
        u = env.unwrapped
        self._A, self._size = len(self.observation_space.spaces), (u.width, u.height)
        extra = {'goal_known': spaces.Discrete(2), 'goal_cell': spaces.Box(0, max(self._size), (2,), np.int64),
                 'free': spaces.MultiBinary(self._size), 'seen': spaces.MultiBinary(self._size),
                 'goal_moves': spaces.Box(-1, np.iinfo(np.int64).max, self._size, np.int64)}
        self.observation_space = spaces.Dict({a: spaces.Dict({**self.observation_space[a].spaces, **extra})
                                              for a in range(self._A)})
        self.stats = dict(goal_steps=0, goal_right=0, walls=0, walls_right=0)
        view = MESSAGE_TOKENS
        self._x, self._y = np.arange(view) % 7, np.arange(view) // 7
        self._kinds = {k: TILE_NAMES.index(k) for k in ('wall', 'goal', 'unseen')}

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        u, (W, H) = self.env.unwrapped, self._size
        self._map = np.full((self._A, W, H), -1, np.int8)
        self._border = np.zeros((W, H), bool)
        self._border[[0, -1], :] = self._border[:, [0, -1]] = True
        self._map[:, self._border] = 1
        self._votes = np.zeros((self._A, W, H), np.int32)
        self._true_walls = np.array([[(o := u.grid.get(x, y)) is not None and not o.can_overlap() for y in range(H)]
                                     for x in range(W)])
        self._comm = find_wrapper(self.env, C3WMCommunication)
        return self._observe(obs), infos

    def step(self, actions):
        obs, *rest = self.env.step(actions)
        return (self._observe(obs), *rest)

    def _place(self, a: int, codes, pose):
        px, py, d = (int(v) for v in pose)
        (W, H), kinds = self._size, self.table[np.asarray(codes, np.int64)]
        (fx, fy), ahead, right = C3WMCost.HEADINGS[d % 4], 6 - self._y, self._x - 3
        cx, cy = px + fx * ahead - fy * right, py + fy * ahead + fx * right
        use = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H) & (kinds != self._kinds['unseen'])
        wall = use & (kinds == self._kinds['wall'])
        self._map[a, cx[wall], cy[wall]] = 1
        self._map[a, cx[use & ~wall], cy[use & ~wall]] = 0
        goal = use & (kinds == self._kinds['goal'])
        np.add.at(self._votes[a], (cx[goal], cy[goal]), 1)

    def _observe(self, obs: dict) -> dict:
        u = self.env.unwrapped
        own = np.asarray(self.tokens(np.stack([obs[a]['pixels'] for a in range(self._A)])))
        for a in range(self._A):
            pos = np.asarray(obs[a]['position'])
            if (pos >= 0).all():
                self._place(a, own[a], (*pos, int(obs[a]['direction'])))
            if self.messages and self._comm is not None:
                mem = self._comm._memory
                for j in np.flatnonzero(mem['received'][a] & (mem['age'][a] == 0)):
                    self._place(a, mem['tokens'][a, j], mem['sender'][a, j])
        out, goal = dict(obs), np.asarray(u.goal_pos)
        for a in range(self._A):
            self._map[a, self._border] = 1
            free = self._map[a] != 1
            known = self._votes[a].max() > 0
            cell = np.array(np.unravel_index(self._votes[a].argmax(), self._size), np.int64) if known else np.zeros(2, np.int64)
            walls = (self._map[a] == 1) & ~self._border
            self.stats['walls'] += int(walls.sum())
            self.stats['walls_right'] += int((walls & self._true_walls).sum())
            if known:
                self.stats['goal_steps'] += 1
                self.stats['goal_right'] += int((cell == goal).all())
            out[a] = {**obs[a], 'goal_known': int(known), 'goal_cell': cell, 'free': free.astype(np.int8),
                      'seen': (self._map[a] >= 0).astype(np.int8),
                      'goal_moves': _moves_to(free, cell) if known else np.full(self._size, -1, np.int64)}
        return out

class LatentMemory(gym.Wrapper):

    def __init__(self, env: gym.Env, model, head, messages: bool = True, device='cpu', tokens: Callable | None = None,
                 counterfactuals: int = 0, seed: int = 0, key: str = 'memory', record_surprise: bool = False,
                 forget_above: float | None = None, forget_keep: float = 0.0, history: int = 3):
        super().__init__(env)
        if head.reads == 'tokens' and tokens is None:
            raise ValueError("a memory head reading tokens needs `tokens` (vq_tokens) for the own view")
        self.model, self.head, self.messages, self.device, self.tokens = model, head, messages, device, tokens
        self._A, hidden = len(self.observation_space.spaces), head.config['hidden']
        self.key = key
        extra = {key: spaces.Box(-np.inf, np.inf, (hidden,), np.float32)}
        if counterfactuals:
            extra.update({f'{key}_cf': spaces.Box(-np.inf, np.inf, (1 + counterfactuals, hidden), np.float32),
                          f'{key}_cf_heard': spaces.Box(0, 1, (1,), np.float32)})
        self.observation_space = spaces.Dict({a: spaces.Dict({**self.observation_space[a].spaces, **extra})
                                              for a in range(self._A)})
        self.counterfactuals, self._rng, self._pool = counterfactuals, np.random.default_rng(seed), []
        self.record = record_surprise or forget_above is not None
        self.forget_above, self.forget_keep, self.history = forget_above, float(forget_keep), int(history)
        self.errors, self.forgot, self._seed = {}, {}, None

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self._h, self._action = None, np.full(self._A, MemoryCostHead.NO_ACTION)
        self.episode = [[] for _ in range(self._A)]
        self._views, self._actions, self._seed = [], [], None if seed is None else int(seed)
        if self.record and seed is not None:
            self.errors[int(seed)], self.forgot[int(seed)] = [], []
        return self._observe(obs), infos

    def step(self, actions):
        self._action = np.array([int(np.asarray(actions[a]).reshape(-1)[0]) for a in range(self._A)])
        self._actions.append(self._action)
        obs, *rest = self.env.step(actions)
        return (self._observe(obs), *rest)

    @torch.no_grad()
    def _observe(self, obs: dict) -> dict:
        A, dev = self._A, self.device
        pixels = torch.as_tensor(np.stack([obs[a]['pixels'] for a in range(A)]), device=dev).permute(0, 3, 1, 2)
        emb = self.model.encode({'pixels': jepa_transform(pixels)[:, None]})['emb'][:, 0]
        for a in range(A):
            self.episode[a].append(emb[a])
        if self.record:
            self._surprise(emb)
        msg = {}
        if self.messages and 'msg_tokens' in obs[0]:
            get = lambda k: torch.as_tensor(np.stack([obs[a][k] for a in range(A)]), device=dev)
            msg = dict(tokens=get('msg_tokens').long(), pose=get('msg_pose').long(),
                       new=get('msg_received').bool() & (get('msg_age') == 0))
        if self.head.reads == 'tokens':
            pix = np.stack([obs[a]['pixels'] for a in range(A)])
            msg['own'] = torch.as_tensor(np.asarray(self.tokens(pix)), device=dev).long()
        action = torch.as_tensor(self._action, device=dev)
        x = self.head.inputs(emb, action, **msg)
        before, (_, self._h) = self._h, self.head.observe(x[:, None], self._h)
        h = self._h.cpu().numpy().astype(np.float32)
        out = {**obs, **{a: {**obs[a], self.key: h[a]} for a in range(A)}}
        if self.counterfactuals:
            self._counterfactual(out, emb, action, msg, before, h)
        return out

    @torch.no_grad()
    def _surprise(self, emb: torch.Tensor):
        T, error = self.history, torch.full((self._A,), float('nan'), device=emb.device)
        if len(self._views) >= T:
            views = torch.stack(self._views[-T:], 1)
            acts = torch.as_tensor(np.stack(self._actions[-T:], 1), device=emb.device).long()
            pred = self.model.predict(views, self.model.action_encoder(acts).reshape(self._A, T, -1))[:, -1]
            error = (pred - emb).pow(2).mean(-1)
        self._views.append(emb)
        surprised = (error > self.forget_above) if self.forget_above is not None else torch.zeros_like(error, dtype=torch.bool)
        if self._h is not None and surprised.any():
            self._h = torch.where(surprised[:, None], self._h * self.forget_keep, self._h)
        if self._seed in self.errors:
            self.errors[self._seed].append(error.cpu().numpy())
            self.forgot[self._seed].append(surprised.cpu().numpy())

    @torch.no_grad()
    def _counterfactual(self, out: dict, emb, action, msg: dict, before, h):
        n, A = self.counterfactuals, self._A
        arrived = msg['new'].any(-1).tolist() if 'new' in msg else [False] * A
        for a in range(A):
            cf, measured = np.repeat(h[a][None], 1 + n, 0), arrived[a] and len(self._pool) > 0
            if measured:
                donors = [None] + [self._pool[i] for i in self._rng.integers(len(self._pool), size=n)]
                for k, donor in enumerate(donors):
                    m = {key: v[a:a + 1].clone() for key, v in msg.items()}
                    if donor is None:
                        m['new'] = torch.zeros_like(m['new'])
                    else:
                        m['tokens'], m['pose'], m['new'] = (t[None].to(m['tokens'].device) for t in donor)
                    x = self.head.inputs(emb[a:a + 1], action[a:a + 1], **m)
                    _, hk = self.head.observe(x[:, None], None if before is None else before[a:a + 1])
                    cf[k] = hk[0].cpu().numpy()
            out[a] = {**out[a], f'{self.key}_cf': cf.astype(np.float32), f'{self.key}_cf_heard': np.array([float(measured)], np.float32)}
        for a in range(A):
            if arrived[a]:
                self._pool.append(tuple(msg[k][a].clone() for k in ('tokens', 'pose', 'new')))
        del self._pool[:-500]

class StuckEscape(gym.Wrapper):

    def __init__(self, env: gym.Env, window: int = 12, cells: int = 3):
        super().__init__(env)
        self.window, self.cells, self._A = window, cells, len(self.observation_space.spaces)
        self.observation_space = spaces.Dict({a: spaces.Dict({**self.observation_space[a].spaces,
                                                              'escape': spaces.Box(0, 1, (1,), np.float32)})
                                              for a in range(self._A)})

    def reset(self, *, seed=None, options=None):
        from collections import deque
        obs, infos = self.env.reset(seed=seed, options=options)
        self._recent = [deque(maxlen=self.window) for _ in range(self._A)]
        self._visited, self._escaping = [set() for _ in range(self._A)], np.zeros(self._A, bool)
        return self._observe(obs), infos

    def step(self, actions):
        obs, *rest = self.env.step(actions)
        return (self._observe(obs), *rest)

    def _observe(self, obs: dict) -> dict:
        for a, agent in enumerate(self.env.unwrapped.agents):
            if agent.state.terminated:
                self._escaping[a] = False
                continue
            cell = tuple(int(v) for v in agent.state.pos)
            if cell not in self._visited[a]:
                if self._escaping[a]:
                    self._recent[a].clear()
                self._escaping[a] = False
            self._visited[a].add(cell)
            self._recent[a].append(cell)
            if len(self._recent[a]) == self.window and len(set(self._recent[a])) <= self.cells:
                self._escaping[a] = True
        return {**obs, **{a: {**obs[a], 'escape': np.array([float(self._escaping[a])], np.float32)} for a in range(self._A)}}

def _saved_episodes(path, layout: int) -> list:
    z = np.load(path)
    out = []
    for seed, cells, dirs in zip(z['seeds'], z['cells'], z['dirs']):
        n = int(np.flatnonzero((cells >= 0).all(-1).any(-1)).max()) + 1
        out.append(dict(seed=int(seed), layout=layout, cells=cells[:n], dirs=dirs[:n]))
    return out

def save_trajectories(episodes: list, out_dir) -> list:
    from pathlib import Path
    from c3_wm.comm.channel_data import file_lock, replace_npz
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_layout = {}
    for e in episodes:
        by_layout.setdefault(int(e['layout']), {})[int(e['seed'])] = e
    paths = []
    for layout, new in sorted(by_layout.items()):
        path = out_dir / f'trajectories_{layout}.npz'
        with file_lock(path):
            old = {e['seed']: e for e in _saved_episodes(path, layout)} if path.exists() else {}
            eps = [{**old, **new}[s] for s in sorted({**old, **new})]
            steps, agents = max(len(e['cells']) for e in eps), eps[0]['cells'].shape[1]
            cells = np.full((len(eps), steps, agents, 2), -1, np.int16)
            dirs = np.full((len(eps), steps, agents), -1, np.int8)
            for k, e in enumerate(eps):
                cells[k, :len(e['cells'])], dirs[k, :len(e['dirs'])] = e['cells'], e['dirs']
            replace_npz(path, cells=cells, dirs=dirs, behaviour=np.array(['planner'] * len(eps)), layout=layout,
                        seeds=np.array([e['seed'] for e in eps]))
        paths.append(path)
    return paths

def find_wrapper(env, cls):
    while env is not None:
        if isinstance(env, cls):
            return env
        env = getattr(env, 'env', None)
    return None

def _merged(parts: list) -> dict:
    out = {k: np.concatenate([p[k] for p in parts]) for k in ('episode_successes', 'episode_returns', 'episode_lengths', 'seeds')}
    out['success_rate'] = float(out['episode_successes'].mean()) * 100.0
    reached = out['episode_returns'] > 0
    out['agent_success_rate'] = float(reached.mean()) * 100.0
    out['any_success_rate'] = float(reached.any(axis=1).mean()) * 100.0
    out['mean_return'] = float(out['episode_returns'].sum(axis=1).mean())
    out['mean_length'] = float(out['episode_lengths'].mean())
    return out

STARTS = ('distance', 'in_view', 'seen_at', 'reached_at')
DISTANCE_BINS = (0, 10, 15, 20, 1000)

def by_start(reached, distance, in_view) -> dict:
    others = (in_view.sum(1, keepdims=True) - in_view) > 0
    groups = {'in view': in_view, 'out of view': ~in_view, 'out of view, partner sees it': ~in_view & others,
              'out of view, nobody sees it': ~in_view & ~others}
    for lo, hi in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
        groups[f'{lo}-{hi - 1} moves' if hi < 1000 else f'{lo}+ moves'] = (distance >= lo) & (distance < hi)
    return {k: {'agents': int(m.sum()), 'success': float(reached[m].mean()) * 100 if m.any() else None} for k, m in groups.items()}

def age_of_information(ages) -> dict | None:
    a = np.asarray(ages, np.int64)
    if not len(a):
        return None
    heard = a[a >= 0]
    return dict(mean=float(heard.mean()) if len(heard) else None, median=float(np.median(heard)) if len(heard) else None,
                p95=float(np.percentile(heard, 95)) if len(heard) else None, never=float((a < 0).mean()), rounds=int(len(a)))

def decision_time(times, num_envs: int) -> dict | None:
    t = np.asarray(times[1:], float)
    if not len(t):
        return None
    return dict(call_median=float(np.median(t)), call_p95=float(np.percentile(t, 95)), num_envs=int(num_envs),
                per_env_median=float(np.median(t)) / num_envs, calls=int(len(t)))

NO_COST = 1e9

def fewest_actions(u) -> np.ndarray:
    from collections import deque
    W, H = u.grid.width, u.grid.height
    free = lambda x, y: 0 <= x < W and 0 <= y < H and ((o := u.grid.get(x, y)) is None or o.can_overlap())
    vec = ((1, 0), (0, 1), (-1, 0), (0, -1))
    out = np.full((W, H, 4), np.inf)
    gx, gy = (int(v) for v in u.goal_pos)
    queue = deque()
    for d in range(4):
        out[gx, gy, d] = 0
        queue.append((gx, gy, d))
    while queue:
        x, y, d = queue.popleft()
        before = [(x, y, (d + 1) % 4), (x, y, (d - 1) % 4)]
        px, py = x - vec[d][0], y - vec[d][1]
        if free(px, py) and (px, py) != (gx, gy):
            before.append((px, py, d))
        for s in before:
            if out[s] == np.inf:
                out[s] = out[x, y, d] + 1
                queue.append(s)
    return out

def decision_rows(world, cost, actions) -> list:
    from stable_marl.envs.multigrid.obs import get_view_exts
    actions = np.asarray(actions.cpu() if hasattr(actions, 'cpu') else actions).reshape(len(world.envs.envs), -1)
    vec, rows = ((1, 0), (0, 1), (-1, 0), (0, -1)), []
    live = [e for e, env in enumerate(world.envs.envs) if not all(a.state.terminated for a in env.unwrapped.agents)]
    for e, env in enumerate(world.envs.envs):
        u = env.unwrapped
        field, size = fewest_actions(u), u.agents[0].view_size
        gx, gy = (int(v) for v in u.goal_pos)
        rec = find_wrapper(env, StartRecorder)
        s = rec.starts.get(rec._seed) if rec is not None else None
        for a, agent in enumerate(u.agents):
            if agent.state.terminated:
                continue
            (x, y), d = (int(v) for v in agent.state.pos), int(agent.state.dir)
            tx, ty = get_view_exts(np.array([d]), np.array([[x, y]]), size)[0]
            fx, fy = x + vec[d][0], y + vec[d][1]
            ahead = field[fx, fy, d] if 0 <= fx < u.grid.width and 0 <= fy < u.grid.height and np.isfinite(field[fx, fy, d]) \
                else field[x, y, d]
            after = [field[x, y, (d - 1) % 4], field[x, y, (d + 1) % 4], ahead]
            first = cost.first_costs.get(a) if cost is not None and hasattr(cost, 'first_costs') else None
            row = live.index(e) if first is not None and len(first) == len(live) else None
            head = [float(v) if np.isfinite(v) else NO_COST for v in first[row].tolist()] if row is not None else [NO_COST] * 3
            flat = cost._flat.get(a) if row is not None and getattr(cost, 'explore', False) else None
            rows.append([int(tx <= gx < tx + size and ty <= gy < ty + size), int(actions[e, a]),
                         *(1 + float(v) for v in after), *head, int(bool(flat[row])) if flat is not None else 0,
                         int(s['seen_at'][a] >= 0) if s is not None else -1, rec._seed if s is not None else -1, a,
                         rec._t if rec is not None else -1])
    return rows

def decision_quality(rows) -> dict | None:
    if not rows:
        return None
    r = np.asarray(rows, float)
    view, chosen, after, head, explored = r[:, 0] > 0, r[:, 1].astype(int), r[:, 2:5], r[:, 5:8], r[:, 8] > 0
    i, best = np.arange(len(r)), r[:, 2:5].min(1)
    ok = np.isfinite(best)
    regret = after[i, chosen] - best
    sampled = (head < NO_COST / 2).all(1)
    pick = head.argmin(1)
    head_regret = after[i, pick] - best
    mean = lambda x: float(x.mean()) if len(x) else None
    out = {}
    groups = [('goal in view', view & ok), ('goal not in view', ~view & ok)]
    if r.shape[1] > 9 and (r[:, 9] >= 0).all():
        groups += [('goal remembered', ~view & (r[:, 9] > 0) & ok), ('goal never seen', ~view & (r[:, 9] == 0) & ok)]
    for name, m in (*groups, ('all', ok)):
        bad = m & (regret > 0)
        out[name] = dict(decisions=int(m.sum()), optimal=mean(regret[m] == 0), regret=mean(regret[m]),
                         cost_optimal=mean(head_regret[m & sampled] == 0),
                         suboptimal_cost_preferred_it=mean(pick[bad & sampled] == chosen[bad & sampled]),
                         suboptimal_cost_ranked_optimal_first=mean(head_regret[bad & sampled] == 0),
                         suboptimal_explored=mean(explored[bad]), explored=mean(explored[m]))
    return out

def deploy(
    make_world: Callable,
    make_policy: Callable,
    methods,
    episodes: int,
    seed: int = 0,
    budget=None,
    decisions: bool = False,
    reseed: bool = False,
    resample_from: int | None = None,
) -> dict:
    from c3_wm.comm.evaluation import network_metrics
    out = {}
    for method in methods:
        world, policy = make_world(method), make_policy(method)
        times, act, decided = [], policy.get_action, []
        cost = getattr(getattr(policy, 'solver', None), 'cost', None)
        measured = cost if cost is not None else getattr(getattr(policy, 'model', None), 'influence', None)
        if getattr(cost, 'novelty', None) == 'episode':
            cost.episodic = lambda _w=world: [find_wrapper(env, LatentMemory).episode for env in _w.envs.envs
                                              if not all(x.state.terminated for x in env.unwrapped.agents)]

        def timed(infos, _act=act, _times=times):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            began = time.perf_counter()
            actions = _act(infos)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _times.append(time.perf_counter() - began)
            if decisions:
                decided.extend(decision_rows(world, cost, actions))
            return actions
        gen = getattr(getattr(policy, 'solver', None), 'torch_gen', None)
        if (reseed or resample_from is not None) and gen is None:
            raise ValueError(f"{method}: reseed / resample_from need a planner (a solver with torch_gen)")
        if resample_from is not None and not reseed:
            raise ValueError("resample_from: with reseed")
        steps, first = [0], [seed]

        def counted(infos, _act=timed):
            if resample_from is not None and steps[0] == resample_from:
                gen.manual_seed(int(np.random.SeedSequence([first[0], 1]).generate_state(1)[0]))
            steps[0] += 1
            return _act(infos)
        policy.get_action = counted
        world.set_policy(policy)
        n, parts, began = world.num_envs, [], time.time()
        for start in range(0, episodes, n):
            if reseed:
                first[0], steps[0] = seed + start, 0
                gen.manual_seed(int(np.random.SeedSequence([seed + start, 0]).generate_state(1)[0]))
            parts.append(world.evaluate(episodes=min(n, episodes - start), seed=seed + start, reset_mode='wait'))
            done, elapsed = start + len(parts[-1]['seeds']), time.time() - began
            success = np.concatenate([p['episode_successes'] for p in parts]).mean()
            agents = (np.concatenate([p['episode_returns'] for p in parts]) > 0).mean()
            length = np.concatenate([p['episode_lengths'] for p in parts]).mean()
            logger.info(f"{method}: {done}/{episodes} episodes, success {100 * success:.1f}% (agents {100 * agents:.1f}%), "
                        f"mean length {length:.0f}; "
                        f"{elapsed / len(parts):.0f} s a round, {elapsed / 60:.1f} min so far, "
                        f"~{elapsed / done * (episodes - done) / 60:.0f} min left")
        result = _merged(parts)
        result['decisions'], result['decision_rows'] = decision_quality(decided), decided or None
        result['landscape'] = measured.landscape() if hasattr(measured, 'landscape') else None
        result['landscape_raw'] = measured.raw() if hasattr(measured, 'raw') else None
        rows = [r for env in world.envs.envs for r in (find_wrapper(env, C3WMCommunication).records or [])]
        result['link'] = network_metrics(rows, budget, MESSAGE_TOKENS, [0.0]) if rows and budget is not None else None
        result['link_rows'], result['budget'] = (rows, asdict(budget)) if rows and budget is not None else (None, None)
        ages = [x for env in world.envs.envs for x in find_wrapper(env, C3WMCommunication).ages]
        result['age'], result['age_rows'] = age_of_information(ages), ages or None
        result['decision'], result['decision_times'] = decision_time(times, world.num_envs), times
        starts = {s: v for env in world.envs.envs if (r := find_wrapper(env, StartRecorder)) is not None
                  for s, v in r.starts.items()}
        if starts and all(int(s) in starts for s in result['seeds']):
            result['starts'] = {k: np.stack([starts[int(s)][k] for s in result['seeds']]) for k in STARTS}
            result['by_start'] = by_start(result['episode_returns'] > 0, result['starts']['distance'], result['starts']['in_view'])
        memories = [m for env in world.envs.envs if (m := find_wrapper(env, LatentMemory)) is not None and m.record]
        errors = {s: (e, m.forgot[s]) for m in memories for s, e in m.errors.items()}
        if errors and all(int(s) in errors for s in result['seeds']):
            result['prediction_error'] = [np.stack(errors[int(s)][0]).tolist() for s in result['seeds']]
            result['forgot'] = [np.stack(errors[int(s)][1]).astype(int).tolist() for s in result['seeds']]
        events = {s: v for env in world.envs.envs if (r := find_wrapper(env, Surprise)) is not None
                  for s, v in r.events.items()}
        if events and all(int(s) in events for s in result['seeds']):
            result['surprise'] = {k: np.stack([np.asarray(events[int(s)][k]) for s in result['seeds']])
                                  for k in ('at', 'size', 'seen_after')}
        beliefs = [b.stats for env in world.envs.envs if (b := find_wrapper(env, BeliefMap)) is not None]
        result['belief'] = {k: sum(s[k] for s in beliefs) for k in beliefs[0]} if beliefs else None
        recorders = [r for env in world.envs.envs if (r := find_wrapper(env, TrajectoryRecorder)) is not None]
        if recorders:
            played, kept = set(int(s) for s in result['seeds']), {}
            for e in (e for r in recorders for e in r.episodes):
                if e['seed'] in played:
                    kept.setdefault(e['seed'], e)
            result['trajectories'] = [kept[s] for s in sorted(kept)]
        world.close()
        out[method] = result
    return out

def merge_results(parts: list) -> dict:
    from c3_wm.comm.evaluation import network_metrics
    from c3_wm.comm.power_control import Budget
    methods = list(parts[0])
    if any(list(p) != methods for p in parts):
        raise ValueError(f"the parts' methods differ: {[list(p) for p in parts]}")
    out = {}
    for method in methods:
        rs = sorted((p[method] for p in parts), key=lambda r: min(r['seeds']))
        seeds = [int(s) for r in rs for s in r['seeds']]
        if len(set(seeds)) < len(seeds):
            raise ValueError(f"{method}: the parts share episodes")
        result = _merged([{k: np.asarray(r[k]) for k in ('episode_successes', 'episode_returns', 'episode_lengths', 'seeds')}
                          for r in rs])
        raws = [r.get('landscape_raw') for r in rs]
        result['landscape_raw'] = {k: [v for x in raws for v in x[k]] for k in raws[0]} if all(raws) else None
        result['landscape'] = C3WMCost.landscape_of(result['landscape_raw']) if all(raws) else None
        rows, budget = [tuple(row) for r in rs for row in (r.get('link_rows') or [])], rs[0].get('budget')
        result['link'] = network_metrics(rows, Budget(**budget), MESSAGE_TOKENS, [0.0]) if rows and budget else None
        result['link_rows'], result['budget'] = (rows, budget) if rows and budget else (None, None)
        ages = [x for r in rs for x in (r.get('age_rows') or [])]
        result['age'], result['age_rows'] = age_of_information(ages), ages or None
        decisions = [r.get('decision') for r in rs]
        times = [t for r in rs for t in (r.get('decision_times') or [])[1:]]
        result['decision'] = decision_time([0.0] + times, decisions[0]['num_envs']) if all(decisions) else None
        result['decision_times'] = times or None
        decided = [row for r in rs for row in (r.get('decision_rows') or [])]
        result['decisions'], result['decision_rows'] = decision_quality(decided), decided or None
        beliefs = [r.get('belief') for r in rs]
        result['belief'] = {k: sum(b[k] for b in beliefs) for k in beliefs[0]} if all(beliefs) else None
        if all(r.get('starts') for r in rs):
            keys = [k for k in STARTS if all(k in r['starts'] for r in rs)]
            result['starts'] = {k: np.concatenate([np.asarray(r['starts'][k]) for r in rs]) for k in keys}
            result['by_start'] = by_start(result['episode_returns'] > 0, result['starts']['distance'],
                                          result['starts']['in_view'].astype(bool))
        for k in ('prediction_error', 'forgot'):
            if all(r.get(k) for r in rs):
                result[k] = [e for r in rs for e in r[k]]
        if all(r.get('surprise') for r in rs):
            result['surprise'] = {k: np.concatenate([np.asarray(r['surprise'][k]) for r in rs]) for k in rs[0]['surprise']}
        out[method] = result
    return out
