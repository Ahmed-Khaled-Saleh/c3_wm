from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space


__all__ = ['BEHAVIOURS', 'EnvPool', 'MultiAgentEnvPool', 'set_env_state', 'goal_moves', 'LayoutWrapper']


class EnvPool:

    def __init__(self, env_fns: list):
        self.envs = [fn() for fn in env_fns]
        self._single_env = self.envs[0]
        self._stacked_infos: dict[str, Any] | None = None
        self.seeds = np.zeros(len(self.envs), dtype=np.int64)
        self._action_space = batch_space(
            self._single_env.action_space, len(self.envs)
        )
        self._observation_space = batch_space(
            self._single_env.observation_space, len(self.envs)
        )

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    @property
    def single_action_space(self) -> gym.Space:
        return self._single_env.action_space

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def single_observation_space(self) -> gym.Space:
        return self._single_env.observation_space

    @property
    def variation_space(self):
        return getattr(self._single_env.unwrapped, 'variation_space', None)

    @property
    def single_variation_space(self):
        return self.variation_space

    def reset(
        self,
        seed: int | list[int | None] | None = None,
        options: dict | list[dict | None] | None = None,
        mask: np.ndarray | None = None,
    ) -> tuple[None, dict]:
        seeds = _broadcast_arg(seed, self.num_envs, increment=True)
        opts = _broadcast_arg(options, self.num_envs)

        per_env_infos = [None] * self.num_envs
        for i, env in enumerate(self.envs):
            if mask is not None and not mask[i]:
                continue
            _, per_env_infos[i] = env.reset(seed=seeds[i], options=opts[i])
            if seeds[i] is not None:
                self.seeds[i] = seeds[i]

        if self._stacked_infos is None or mask is None:
            self._stacked_infos = _stack_fresh(per_env_infos)
        else:
            for i, info in enumerate(per_env_infos):
                if info is not None:
                    _write_env_info(self._stacked_infos, i, info)

        return None, self._stacked_infos

    def step(
        self, actions: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[None, np.ndarray, np.ndarray, np.ndarray, dict]:
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminateds = np.zeros(self.num_envs, dtype=bool)
        truncateds = np.zeros(self.num_envs, dtype=bool)

        for i, env in enumerate(self.envs):
            if mask is not None and not mask[i]:
                continue
            _, rewards[i], terminateds[i], truncateds[i], info = env.step(
                actions[i]
            )
            _write_env_info(self._stacked_infos, i, info)

        return None, rewards, terminateds, truncateds, self._stacked_infos

    def close(self):
        for env in self.envs:
            env.close()


def _broadcast_arg(arg, n: int, increment: bool = False) -> list:
    if arg is None:
        return [None] * n
    if isinstance(arg, list):
        return arg
    if isinstance(arg, np.ndarray):
        return list(arg)
    if increment and isinstance(arg, int):
        return [arg + i for i in range(n)]
    return [arg] * n


def _stack_fresh(per_env_infos: list[dict]) -> dict[str, Any]:
    keys = per_env_infos[0].keys()
    stacked = {}
    for k in keys:
        vals = [info[k] for info in per_env_infos]
        first = vals[0]
        if isinstance(first, torch.Tensor):
            stacked[k] = torch.stack(vals).unsqueeze(1)
        elif isinstance(first, np.ndarray):
            stacked[k] = np.stack(vals)[:, None, ...]
        elif isinstance(first, (bool, int, float, np.number)):
            stacked[k] = np.array(vals)[:, None]
        else:
            stacked[k] = [[v] for v in vals]
    return stacked


def _write_env_info(stacked: dict, idx: int, info: dict) -> None:
    for k, v in info.items():
        if k not in stacked:
            continue
        buf = stacked[k]
        if isinstance(buf, torch.Tensor):
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v, dtype=buf.dtype, device=buf.device)
            buf[idx, 0] = v
        elif isinstance(buf, np.ndarray):
            buf[idx, 0] = v
        elif isinstance(buf, list):
            buf[idx][0] = v


class MultiAgentEnvPool:

    def __init__(self, env_fns: list[Callable[[], Any]], agents: list | None = None):
        self.envs = [fn() for fn in env_fns]
        self._single_env = self.envs[0]
        self.agents = list(agents) if agents is not None else list(self._single_env.possible_agents)

        self._stacked_infos: dict[Any, dict[str, Any]] | None = None
        self.seeds = np.zeros(len(self.envs), dtype=np.int64)

        self._action_spaces = {
            a: batch_space(self._single_env.action_space(a), self.num_envs)
            for a in self.agents
        }
        self._observation_spaces = {
            a: batch_space(self._single_env.observation_space(a), self.num_envs)
            for a in self.agents
        }

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    def action_space(self, agent):
        return self._action_spaces[agent]

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def single_action_space(self, agent):
        return self._single_env.action_space(agent)

    def single_observation_space(self, agent):
        return self._single_env.observation_space(agent)

    def reset(
        self,
        seed: int | list[int | None] | None = None,
        options: dict | list[dict | None] | None = None,
        mask: np.ndarray | None = None,
    ) -> tuple[None, dict]:
        seeds = _broadcast_arg(seed, self.num_envs, increment=True)
        opts = _broadcast_arg(options, self.num_envs)

        per_agent_infos = {a: [None] * self.num_envs for a in self.agents}
        for i, env in enumerate(self.envs):
            if mask is not None and not mask[i]:
                continue
            obs, infos = env.reset(seed=seeds[i], options=opts[i])
            if seeds[i] is not None:
                self.seeds[i] = seeds[i]
            for a in self.agents:
                if a in obs:
                    per_agent_infos[a][i] = {**obs[a], **infos.get(a, {})}

        if self._stacked_infos is None or mask is None:
            self._stacked_infos = {
                a: _stack_fresh([info or {} for info in per_agent_infos[a]])
                for a in self.agents
            }
        else:
            for a in self.agents:
                for i, info in enumerate(per_agent_infos[a]):
                    if info is not None:
                        _write_env_info(self._stacked_infos[a], i, info)

        return None, self._stacked_infos

    def step(
        self, actions: dict[Any, np.ndarray], mask: np.ndarray | None = None,
        noop_action: Any = 3,
    ) -> tuple[None, dict, dict, dict, dict]:
        rewards = {a: np.zeros(self.num_envs, dtype=np.float32) for a in self.agents}
        terminateds = {a: np.zeros(self.num_envs, dtype=bool) for a in self.agents}
        truncateds = {a: np.zeros(self.num_envs, dtype=bool) for a in self.agents}

        for i, env in enumerate(self.envs):
            if mask is not None and not mask[i]:
                continue

            live_agents = set(env.agents)
            step_actions = {
                a: (actions[a][i] if a in live_agents else noop_action)
                for a in self.agents if a in actions
            }

            obs, r, term, trunc, infos = env.step(step_actions)

            for a in self.agents:
                if a not in obs:
                    continue
                rewards[a][i] = r.get(a, 0.0)
                terminateds[a][i] = term.get(a, False)
                truncateds[a][i] = trunc.get(a, False)
                info = {**obs[a], **infos.get(a, {})}
                _write_env_info(self._stacked_infos[a], i, info)

        return None, rewards, terminateds, truncateds, self._stacked_infos

    def close(self) -> None:
        for env in self.envs:
            env.close()


def set_env_state(pool, env_idx, goal_pos, agent_positions, agent_directions, agent_ids):
    unwrapped = pool.envs[env_idx].env.unwrapped
    goal_pos = tuple(int(v) for v in goal_pos)

    old_goal_pos = tuple(unwrapped.goal_pos)
    if old_goal_pos != goal_pos:
        goal_obj = unwrapped.grid.get(*old_goal_pos)
        assert unwrapped.grid.get(*goal_pos) is None, \
            f"env {env_idx}: recorded goal_pos {goal_pos} is occupied — layout mismatch with recorded episode"
        unwrapped.grid.set(*old_goal_pos, None)
        unwrapped.grid.set(*goal_pos, goal_obj)
        goal_obj.cur_pos = goal_pos
        unwrapped.goal_pos = goal_pos

    for agent_id in agent_ids:
        agent = unwrapped.agents[agent_id]
        pos = tuple(int(v) for v in agent_positions[agent_id])
        cell = unwrapped.grid.get(*pos)
        assert cell is None or cell.can_overlap(), \
            f"env {env_idx}: agent {agent_id} recorded pos {pos} is inside an object — layout mismatch"
        agent.state.pos = pos
        agent.state.dir = int(agent_directions[agent_id])

import gymnasium as gym
import numpy as np

BEHAVIOURS = ('random', 'epsilon_expert', 'waypoint')


def goal_moves(u) -> dict:
    from collections import deque
    goal = tuple(int(v) for v in u.goal_pos)
    free = lambda x, y: 0 <= x < u.grid.width and 0 <= y < u.grid.height and \
        ((o := u.grid.get(x, y)) is None or o.can_overlap())
    moves, queue = {goal: 0}, deque([goal])
    while queue:
        x, y = cell = queue.popleft()
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nxt not in moves and free(*nxt):
                moves[nxt] = moves[cell] + 1
                queue.append(nxt)
    return moves

class LayoutWrapper(gym.Wrapper):

    def __init__(self, env, layouts, policy=None, variations=(), max_start_distance=None, edits=None, values=None):
        super().__init__(env)
        self.layouts, self.policy, self.layout = list(layouts), policy, None
        self.max_start_distance, self.edits, self.values = max_start_distance, edits or {}, dict(values or {})
        if variations:
            env.unwrapped.DEFAULT_VARIATIONS = tuple(variations)

    def reset(self, *, seed=None, options=None):
        rng = np.random.default_rng(None if seed is None else [int(seed), 7])
        self.layout = int(self.layouts[int(seed) % len(self.layouts)] if seed is not None else rng.choice(self.layouts))
        if self.values:
            options = {**(options or {}), 'variation_values': self.values}
        if self.layout in self.edits:
            from c3_wm.comm.channel_data import reset_layout
            reset_layout(self.env, self.layout, self.edits, options=options)
        else:
            self.env.reset(seed=self.layout, options=options)
        u = self.env.unwrapped
        self._place(u, rng, self.max_start_distance)
        u._goal_infos = u.goal_infos()
        return u.gen_obs(), self._tag(u._infos())

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return obs, rewards, terminations, truncations, self._tag(infos)

    def _tag(self, infos):
        u = self.env.unwrapped
        name = self.policy.behaviour(u) if self.policy is not None else None
        tags = {'layout': np.int64(self.layout), 'behaviour': np.int64(BEHAVIOURS.index(name) if name else -1)}
        return {a: {**infos.get(a, {}), **tags} for a in range(u.num_agents)}

    @staticmethod
    def _place(u, rng, max_moves=None):
        from stable_marl.envs.multigrid.core.constants import Color
        from stable_marl.envs.multigrid.core.world_object import Goal
        u.grid.set(*(int(v) for v in u.goal_pos), None)
        free = [(x, y) for x in range(u.grid.width) for y in range(u.grid.height)
                if (o := u.grid.get(x, y)) is None or o.can_overlap()]
        goal = free[rng.integers(len(free))]
        u.grid.set(*goal, Goal(Color.from_index(int(u._variation('goal.color')))))
        u.goal_pos = np.asarray(goal)
        far = lambda c: abs(c[0] - goal[0]) + abs(c[1] - goal[1])
        cells = [c for c in free if c != goal]
        reach = min(int(u._variation('goal.min_spawn_distance')), max(far(c) for c in cells))
        starts = [c for c in cells if far(c) >= reach]
        if max_moves is not None:
            moves = goal_moves(u)
            near = [c for c in starts if 0 < moves.get(c, -1) <= max_moves]
            starts = near if len(near) >= len(u.agents) else \
                sorted((c for c in cells if c in moves), key=moves.get)[:len(u.agents)]
        for agent, k in zip(u.agents, rng.choice(len(starts), len(u.agents), replace=False)):
            agent.state.pos = starts[k]
            agent.state.dir = int(rng.integers(4))
