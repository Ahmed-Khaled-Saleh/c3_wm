import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import stable_marl as sm
from stable_marl.planning.solver import CategoricalCEMSolver
from stable_marl.policy import FeedForwardPolicy, PlanConfig, RandomPolicy, WorldModelPolicy
from tarmac_baseline import DeliveredLinks, TarMACActor, placeholder_tokens

MESSAGING = ('tarmac', 'commnet', 'mamba', 'codreamer')
SILENT = ('gcrl', 'mappo', 'omnivla', 'dreamer')

from c3_wm.comm.channel_data import PairCache, fading_correlation
from deploy_trace import group_layouts
from c3_wm.comm.evaluation import load_lambda_policies
from c3_wm.comm.power_control import load_power_control
from c3_wm.evaluators.deploy import (ActionSubset, BeliefMap, C3WMCommunication, C3WMCost, GoalMemory, LatentMemory,
                                         StartRecorder, StuckEscape, Surprise, TrajectoryRecorder, c3wm_link, deploy,
                                         jepa_transform, load_head, load_heads, save_trajectories, vq_tokens)
from c3_wm.planners.mpc import DiscreteCEMSolver
from c3_wm.utils.env_utils import LayoutWrapper

HISTORY_KEYS = ('pixels', 'msg_tokens', 'msg_received', 'msg_age', 'msg_pose')


class SampledCEMSolver(CategoricalCEMSolver):

    def __init__(self, cost, temperature: float = 0.0, prior_seed: float = 0.0, **kwargs):
        super().__init__(cost, **kwargs)
        if temperature < 0 or not 0 <= prior_seed <= 1:
            raise ValueError(f"temperature: 0 or above; prior_seed: in [0, 1]; not {temperature}, {prior_seed}")
        if prior_seed and getattr(cost, 'prior_head', None) is None:
            raise ValueError("prior_seed seeds the plans with the prior: with checkpoints.prior")
        if self.batch_size is not None:
            raise ValueError("SampledCEMSolver plans every env at once (batch_size=None): the costs' first_best is per call")
        self.temperature, self.prior_seed, self._seed = temperature, prior_seed, None

    def init_probs(self, n_envs: int) -> torch.Tensor:
        probs = super().init_probs(n_envs)
        if self._seed is not None:
            probs[:, 0, :, 0] = (1 - self.prior_seed) * probs[:, 0, :, 0] + self.prior_seed * self._seed.to(probs)
        return probs

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action=None) -> dict:
        self.cost.first_best = {}
        n_envs = len(next(v for v in info_dict.values() if isinstance(v, (torch.Tensor, np.ndarray, list))))
        self._seed = self.cost.prior_log(self._expand(info_dict, 0, n_envs, num_samples=1)).exp() if self.prior_seed else None
        out = super().solve(info_dict, init_action)
        if not self.temperature:
            return out
        mode = out['actions'][:, 0, :, 0]
        drawn = mode.clone()
        for a, best in self.cost.first_best.items():
            best = best.to(self.device).float()
            logits = -(best - best.min(-1, keepdim=True).values) / self.temperature
            u = torch.rand(logits.shape, generator=self.torch_gen, device=self.device).clamp_min(1e-10)
            pick = (logits - (-u.log()).log()).argmax(-1).cpu()
            flat = self.cost._flat.get(a) if getattr(self.cost, 'explore', False) else None
            drawn[:, a] = pick if flat is None else torch.where(flat.cpu(), mode[:, a], pick)
        self.cost.resampled += (drawn != mode).flatten().tolist()
        out['actions'][:, 0, :, 0] = drawn
        return out

def load_models(cfg, device):
    jepa = hydra.utils.instantiate({k: v for k, v in OmegaConf.to_container(cfg.jepa, resolve=True).items()
                                    if k != 'checkpoint_path'})
    state = torch.load(to_absolute_path(cfg.checkpoints.jepa), map_location='cpu')['wm_model_state_dict']
    if cfg.checkpoints.jepa_strict:
        jepa.load_state_dict(state)
    else:
        keys = jepa.load_state_dict(state, strict=False)
        logger.warning(f"world model loaded non-strictly: missing {keys.missing_keys} (left as initialized), "
                       f"unexpected {keys.unexpected_keys} (ignored)")
    vqvae = hydra.utils.instantiate({**OmegaConf.to_container(cfg.vqvae, resolve=True)})
    vqvae.load_state_dict(torch.load(to_absolute_path(cfg.checkpoints.vqvae), map_location='cpu')['model_state_dict'])
    return jepa.to(device).eval(), vqvae.to(device).eval()


def recolor(env, colors):
    for i, agent in enumerate(env.unwrapped.agents):
        agent.state.color = str(colors[i % len(colors)])
    return env


def variation_values(cfg) -> dict:
    from stable_marl.envs.multigrid.core.constants import Color
    vary = cfg.vary
    named = {vary.wall_color, vary.goal_color, *(cfg.agent_colors or [])}
    for name, rgb in cfg.new_colors.items():
        if name in named and name not in Color.__members__:
            Color.add_color(name, list(rgb))
    factors = {'wall.color': vary.wall_color, 'goal.color': vary.goal_color, 'obstacles.number': vary.obstacles,
               'clutter.number': vary.clutter}
    return {k: Color(str(v)).to_index() if k.endswith('color') else int(v) for k, v in factors.items() if v is not None}


def rho(cfg) -> float:
    link = cfg.link
    return float(link.fading_correlation) if link.fading_correlation is not None else fading_correlation(link.step_seconds,
                                                                                                         link.doppler_hz)


@hydra.main(version_base=None, config_path="../configs", config_name="deploy/config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    ck = cfg.checkpoints
    empty = [k for k in ck if not OmegaConf.is_missing(ck, k) and isinstance(ck[k], str) and not ck[k].strip()]
    if empty:
        raise ValueError(f"checkpoints {empty} given as an empty path (an unset shell variable?): set them, or null")
    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("device=cuda, but no GPU is visible")
    cfg.baseline_link = str(cfg.baseline_link).replace('_', ' ')
    if cfg.baseline_sample:
        torch.manual_seed(cfg.seed)
    jepa, vqvae = load_models(cfg, device)
    env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    world_kwargs = {k: v for k, v in env_cfg.items() if k not in ('_target_', 'variations')}
    world_kwargs['max_steps'] = cfg.max_steps
    layouts, edits = list(range(*cfg.layouts)), {}
    if cfg.layout_group.file:
        layouts, edits = group_layouts(to_absolute_path(cfg.layout_group.file), cfg.layout_group.name)
    recorded = str(cfg.record.method).replace('-', ' ').replace('_', ' ')
    methods = [recorded] if cfg.record.dir else list(cfg.methods)
    radio = {cfg.baseline_link if m in MESSAGING else m for m in methods}
    over_link = {'learned', 'perfect CSI', 'lambda-policy', 'fixed power'} & radio
    values = variation_values(cfg)
    if values:
        logger.info(f"every layout changed: {values}")
    surprise = OmegaConf.to_container(cfg.surprise, resolve=True) if cfg.surprise.kind else None
    withhold = None if cfg.withhold_from is None else (int(cfg.withhold_from), cfg.withhold_steps)
    if surprise:
        logger.info(f"a surprise in every episode: {surprise}")
    if over_link and ({'obstacles.number', 'clutter.number'} & values.keys() or cfg.surprise.kind == 'wall'):
        raise ValueError(f"vary's obstacles or clutter, or a wall surprise: new walls, but the link's channel is traced in "
                         f"the unchanged layouts ({sorted(over_link)}): the perfect channel or no communication")
    saved = load_power_control(to_absolute_path(cfg.checkpoints.power_control)) if over_link else None
    policies = None
    if 'lambda-policy' in radio:
        path = cfg.checkpoints.lambda_policy or Path(to_absolute_path(cfg.checkpoints.power_control)).parent / 'lambda_policy.pkl'
        policies = load_lambda_policies(to_absolute_path(str(path)))
    trace = OmegaConf.to_container(cfg.tracing, resolve=True)
    caches = {}

    def cache_for(layout):
        if layout not in caches:
            caches[layout] = PairCache(Path(cfg.data_dir) / 'layouts' / f'pairs_{layout}.npz', env_cfg, layout,
                                       trace['frequency'], trace['max_depth'], trace['scene'], edits=edits,
                                       **trace['solver'])
        return caches[layout]

    tokens = vq_tokens(vqvae, device)
    tile_codes = None
    if cfg.planner.goal_cost and cfg.planner.goal_source == 'belief':
        if not cfg.checkpoints.tile_codes:
            raise ValueError("planner.goal_source=belief reads the tokens with checkpoints.tile_codes (tile_codes.py's)")
        codes = np.load(to_absolute_path(cfg.checkpoints.tile_codes))
        from c3_wm.trainers.wm_tune import codebook_id
        if str(codes['codebook']) != codebook_id(vqvae):
            raise ValueError(f"checkpoints.tile_codes reads codebook {codes['codebook']}, not this VQ-VAE's {codebook_id(vqvae)}")
        tile_codes = codes['table']
    elif cfg.planner.goal_source not in ('oracle', 'belief'):
        raise ValueError(f"planner.goal_source: oracle or belief, not {cfg.planner.goal_source!r}")
    wall_head = load_head(to_absolute_path(cfg.checkpoints.wall_head)).to(device) if cfg.checkpoints.wall_head else None
    if wall_head is not None and wall_head.kind != 'tiles':
        raise ValueError(f"checkpoints.wall_head: a tiles head (goal_head.py head.kind=tiles), not {wall_head.kind}")
    memory_head = load_head(to_absolute_path(cfg.checkpoints.memory_cost)).to(device) if cfg.checkpoints.memory_cost else None
    if memory_head is not None and memory_head.kind != 'memory_cost':
        raise ValueError(f"checkpoints.memory_cost: memory_cost.py's head, not a {memory_head.kind} one")
    if memory_head is not None and (cfg.planner.goal_cost or cfg.checkpoints.goal_head):
        raise ValueError("checkpoints.memory_cost is the planner's cost: not with planner.goal_cost or checkpoints.goal_head")
    prior_head = load_head(to_absolute_path(cfg.checkpoints.prior)).to(device) if cfg.checkpoints.prior else None
    if prior_head is not None:
        if prior_head.kind not in ('policy_prior', 'recurrent_prior') or memory_head is None:
            raise ValueError("checkpoints.prior: policy_prior.py's head, with checkpoints.memory_cost (the planner's cost)")
        trained_on = torch.load(to_absolute_path(cfg.checkpoints.prior), map_location='cpu')['checkpoints'].get('memory_cost')
        if trained_on and Path(trained_on).parent.name != Path(cfg.checkpoints.memory_cost).parent.name:
            logger.warning(f"the prior was trained on the memory of {trained_on}, deployed with {cfg.checkpoints.memory_cost}")
    if cfg.planner.escape_window and not cfg.planner.explore:
        raise ValueError("planner.escape_window escapes by exploring's novelty: with planner.explore=true")
    heads = cfg.checkpoints.goal_head
    goal_head = load_heads(to_absolute_path(heads) if isinstance(heads, str) else [to_absolute_path(h) for h in heads]
                           ).to(device) if heads else None

    def make_world(method):
        radio = cfg.baseline_link if method in MESSAGING else method
        silent = radio in ('no communication', 'random', *SILENT)
        link = None if radio in ('no communication', 'random', 'perfect channel', *SILENT) else \
            c3wm_link(radio, cache_for, saved, crc=cfg.link.crc, rho=rho(cfg), rician=cfg.link.rician, policies=policies,
                      policy=cfg.link.lambda_policy.name, target=cfg.link.lambda_policy.target)
        packets = placeholder_tokens if method in MESSAGING else tokens
        record = [TrajectoryRecorder] if cfg.record.dir and method == recorded else []
        heard = cfg.planner.goal_from == 'messages'
        oracle = [] if not cfg.planner.goal_cost else \
            [lambda e: BeliefMap(e, tokens, tile_codes, messages=heard)] if cfg.planner.goal_source == 'belief' else \
            [lambda e: GoalMemory(e, messages=heard)]
        if memory_head is not None and method not in (*SILENT, *MESSAGING):
            oracle = [lambda e: LatentMemory(e, jepa, memory_head, messages=heard, device=device, tokens=tokens,
                                             counterfactuals=cfg.planner.cic, seed=cfg.seed,
                                             record_surprise=cfg.planner.record_surprise,
                                             forget_above=cfg.planner.forget_above, forget_keep=cfg.planner.forget_keep,
                                             history=cfg.pipeline.history_size)]
            if prior_head is not None and prior_head.kind == 'recurrent_prior':
                oracle.append(lambda e: LatentMemory(e, jepa, prior_head, messages=heard, device=device, tokens=tokens,
                                                     counterfactuals=cfg.planner.cic, seed=cfg.seed, key='prior_memory'))
            if cfg.planner.escape_window:
                oracle.append(lambda e: StuckEscape(e, cfg.planner.escape_window, cfg.planner.escape_cells))
        return sm.World(cfg.env_id, num_envs=cfg.num_envs, goal_conditioned=True, **world_kwargs,
                        pre_wrappers=[*([lambda e: recolor(e, list(cfg.agent_colors))] if cfg.agent_colors else []),
                                      lambda e: LayoutWrapper(e, layouts, variations=tuple(env_cfg.get('variations') or ()),
                                                              max_start_distance=cfg.max_start_distance, edits=edits,
                                                              values=values),
                                      *([lambda e: Surprise(e, **surprise)] if surprise else []),
                                      StartRecorder, *record,
                                      lambda e: C3WMCommunication(e, None if silent else packets, link,
                                                                  max_age=jepa.predictor.max_age, delivery=cfg.delivery,
                                                                  withhold=withhold),
                                      *oracle, lambda e: ActionSubset(e, cfg.actions)])

    def make_policy(method):
        if method == 'random':
            return RandomPolicy(seed=cfg.seed)
        temperature = 1.0 if cfg.baseline_sample else 0.0
        if method == 'gcrl':
            from gcrl_baseline import GCRLActor
            from stable_marl.wm.gcrl import IndependentGCRL
            if not cfg.checkpoints.gcrl:
                raise ValueError("method gcrl acts with checkpoints.gcrl (gcrl_baseline.py's gcrl.pkl)")
            path = Path(to_absolute_path(cfg.checkpoints.gcrl))
            trained = json.loads((path.parent / 'history.json').read_text())
            model = IndependentGCRL.load(path, seed=cfg.seed)
            model.temperature = temperature
            if trained.get('features', 'pixels') == 'vit':
                if trained.get('jepa') and Path(trained['jepa']).name != Path(cfg.checkpoints.jepa).name:
                    logger.warning(f"the GCRL agents were trained on the features of {trained['jepa']}, deployed with {cfg.checkpoints.jepa}")
                return FeedForwardPolicy(GCRLActor(model, jepa=jepa))
            return FeedForwardPolicy(GCRLActor(model, trained['image_size']))
        if method == 'mappo':
            from mappo_baseline import MAPPOActor
            from stable_marl.wm.mappo import MAPPO
            if not cfg.checkpoints.mappo:
                raise ValueError("method mappo acts with checkpoints.mappo (mappo_baseline.py's mappo.pt)")
            model = MAPPO.load(to_absolute_path(cfg.checkpoints.mappo), map_location=device).to(device)
            model.temperature = temperature
            trained_on = model.checkpoint.get('jepa')
            if trained_on and Path(trained_on).name != Path(cfg.checkpoints.jepa).name:
                logger.warning(f"MAPPO was trained on the features of {trained_on}, deployed with {cfg.checkpoints.jepa}")
            return FeedForwardPolicy(MAPPOActor(model, jepa))
        if method == 'tarmac':
            from stable_marl.wm.tarmac import TarMAC
            if not cfg.checkpoints.tarmac:
                raise ValueError("method tarmac acts with checkpoints.tarmac (tarmac_baseline.py's tarmac.pt)")
            model = TarMAC.load(to_absolute_path(cfg.checkpoints.tarmac), map_location=device).to(device)
            model.temperature = temperature
            model.drop_prob = 0.0
            trained_on = model.checkpoint.get('jepa')
            if trained_on and Path(trained_on).name != Path(cfg.checkpoints.jepa).name:
                logger.warning(f"TarMAC was trained on the features of {trained_on}, deployed with {cfg.checkpoints.jepa}")
            model.links = DeliveredLinks()
            return FeedForwardPolicy(TarMACActor(model, jepa, model.links, cic=cfg.planner.cic, seed=cfg.seed))
        if method == 'commnet':
            from commnet_baseline import CommNetActor, ViTCommNet
            if not cfg.checkpoints.commnet:
                raise ValueError("method commnet acts with checkpoints.commnet (commnet_baseline.py's commnet.pt)")
            model = ViTCommNet.load(to_absolute_path(cfg.checkpoints.commnet), map_location=device).to(device)
            model.drop_prob = 0.0
            trained_on = model.checkpoint.get('jepa')
            if trained_on and Path(trained_on).name != Path(cfg.checkpoints.jepa).name:
                logger.warning(f"CommNet was trained on the features of {trained_on}, deployed with {cfg.checkpoints.jepa}")
            return FeedForwardPolicy(CommNetActor(model, jepa, sample=cfg.baseline_sample, cic=cfg.planner.cic, seed=cfg.seed))
        if method == 'omnivla':
            from omnivla_baseline import OmniVLAActor, ViTOmniVLA, omnivla_from
            if not cfg.checkpoints.omnivla:
                raise ValueError("method omnivla acts with checkpoints.omnivla (omnivla_baseline.py's omnivla.pt)")
            model = omnivla_from(to_absolute_path(cfg.checkpoints.omnivla), map_location=device).to(device).eval()
            model.temperature = temperature
            return FeedForwardPolicy(OmniVLAActor(model, jepa if isinstance(model, ViTOmniVLA) else None))
        if method == 'mamba':
            from mamba_baseline import mamba_policy
            from stable_marl.wm.mamba import Mamba
            if not cfg.checkpoints.mamba:
                raise ValueError("method mamba acts with checkpoints.mamba (mamba_baseline.py's mamba.pt)")
            mamba = Mamba.load(to_absolute_path(cfg.checkpoints.mamba))
            trained_on = mamba.checkpoint.get('jepa')
            if trained_on and Path(trained_on).name != Path(cfg.checkpoints.jepa).name:
                logger.warning(f"MAMBA was trained on the features of {trained_on}, deployed with {cfg.checkpoints.jepa}")
            return mamba_policy(mamba, jepa, sample=cfg.baseline_sample)
        if method in ('dreamer', 'codreamer'):
            from dreamer_baseline import dreamer_actor
            if not cfg.checkpoints[method]:
                raise ValueError(f"method {method} acts with checkpoints.{method} (dreamer_baseline.py's {method}.pt)")
            return FeedForwardPolicy(dreamer_actor(method, to_absolute_path(cfg.checkpoints[method]), jepa,
                                                   str(cfg.checkpoints.jepa), device, sample=cfg.baseline_sample,
                                                   cic=cfg.planner.cic, seed=cfg.seed))
        p = cfg.planner
        cost = C3WMCost(jepa, cfg.pipeline.history_size, messages=p.predictor_messages, cost=p.cost, every=p.n_steps,
                        explore=p.explore, explore_below=p.explore_below, message_effect=p.message_effect,
                        goal_cost=p.goal_cost, goal_head=goal_head, wall_head=wall_head, wall_above=p.wall_above,
                        memory_head=memory_head, move_messages=p.move_messages, step_cost=p.step_cost,
                        near_step_cost=p.near_step_cost, near_below=p.near_below,
                        novelty=p.novelty, explore_within=p.explore_within, imagine_noise=p.imagine_noise,
                        cic=bool(p.cic), prior_head=prior_head, prior_weight=p.prior_weight)
        if p.solver == 'stable_marl' and (p.temperature or p.prior_seed):
            solver = SampledCEMSolver(cost, temperature=p.temperature, prior_seed=p.prior_seed, batch_size=None,
                                      num_samples=p.num_samples, n_steps=p.n_steps, topk=p.topk, mode='per_agent',
                                      device=device, seed=cfg.seed)
        elif p.solver == 'stable_marl':
            solver = CategoricalCEMSolver(cost, batch_size=None, num_samples=p.num_samples, n_steps=p.n_steps, topk=p.topk,
                                          mode='per_agent', device=device, seed=cfg.seed)
        elif p.solver == 'mpc' and (p.temperature or p.prior_seed):
            raise ValueError("planner.temperature, planner.prior_seed: with planner.solver=stable_marl only")
        elif p.solver == 'mpc':
            solver = DiscreteCEMSolver(cost, pop_size=p.num_samples, topk=p.topk, opt_steps=p.n_steps, mode='per_agent',
                                       device=device, seed=cfg.seed)
        else:
            raise ValueError(f"planner.solver: stable_marl or mpc, not {p.solver!r}")
        return WorldModelPolicy(solver, PlanConfig(horizon=p.horizon, receding_horizon=p.receding_horizon,
                                                   history_len=cfg.pipeline.history_size),
                                transform={'pixels': jepa_transform, 'goal': jepa_transform}, history_keys=HISTORY_KEYS)

    results = deploy(make_world, make_policy, methods, cfg.episodes, cfg.seed,
                     saved['budget'] if saved is not None else None, decisions=cfg.diagnose_decisions,
                     reseed=cfg.planner.reseed, resample_from=cfg.planner.resample_from)
    if cfg.label:
        results = {f"{m} ({cfg.label})": r for m, r in results.items()}
    for method, r in results.items():
        if 'trajectories' in r:
            episodes = r.pop('trajectories')
            paths = save_trajectories(episodes, to_absolute_path(cfg.record.dir))
            logger.info(f"{method}: {len(episodes)} episodes' trajectories in {len(paths)} layouts, {to_absolute_path(cfg.record.dir)}")
    for cache in caches.values():
        if cache.unsaved:
            cache.save()

    out = Path(to_absolute_path(cfg.out_dir))
    save_results(results, OmegaConf.to_container(cfg, resolve=True), out)
    report(results, OmegaConf.to_container(cfg, resolve=True), out)


def save_results(results: dict, config: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), out / 'config.yaml')
    plain = lambda v: v.tolist() if isinstance(v, np.ndarray) else {k: plain(x) for k, x in v.items()} if isinstance(v, dict) else v
    (out / 'results.json').write_text(json.dumps({m: {k: plain(v) for k, v in r.items()} for m, r in results.items()},
                                                 indent=1, default=float))


def report(results: dict, config: dict, out: Path):
    rows = []
    for method, r in results.items():
        link, land, b = r['link'] or {}, r.get('landscape') or {}, r.get('belief') or {}
        belief = {'belief_goal_right': b['goal_right'] / b['goal_steps'] if b.get('goal_steps') else None,
                  'belief_wall_precision': b['walls_right'] / b['walls'] if b.get('walls') else None} if b else {}
        rows.append(dict(method=method, success_rate=r['success_rate'], agent_success_rate=r['agent_success_rate'],
                         any_success_rate=r['any_success_rate'], mean_length=r['mean_length'],
                         mean_return=r['mean_return'], **{f'link_{k}': v for k, v in link.items()}, **belief,
                         **{f'planner_{k}': v for k, v in land.items()},
                         **{f'success_{k}': g['success'] for k, g in (r.get('by_start') or {}).items()}))
        logger.info(f"{method:18s} success {r['success_rate']:5.1f}%  agents {r['agent_success_rate']:5.1f}%  "
                    f"length {r['mean_length']:6.1f}"
                    + (f"  first-action gap {land['first_action_gap']:.3f} (ties {100 * land['first_action_ties']:.0f}%)"
                       if land.get('first_action_gap') is not None else '')
                    + (f"  explored {100 * land['explored']:.0f}%" if land.get('explored') is not None else '')
                    + (f"  heard {100 * land['heard']:.0f}%, messages change the first action {100 * land['message_changes']:.0f}%"
                       if land.get('message_changes') is not None else '')
                    + (f"  goal known {100 * land['goal_known']:.0f}%" if land.get('goal_known') is not None else '')
                    + (f"  CIC {land['cic']:.3f} bits (silent changes {100 * land['memory_silent_changes']:.0f}%, other content {100 * land['content_changes']:.0f}%)"
                       if land.get('cic') is not None else '')
                    + (f"  believed goal right {100 * belief['belief_goal_right']:.0f}%, walls {100 * belief['belief_wall_precision']:.0f}%"
                       if belief.get('belief_goal_right') is not None and belief.get('belief_wall_precision') is not None else '')
                    + (f"  reached {100 * link['reached']:.1f}%  power {link['power']:.2f}×  withheld {100 * link['withheld']:.1f}%"
                       if link else ''))
        if r.get('by_start'):
            logger.info(f"{'':18s} by start: " + ', '.join(f"{k} {g['success']:.0f}% ({g['agents']})" if g['success'] is not None
                                                          else f"{k} - (0)" for k, g in r['by_start'].items()))
    w = config['wandb']
    if w['enabled'] and w['mode'] != 'disabled':
        import wandb
        Path(config['paths']['wandb']).mkdir(parents=True, exist_ok=True)
        run = wandb.init(project=w['project'], name=w['name'] or f"deploy-{os.getenv('SLURM_JOB_ID', 'local')}",
                         mode=w['mode'], dir=config['paths']['wandb'], config=config)
        columns = sorted({k for row in rows for k in row}, key=lambda k: (k != 'method', k))
        run.log({'deploy/methods': wandb.Table(columns=columns, data=[[row.get(c) for c in columns] for row in rows])})
        run.finish()
    logger.info(f"results in {out / 'results.json'}")


if __name__ == "__main__":
    main()
