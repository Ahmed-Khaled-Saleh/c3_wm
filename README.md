# C3-WM

Code for C3-WM, a communication–control co-adaptive world model for multi-agent planning over wireless links.
Agents encode their views with a JEPA world model, exchange VQ-VAE tokens of their observations, and plan with
CEM in the latent space. Messages are sent over a ray-traced wireless channel, and the transmit power is set by
the λ-policy.

## Installation

Python 3.10 or later and a CUDA GPU are needed. Ray tracing uses Sionna RT on the GPU.

```bash
git clone https://github.com/Ahmed-Khaled-Saleh/c3_wm.git
cd c3_wm
pip install -e .
pip install -e ".[baselines]"
```

The environment (FindGoal) and the communication link come from two companion packages, `stable-marl` and
`comm-core`, which are installed from GitHub as dependencies. The ViT encoder comes from `stable-pretraining`.

Results are logged to Weights & Biases. Set `WANDB_API_KEY` in the environment or in a `.env` file at the
repository root, or pass `wandb.mode=disabled` (or `wandb.enabled=false`) to any script.

## Layout

```
c3_wm/        the package: models, data, trainers, planner, deployment, power control
configs/      Hydra configs, one folder per entry point
mains/        entry points
```

All commands below are run from `mains/`.

## Data and output locations

Everything is written under one root folder, set with the `C3WM_ROOT` environment variable (default
`$HOME/c3_wm`), or with `paths.root=<dir>` on the command line:

```
<root>/datasets/c3_wm/findgoal/<dataset>/
<root>/experiments/c3_wm/findgoal/<phase>/<date>_<time>_job_<id>/{hydra,wandb,checkpoints,media}
```

To add another machine, create `configs/paths/<name>.yaml` that sets `root`, and select it with
`paths=<name>` or `C3WM_PLATFORM=<name>`.

Scripts that read `SLURM_ARRAY_TASK_ID` and `SLURM_ARRAY_TASK_COUNT` split their work over a Slurm array
(`collect_world.py`, `comm_power.py stage=trace`, `deploy_trace.py`). Without them they run as a single job.

## Reproducing the paper

The defaults in `configs/` are the settings of the paper. Checkpoint paths are filled in by hand: each step
below names the file it produces, which a later step takes as an argument.

### 1. Dataset

The dataset used in the paper is on Hugging Face:

```bash
hf download ahmedkhaled/C3-wm-V1 --repo-type=dataset \
    --local-dir $C3WM_ROOT/datasets/c3_wm/findgoal/findgoal_layouts
```

It can also be collected again:

```bash
python collect_world.py
```

200 training layouts and 60 validation and test layouts, 40 episodes each. Writes
`datasets/.../findgoal_layouts/dataset.h5`.

### 2. VQ-VAE (message encoder)

```bash
python train.py
```

Checkpoint: `phase-1-vqvae_pretraining/<run>/checkpoints/best_state_*.pt` (VQVAE below).

### 3. World model without messages

```bash
python train.py pipeline=wm_train logging_params.project_name=phase-0-wm-pretraining
```

Checkpoint: `phase-0-wm-pretraining/<run>/checkpoints/best_state_*.pt` (BLIND below).

### 4. World model with messages (fine-tuning)

```bash
python train.py pipeline=wm_finetune logging_params.project_name=phase-2-wm-finetune \
    pipeline.model.jepa.checkpoint_path=$BLIND pipeline.model.vqvae.checkpoint_path=$VQVAE
```

Checkpoint: `phase-2-wm-finetune/<run>/checkpoints/best_state_*.pt` (WM below).

### 5. Planning modules

```bash
python embed_cache.py checkpoints.jepa=$WM checkpoints.vqvae=$VQVAE out_dir=$CACHE
python memory_cost.py cache=$CACHE
python policy_prior.py cache=$CACHE checkpoints.memory_cost=$MEMORY
```

`memory_cost.py` writes `memory_cost.pt` (MEMORY), and `policy_prior.py` writes `policy_prior.pt` (PRIOR).
`embed_cache.py` can be split over GPUs with `part=<i> parts=<n>` and the same `out_dir`.

### 6. Deployment layouts

`configs/deploy/layouts.json` holds the layout groups used in the paper (in-distribution, slightly OOD, OOD). It
can be regenerated with `python deploy_layouts.py`. The channels of these layouts are traced once:

```bash
python deploy_trace.py
python deploy_trace.py layout_group.name=in-distribution
python deploy_trace.py layout_group.name=slightly-OOD
```

### 7. Power control

The λ-policy is trained on channels along the planner's own trajectories. First collect the rollouts, then record
the planner over a perfect channel on the calibration and test layouts:

```bash
python comm_power.py stage=collect
python deploy.py "methods=[perfect channel]" layouts=[200,260] layout_group.file=null episodes=600 \
    checkpoints.jepa=$WM checkpoints.vqvae=$VQVAE checkpoints.memory_cost=$MEMORY checkpoints.prior=null \
    checkpoints.power_control=null planner.temperature=0 planner.novelty=history planner.explore_within=null \
    planner.explore_below=0.05 record.dir=$C3WM_ROOT/datasets/c3_wm/findgoal/comm_power/deploy_fixed/trajectories_planner
```

Then trace, train and evaluate:

```bash
python comm_power.py stage=trace
python comm_power.py stage=train
python comm_power.py stage=evaluate power_control_from=<train run folder>
```

The train run writes `power_control.pkl` and `lambda_policy.pkl` (POWER below is the path to `power_control.pkl`).
The evaluation reports the reliability against the transmit power of every policy and the sweep over the target
SNR. `python comm_power.py` with no stage runs the four stages in order.

### 8. Deployment

```bash
D="checkpoints.jepa=$WM checkpoints.vqvae=$VQVAE checkpoints.memory_cost=$MEMORY checkpoints.prior=$PRIOR checkpoints.power_control=$POWER"
python deploy.py $D "methods=[lambda-policy]"
```

Each run plays the 1,024 test episodes of the OOD layout group and writes `results.json`. Runs can be split
over several jobs: part `i` runs with `seed=<first episode> episodes=<count> paths.run=<folder>/part_<i>_job_<id>`,
and `python deploy_merge.py <folder>` merges the parts. Methods are compared episode by episode with
`python deploy_compare.py <results.json> ...`, which also gives the success rate within a step budget.

Overrides used for each result, added to `$D`:

| Result | Overrides |
|---|---|
| C3-WM, no messages | `"methods=[no communication]"` |
| C3-WM, perfect channel | `"methods=[perfect channel]" planner.cic=8` |
| C3-WM, λ-policy (0.90, 0.95, 0.99) | `"methods=[lambda-policy]" link.lambda_policy.target=0.9` (0.95 is the default) |
| Perfect CSI, margin policy, fixed power | `"methods=[perfect CSI]"`, `"methods=[learned]"`, `"methods=[fixed power]"` |
| Message erasures | `"methods=[perfect channel]" delivery=0.75` (0.5, 0.25) |
| Memory only / predictor only | `"methods=[perfect channel]" planner.predictor_messages=false` / `planner.goal_from=own` |
| Edited layouts | `layout_group.name=in-distribution` or `layout_group.name=slightly-OOD` |
| Appearance and layout changes | `agent_colors=[purple,yellow]`, `vary.wall_color=orange`, `vary.goal_color=cyan`, `vary.obstacles=2 vary.clutter=3`, `vary.obstacles=12 vary.clutter=20` |
| Changes during an episode | `surprise.kind=goal`, `surprise.kind=goal surprise.when=20`, `surprise.kind=wall` |
| Prediction error after a change | `planner.record_surprise=true` |
| Memory reset | `planner.record_surprise=true planner.forget_above=1.0` |
| Number of agents | `env.agents=3 agent_colors=[red,blue]` (up to 6) |
| Messages stopped from step t | `"methods=[perfect channel]" planner.record_surprise=true planner.reseed=true withhold_from=5`; control: `planner.resample_from=5` instead of `withhold_from` |
| Planning horizon | `"methods=[perfect channel]" planner.horizon=20` |
| Prediction noise | `planner.imagine_noise=0.2` |
| Decision time | `episodes=8 num_envs=1` |

The example episodes come from two recordings of the same seeds, drawn with `filmstrip.py`:

```bash
python deploy.py $D "methods=[no communication]" episodes=256 record.method=no-communication record.dir=$OUT/without
python deploy.py $D "methods=[perfect channel]" episodes=256 record.method=perfect-channel record.dir=$OUT/with
python filmstrip.py --without $OUT/without <its results.json> --with $OUT/with <its results.json> --layout-group OOD
```

The same recordings can be rendered as side-by-side videos (MP4) with
`python render_episodes.py --without ... --with ... --layouts-file ../configs/deploy/layouts.json --seeds 1`.

The effect of messages on the world model's prediction loss is computed with
`python message_gain.py checkpoints.jepa=$WM checkpoints.vqvae=$VQVAE`. The predictor fine-tuned from random
initialization uses `pipeline.blind=encoder` in step 4, with `pipeline.trainer.snapshots=[...]` to keep
intermediate models.

### 9. Baselines

Train each baseline, then deploy it with `deploy.py $D "methods=[<name>]" checkpoints.<name>=<file>`. Table II
reports the better of the most likely actions and the sampled actions (`baseline_sample=true`). Baselines that
communicate send their messages over a perfect channel by default; `baseline_link=lambda-policy` sends them over
the λ-policy link, and `baseline_link=no_communication` blocks them.

| Baseline | Training | Deployment |
|---|---|---|
| GCBC | `python gcrl_baseline.py agent=gcbc steps=1000000 log_every=10000` | `methods=[gcrl] checkpoints.gcrl=.../gcrl.pkl` |
| HIQL | `python gcrl_baseline.py agent=hiql steps=100000` | `methods=[gcrl] checkpoints.gcrl=.../gcrl.pkl` |
| MAPPO | `python mappo_baseline.py checkpoints.jepa=$WM` | `methods=[mappo] checkpoints.mappo=.../mappo.pt` |
| TarMAC | `python tarmac_baseline.py checkpoints.jepa=$WM near_coeff=0.1 train.q_baseline=true train.max_grad_norm=0.5` | `methods=[tarmac] checkpoints.tarmac=.../tarmac.pt` |
| CommNet | `python commnet_baseline.py checkpoints.jepa=$WM cache=$CACHE` | `methods=[commnet] checkpoints.commnet=.../commnet.pt` |
| OmniVLA-edge | `python omnivla_baseline.py pretrained=<omnivla-edge.pth>` | `methods=[omnivla] checkpoints.omnivla=.../omnivla.pt` |
| DreamerV3 | `python dreamer_baseline.py checkpoints.jepa=$WM kind=dreamer features=cnn` | `methods=[dreamer] checkpoints.dreamer=.../dreamer.pt` |
| CoDreamer | `python dreamer_baseline.py checkpoints.jepa=$WM kind=codreamer features=cnn` | `methods=[codreamer] checkpoints.codreamer=.../codreamer.pt` |
| MAMBA | `python mamba_baseline.py checkpoints.jepa=$WM` | `methods=[mamba] checkpoints.mamba=.../mamba.pt` |

The OmniVLA-edge weights (`omnivla-edge.pth`) are available from https://huggingface.co/NHirose/omnivla-edge.

## License

Apache License 2.0.
