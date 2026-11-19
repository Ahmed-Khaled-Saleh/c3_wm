__all__ = ['DIR_VECTORS', 'modulate', 'relative_pose', 'moved_poses', 'latest_messages', 'SIGReg', 'DiscreteActionEncoder',
           'FeedForward', 'Attention', 'MessageConditionedBlock', 'ConditionalBlock', 'Block', 'Transformer',
           'Embedder', 'MLP', 'ARPredictor', 'detach_clone', 'JEPA']

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

DIR_VECTORS = ((1, 0), (0, 1), (-1, 0), (0, -1))

def relative_pose(sender_pos, sender_dir, receiver_pos, receiver_dir):
    vec = torch.tensor(DIR_VECTORS, device=sender_pos.device)
    ahead = vec[receiver_dir]
    right = torch.stack([-ahead[..., 1], ahead[..., 0]], -1)
    d = sender_pos - receiver_pos
    return torch.stack([(d * ahead).sum(-1), (d * right).sum(-1), (sender_dir - receiver_dir) % 4], -1)

def moved_poses(pose, actions):
    vec = torch.tensor(DIR_VECTORS, device=pose.device)
    ahead, right = vec[0], torch.stack([-vec[0, 1], vec[0, 0]])
    sender = pose[..., :1] * ahead + pose[..., 1:2] * right
    pos = torch.zeros(*actions.shape, 2, dtype=torch.long, device=pose.device)[..., 0, :]
    heading = torch.zeros_like(actions[..., 0])
    out = []
    for k in range(actions.shape[-1]):
        a = actions[..., k]
        heading = torch.where(a == 0, (heading - 1) % 4, torch.where(a == 1, (heading + 1) % 4, heading))
        pos = pos + vec[heading] * (a == 2).unsqueeze(-1)
        out.append(relative_pose(sender, pose[..., 2], pos[..., None, :], heading[..., None]))
    return torch.stack(out, -3)

def latest_messages(tokens, received, sender, receiver, steps: int, max_age: int) -> dict:
    B, S, N = received.shape
    t = torch.arange(S, device=received.device).view(1, S, 1).expand(B, S, N)
    last = torch.where(received, t, torch.full_like(t, -1)).cummax(dim=1).values[:, -steps:]
    age = t[:, -steps:] - last
    ok = (last >= 0) & (age <= max_age)
    index = last.clamp(min=0)

    def at_last(x):
        i = index.view(B, steps, N, *[1] * (x.dim() - 3)).expand(B, steps, N, *x.shape[3:])
        return torch.gather(x, 1, i)

    sent_from, now = at_last(sender), receiver[:, -steps:, None]
    pose = relative_pose(sent_from[..., :2], sent_from[..., 2], now[..., :2], now[..., 2])
    return {"tokens": at_last(tokens), "received": ok, "age": torch.where(ok, age, torch.zeros_like(age)), "pose": pose}

class SIGReg(torch.nn.Module):

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class DiscreteActionEncoder(nn.Module):
    def __init__(self, num_actions, action_emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, action_emb_dim)

    def forward(self, action):
        return self.embedding(action)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class MessageConditionedBlock(nn.Module):

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout
        )
        nn.init.constant_(self.cross_attn.out_proj.weight, 0)
        nn.init.constant_(self.cross_attn.out_proj.bias, 0)
        self.cross_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)

    def forward(self, x, c, msg_keys=None, msg_mask=None, return_attention=False, samples=1):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        attn_weights = None
        if msg_keys is not None:
            B, T, D = x.shape
            R, S = B // samples, samples
            keys = self.key_norm(msg_keys)
            query = self.cross_norm(x).view(R, S, T, D).transpose(1, 2).reshape(R * T, S, D)
            read, attn_weights = self.cross_attn(
                query=query, key=keys, value=keys, key_padding_mask=msg_mask, need_weights=return_attention,
            )
            x = x + read.view(R, T, S, D).transpose(1, 2).reshape(B, T, D)
            if attn_weights is not None:
                attn_weights = attn_weights.view(R, T, S, -1).transpose(1, 2).reshape(B, T, -1)

        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        if return_attention:
            return x, attn_weights
        return x

class ConditionalBlock(nn.Module):

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None, msg_keys=None, msg_mask=None, return_attention=False, samples=1):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)
        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        all_attn_weights = []

        for block in self.layers:
            if isinstance(block, MessageConditionedBlock):
                if return_attention:
                    x, attn = block(x, c, msg_keys=msg_keys, msg_mask=msg_mask, return_attention=True, samples=samples)
                    all_attn_weights.append(attn)
                else:
                    x = block(x, c, msg_keys=msg_keys, msg_mask=msg_mask, samples=samples)
            elif isinstance(block, ConditionalBlock):
                x = block(x, c)
            else:
                x = block(x)

        x = self.norm(x)
        if hasattr(self, "output_proj"):
            x = self.output_proj(x)

        if return_attention:
            return x, all_attn_weights
        return x


class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        return self.net(x)


import math
class ARPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        msg_embedding_dim=256,
        num_embeddings=256,
        num_msg_tokens=49,
        max_offset=14,
        max_age=8,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.num_msg_tokens, self.max_offset, self.max_age = num_msg_tokens, max_offset, max_age

        self.msg_token_embedding = nn.Embedding(num_embeddings, msg_embedding_dim)
        self.msg_proj = (
            nn.Linear(msg_embedding_dim, hidden_dim)
            if msg_embedding_dim != hidden_dim
            else nn.Identity()
        )
        self.msg_pos_spatial = nn.Parameter(torch.randn(1, num_msg_tokens, hidden_dim) * 0.02)
        view = math.isqrt(num_msg_tokens)
        self.reach = max_offset + view - 1
        row, col = torch.arange(num_msg_tokens) // view, torch.arange(num_msg_tokens) % view
        self.register_buffer("tile_cells", torch.stack([view - 1 - row, col - view // 2], -1), persistent=False)
        self.tile_ahead = nn.Embedding(2 * self.reach + 1, hidden_dim)
        self.tile_right = nn.Embedding(2 * self.reach + 1, hidden_dim)
        self.msg_heading = nn.Embedding(4, hidden_dim)
        self.msg_age = nn.Embedding(max_age + 1, hidden_dim)
        for e in (self.tile_ahead, self.tile_right, self.msg_heading, self.msg_age):
            nn.init.normal_(e.weight, std=0.02)
        self.no_msg = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=MessageConditionedBlock,
        )

    def tile_positions(self, pose):
        a, r = self.tile_cells[:, 0], self.tile_cells[:, 1]
        turn = (pose[..., 2] % 4).unsqueeze(-1)
        ahead = torch.stack([a, -r, -a, r])[turn.squeeze(-1)]
        right = torch.stack([r, a, -r, -a])[turn.squeeze(-1)]
        return torch.stack([pose[..., 0:1] + ahead, pose[..., 1:2] + right], -1)

    def message_keys(self, messages):
        tokens, received = messages["tokens"], messages["received"]
        B, T, N, L = tokens.shape
        pose, age, m = messages["pose"], messages["age"], self.reach
        cells = self.tile_positions(pose).clamp(-m, m) + m
        about = self.msg_heading(pose[..., 2] % 4) + self.msg_age(age.clamp(0, self.max_age))
        keys = (self.msg_proj(self.msg_token_embedding(tokens)) + self.msg_pos_spatial
                + self.tile_ahead(cells[..., 0]) + self.tile_right(cells[..., 1]) + about.unsqueeze(-2))
        keys = rearrange(keys, "b t n l d -> (b t) (n l) d")
        keys = torch.cat([keys, self.no_msg.expand(B * T, 1, -1)], dim=1)
        mask = rearrange(~received, "b t n -> (b t) n").repeat_interleave(L, dim=1)
        mask = torch.cat([mask, mask.new_zeros(B * T, 1)], dim=1)
        return keys, mask

    def forward(self, x, c, messages=None, return_attention=False, samples=1):
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)

        keys, mask = self.message_keys(messages) if messages is not None else (None, None)
        return self.transformer(x, c, msg_keys=keys, msg_mask=mask, return_attention=return_attention, samples=samples)


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        **kwargs
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.sigreg = SIGReg(knots=17, num_proj=1024)

    def encode(self, info):
        pixels = info['pixels']
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb, messages=None, samples=1):
        preds = self.predictor(emb, act_emb, messages=messages, samples=samples)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds


    def loss_fn(self, output, pred_emb, target_emb, emb, lambd):
        output["pred_loss"] = (pred_emb - target_emb).pow(2).mean()
        output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
        output["jepa_loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
        return output


    def rollout(self, info, action_sequence, history_size: int = 3, move_messages: bool = False, noise: float = 0.0):

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        info["observed"] = H
        n_steps = T - H

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        messages, samples = info.get("messages"), 1
        if messages is not None:
            if not move_messages and all(torch.equal(v, v[:, :1].expand_as(v)) for v in messages.values()):
                messages, samples = {k: v[:, 0] for k, v in messages.items()}, S
            else:
                messages = {k: rearrange(v, "b s ... -> (b s) ...") for k, v in messages.items()}
        max_age = self.predictor.max_age
        moved = None
        if messages is not None and move_messages:
            plan = rearrange(action_sequence, "b s t ... -> (b s) t ...").reshape(B * S, T)[:, H - 1:]
            moved = moved_poses(messages["pose"][:, H - 1], plan.long())

        def context(n):
            if messages is None:
                return None
            steps = []
            for k in range(max(0, n - HS), n):
                snap = {key: v[:, min(k, H - 1)] for key, v in messages.items()}
                if k >= H:
                    age = snap["age"] + (k - H + 1)
                    snap["received"] = snap["received"] & (age <= max_age)
                    snap["age"] = age.clamp(max=max_age)
                    if moved is not None:
                        snap["pose"] = moved[:, k - H]
                steps.append(snap)
            return {key: torch.stack([s[key] for s in steps], dim=1) for key in steps[0]}

        HS = history_size
        for t in range(n_steps + 1):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            pred_emb = self.predict(emb_trunc, act_trunc, context(emb.size(1)), samples=samples)[:, -1:]
            if noise:
                pred_emb = pred_emb + noise * torch.randn_like(pred_emb)
            emb = torch.cat([emb, pred_emb], dim=1)

            if t < n_steps:
                next_act = act_future[:, t : t + 1]
                act = torch.cat([act, next_act], dim=1)

        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info


    def criterion(self, info_dict: dict):
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]

        goal = goal_emb[..., -1:, :].expand_as(pred_emb).detach()
        per_step = F.mse_loss(pred_emb, goal, reduction="none").sum(-1)

        H = info_dict.get("observed", 3)
        per_step = per_step[..., H:]

        cost = per_step.min(dim=-1).values
        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, :1] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)
        info_dict["goal_emb"] = goal["emb"].unsqueeze(2)
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        return cost
