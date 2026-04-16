# npc_surrogate_mlp.py — Enhanced NPC Hazard MLP Surrogate
# Summary of changes from the base version:
# 1) Introduces target_model (EMA-frozen) dedicated to SVGD score/score_and_grad queries;
#    self.model is still used for training (backward-compatible with train_mlp_all_pairs).
# 2) Provides ema_update(tau) and freeze_target(); call ema_update after each training phase
#    to synchronize training weights into target_model with gradient stability.
# 3) score_and_grad supports an optional “logit gradient” mode (use_logit_grad=True)
#    that strengthens gradient signal in extreme probability regions (default False).
# 4) Map-derived features are clamped to reduce the effect of outliers on numerical stability.
# 5) All inference paths use target_model.eval() with requires_grad=False, ensuring
#    gradients are built only with respect to (ds, dd, dyaw).

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import carla  # Available at runtime; import failure does not affect offline tooling.
except Exception:  # pragma: no cover
    carla = None


# ============== Geometry helpers (consistent with the main program) ==============

def _yaw_to_unit(yaw_deg: float):
    r = math.radians(yaw_deg)
    return math.cos(r), math.sin(r)


def _relative_yaw_deg(ego_tf, npc_tf) -> float:
    dy = npc_tf.rotation.yaw - ego_tf.rotation.yaw
    while dy >= 180:
        dy -= 360
    while dy < -180:
        dy += 360
    return dy


def _ego_local_sd(ego_tf, loc):
    dx = loc.x - ego_tf.location.x
    dy = loc.y - ego_tf.location.y
    cy, sy = _yaw_to_unit(ego_tf.rotation.yaw)
    s = dx * cy + dy * sy
    d = -dx * sy + dy * cy
    return s, d


def _lane_center_offset(world_map, npc_tf) -> float:
    try:
        wp = world_map.get_waypoint(
            npc_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
    except Exception:
        wp = None
    if not wp:
        return 0.0
    center = wp.transform.location
    right = wp.transform.get_right_vector()
    dvx = npc_tf.location.x - center.x
    dvy = npc_tf.location.y - center.y
    dvz = npc_tf.location.z - center.z
    return dvx * right.x + dvy * right.y + dvz * right.z


def _curvature_approx(world_map, npc_tf, ds=2.0) -> float:
    try:
        wp = world_map.get_waypoint(
            npc_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
    except Exception:
        wp = None
    if not wp:
        return 0.0
    try:
        prevs = wp.previous(ds)
        nexts = wp.next(ds)
        if not prevs or not nexts:
            return 0.0
        yaw0 = prevs[0].transform.rotation.yaw
        yaw2 = nexts[0].transform.rotation.yaw
        dyaw = math.radians(((yaw2 - yaw0 + 180) % 360) - 180)
        return float(abs(dyaw) / (2.0 * ds))
    except Exception:
        return 0.0


# ============== MLP model ==============

class _HazardMLP(nn.Module):
    def __init__(self, in_dim=8, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1), nn.Sigmoid(),  # Sigmoid output in [0, 1] representing collision hazard probability.
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class SurrogateOptions:
    use_logit_grad: bool = False  # If True, use logit(p) gradients to amplify signal near 0/1 boundaries.
    clamp_lane_lat: float = 3.0   # Lateral offset from lane center clamped to this range (meters).
    clamp_curv: float = 0.5       # Road curvature clamp (1/m); values above this are usually noise.
    clamp_spd_norm: Tuple[float, float] = (0.0, 2.0)  # Normalization range for the speed feature.


class NPCHazardMLPSurrogate:
    """
    MLP surrogate that predicts NPC collision hazard from initial-pose features.

    Input feature vector (consistent with score_and_grad):
        [ s/12, d/1.5, sin(Δyaw), cos(Δyaw), lane_lat, curv, is_junc, spd_norm ]
    where s and d are longitudinal/lateral offsets in the ego frame (m), Δyaw is the
    relative heading (rad), lane_lat is signed lateral offset from lane center (m),
    curv is road curvature (1/m), is_junc is 0/1 junction flag, and spd_norm is the
    normalized speed limit.

    Usage:
      - Training: operate on self.model (backward-compatible with train_mlp_all_pairs).
      - Inference / gradient queries: use self.target_model (EMA-frozen) via score / score_and_grad.
      - After training, call ema_update(tau) to sync weights into target_model,
        or call freeze_target() on first use to copy weights directly.
    """

    def __init__(self, device: Optional[str] = None, ckpt_path: str = "mlp_frozen.pt", opts: Optional[SurrogateOptions] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = _HazardMLP().to(self.device)           # Training model.
        self.target_model = _HazardMLP().to(self.device)    # Frozen inference/gradient model (EMA target).
        self.opts = opts or SurrogateOptions()
        # Attempt to load pre-trained weights into the training model.
        self._load_if_exists(ckpt_path)
        # Initialize target weights from the training model and freeze them.
        self.freeze_target(copy_from_model=True)

    # --------- I/O ---------
    def _load_if_exists(self, path):
        if path and os.path.isfile(path):
            try:
                payload = torch.load(path, map_location=self.device)
                state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
                self.model.load_state_dict(state)
                self.model.eval()
                print(f"[SVGD-MLP] loaded weights from {path}")
            except Exception as e:
                print(f"[SVGD-MLP] failed to load {path}: {e}. Using random init.")
        else:
            print("[SVGD-MLP] no checkpoint found, using randomly-initialized MLP")

    def save(self, path="mlp_frozen.pt", save_target: bool = False):
        try:
            to_save = self.target_model if save_target else self.model
            torch.save({"state_dict": to_save.state_dict()}, path)
            print(f"[SVGD-MLP] saved {'target_model' if save_target else 'model'} to {path}")
        except Exception as e:
            print(f"[SVGD-MLP] save error: {e}")

    # --------- Target network (EMA-frozen) management ---------
    @torch.no_grad()
    def ema_update(self, tau: float = 0.05):
        """Sync training weights into target_model via EMA: target = (1-tau)*target + tau*model."""
        for p_t, p_s in zip(self.target_model.parameters(), self.model.parameters()):
            p_t.copy_(p_t * (1.0 - tau) + p_s * tau)
        self._freeze_(self.target_model)

    @torch.no_grad()
    def freeze_target(self, copy_from_model: bool = False):
        if copy_from_model:
            self.target_model.load_state_dict(self.model.state_dict())
        self._freeze_(self.target_model)

    @staticmethod
    def _freeze_(m: nn.Module):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    # --------- Core API: score, score_and_grad ---------
    @torch.no_grad()
    def score(self, world_map, ego_tf, npc_tf) -> float:
        feats = self._build_feats(world_map, ego_tf, npc_tf)
        x = torch.from_numpy(feats).float().unsqueeze(0).to(self.device)  # [1,8]
        self.target_model.eval()
        p = self.target_model(x).squeeze().item()
        return float(p)

    def score_and_grad(self, world_map, ego_tf, npc_tf, x_vec, h=0.5):  # h retained for call-signature compatibility; unused.
        “””
        Compute hazard score and its gradient with respect to (ds, dd, dyaw_deg).

        Args:
            x_vec: [ds, dd, dyaw_deg] — perturbation offsets in meters/meters/degrees.
            h: Unused; kept for backward-compatible call signatures.

        Returns:
            (f, grad): f in [0, 1]; grad is the partial derivative w.r.t. (ds, dd, dyaw_deg).
            Only builds the computation graph w.r.t. x_vec; target_model parameters are frozen.
            When use_logit_grad=True, returns the gradient of logit(f) (clipped to prevent divergence).
        “””
        # Differentiable input variables.
        ds = torch.tensor(float(x_vec[0]), dtype=torch.float32, device=self.device, requires_grad=True)
        dd = torch.tensor(float(x_vec[1]), dtype=torch.float32, device=self.device, requires_grad=True)
        dy = torch.tensor(float(x_vec[2]), dtype=torch.float32, device=self.device, requires_grad=True)  # degrees

        # Baseline pose of the NPC in the ego frame.
        s0, d0 = _ego_local_sd(ego_tf, npc_tf.location)
        base_rel_deg = _relative_yaw_deg(ego_tf, npc_tf)

        # Apply the (ds, dd, dyaw) increments to the features using the same scaling as _build_feats.
        s = (torch.tensor(s0, device=self.device) + ds) / 12.0   # Normalize by expected longitudinal range ~12 m.
        d = (torch.tensor(d0, device=self.device) + dd) / 1.5    # Normalize by half a standard lane width ~1.5 m.
        rel_rad = torch.tensor(math.radians(base_rel_deg), device=self.device) + dy * (math.pi / 180.0)
        sp = torch.sin(rel_rad)
        cp = torch.cos(rel_rad)

        # Map-derived features are treated as constants (not part of the computation graph).
        lane_lat = float(_lane_center_offset(world_map, npc_tf))
        lane_lat = float(max(-self.opts.clamp_lane_lat, min(self.opts.clamp_lane_lat, lane_lat)))
        curv = float(_curvature_approx(world_map, npc_tf, ds=2.0))
        curv = float(max(-self.opts.clamp_curv, min(self.opts.clamp_curv, curv)))
        try:
            wp = world_map.get_waypoint(npc_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        except Exception:
            wp = None
        is_junc = 1.0 if (wp and wp.is_junction) else 0.0
        spd = 0.0
        if wp:
            try:
                spd = float(getattr(wp, "speed_limit", 0.0))
            except Exception:
                try:
                    spd = float(wp.get_speed_limit())
                except Exception:
                    spd = 0.0
        lo, hi = self.opts.clamp_spd_norm
        spd_norm = float(np.clip(spd / 30.0, lo, hi))  # Normalize speed limit by 30 km/h reference value.

        feats = torch.stack([
            s, d,
            sp, cp,
            torch.tensor(lane_lat, dtype=torch.float32, device=self.device),
            torch.tensor(curv, dtype=torch.float32, device=self.device),
            torch.tensor(is_junc, dtype=torch.float32, device=self.device),
            torch.tensor(spd_norm, dtype=torch.float32, device=self.device),
        ], dim=0).unsqueeze(0)  # [1,8]

        # Forward pass through frozen target model (gradients are built only for x_vec).
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad_(False)
        y = self.target_model(feats)  # [1,1] in [0,1]
        p = y.squeeze()               # Scalar tensor.

        # Choose gradient objective: raw probability or logit.
        if self.opts.use_logit_grad:
            # logit(p) = log(p/(1-p)); ∂logit/∂x = ∂p/∂x / (p*(1-p)).
            # Add epsilon guards on the denominator and clip the overall gradient for robustness.
            logit = torch.log(torch.clamp(p, 1e-6, 1-1e-6)) - torch.log(torch.clamp(1-p, 1e-6, 1.0))
            obj = logit
        else:
            obj = p

        # Backpropagate into (ds, dd, dy).
        self.target_model.zero_grad(set_to_none=True)
        for t in (ds, dd, dy):
            if t.grad is not None:
                t.grad = None
        obj.backward()

        g_ds = ds.grad.item() if ds.grad is not None else 0.0
        g_dd = dd.grad.item() if dd.grad is not None else 0.0
        g_dy = dy.grad.item() if dy.grad is not None else 0.0

        # Clip gradient to prevent divergence (especially relevant when use_logit_grad=True).
        g = np.array([g_ds, g_dd, g_dy], dtype=float)
        g = np.clip(g, -10.0, 10.0)
        return float(p.item()), g

    # --------- Internal: build feature vector consistent with the forward pass ---------
    def _build_feats(self, world_map, ego_tf, npc_tf):
        s0, d0 = _ego_local_sd(ego_tf, npc_tf.location)
        rel_deg = _relative_yaw_deg(ego_tf, npc_tf)
        rel_rad = math.radians(rel_deg)
        lane_lat = _lane_center_offset(world_map, npc_tf)
        lane_lat = float(max(-self.opts.clamp_lane_lat, min(self.opts.clamp_lane_lat, lane_lat)))
        curv = _curvature_approx(world_map, npc_tf, ds=2.0)
        curv = float(max(-self.opts.clamp_curv, min(self.opts.clamp_curv, curv)))
        try:
            wp = world_map.get_waypoint(npc_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        except Exception:
            wp = None
        is_junc = 1.0 if (wp and wp.is_junction) else 0.0
        spd = 0.0
        if wp:
            try:
                spd = float(getattr(wp, "speed_limit", 0.0))
            except Exception:
                try:
                    spd = float(wp.get_speed_limit())
                except Exception:
                    spd = 0.0
        lo, hi = self.opts.clamp_spd_norm
        spd_norm = float(np.clip(spd / 30.0, lo, hi))  # Normalize speed limit by 30 km/h reference value.
        feats = np.array([
            s0 / 12.0,   # Normalize by expected longitudinal range ~12 m.
            d0 / 1.5,    # Normalize by half a standard lane width ~1.5 m.
            math.sin(rel_rad),
            math.cos(rel_rad),
            lane_lat,
            curv,
            is_junc,
            spd_norm,
        ], dtype=np.float32)
        return feats
