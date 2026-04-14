"""
core/nn/models.py
=================
Neural network model definitions for Archero 2 bot.

All models are lightweight and designed for real-time inference on CPU.
PyTorch is required only at training time; inference can run on any hardware.

Models
------
ScreenClassifier    - Classifies game screen into 9 states (MobileNet-style)
SkillCardClassifier - Identifies skill cards from icon crops (EfficientNet-lite)
CombatDetector      - Detects player/enemies/projectiles (tiny YOLO-style head)
DodgePolicy         - PPO policy network for combat movement
SelectionPolicy     - Ranks skill cards by build value (attention-based)
"""
from __future__ import annotations

SCREEN_CLASSES = [
    "combat",       # 0 - active gameplay
    "pause",        # 1 - pause / wave transition
    "skill",        # 2 - level-up skill selection (3 cards)
    "angel",        # 3 - angel gift (2 cards, green)
    "valkyrie",     # 4 - valkyrie blessing (2 cards, blue/gold)
    "devil",        # 5 - devil deal (1 card, red banner)
    "menu",         # 6 - main menu / lobby
    "death",        # 7 - death screen
    "reward",       # 8 - chapter / stage reward
]
NUM_SCREEN_CLASSES = len(SCREEN_CLASSES)

# Input resolution for all models (keeps inference fast)
INPUT_H, INPUT_W = 224, 224
CARD_H,  CARD_W  = 112, 112   # card crop resolution

# ── Try to import PyTorch; fall back gracefully so the rest of the codebase
#    can import this module even without torch installed. ──────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False
    nn = None           # type: ignore


def _require_torch():
    if not _TORCH:
        raise ImportError(
            "PyTorch is required for neural model training/inference.\n"
            "Install with:  pip install torch torchvision"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Shared building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _make_depthwise_block(in_ch: int, out_ch: int, stride: int = 1):
    """Depthwise-separable conv block (MobileNet style)."""
    _require_torch()
    return nn.Sequential(
        # depthwise
        nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU6(inplace=True),
        # pointwise
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Screen Classifier
# ─────────────────────────────────────────────────────────────────────────────

class ScreenClassifier(nn.Module if _TORCH else object):
    """
    Lightweight CNN that classifies a 224×224 frame into one of 9 screen states.

    Architecture: 6 depthwise-separable blocks → global avg pool → FC
    Parameters:  ~220 K  (runs at 200+ FPS on CPU)
    Input:       (B, 3, 224, 224)  float32, normalised 0-1
    Output:      (B, 9)  logits
    """

    def __init__(self, num_classes: int = NUM_SCREEN_CLASSES):
        _require_torch()
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(
            _make_depthwise_block(16, 32, stride=2),   # 56×56
            _make_depthwise_block(32, 64, stride=2),   # 28×28
            _make_depthwise_block(64, 96, stride=2),   # 14×14
            _make_depthwise_block(96, 128, stride=2),  # 7×7
            _make_depthwise_block(128, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=-1)
            idx    = probs.argmax(dim=-1)
        return idx, probs


# ─────────────────────────────────────────────────────────────────────────────
# 2. Skill Card Classifier
# ─────────────────────────────────────────────────────────────────────────────

class SkillCardClassifier(nn.Module if _TORCH else object):
    """
    Classifies a single 112×112 card-icon crop directly into a skill name.

    Architecture: 5 depthwise blocks → global avg pool → embedding → cosine head
    Parameters:  ~180 K
    Input:       (B, 3, 112, 112)  float32, normalised 0-1
    Output:      (B, num_skills)   logits
    """

    def __init__(self, num_skills: int, embed_dim: int = 128):
        _require_torch()
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
            _make_depthwise_block(16, 32, stride=2),
            _make_depthwise_block(32, 64, stride=2),
            _make_depthwise_block(64, 96, stride=2),
            _make_depthwise_block(96, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.embed  = nn.Linear(128, embed_dim)
        self.head   = nn.Linear(embed_dim, num_skills)

    def forward(self, x):
        feat = F.normalize(self.embed(self.encoder(x)), dim=-1)
        return self.head(feat)

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=-1)
            idx    = probs.argmax(dim=-1)
        return idx, probs

    def get_embedding(self, x) -> "torch.Tensor":
        """Return L2-normalised embedding (for metric-learning / k-NN retrieval)."""
        with torch.no_grad():
            return F.normalize(self.embed(self.encoder(x)), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Combat Object Detector
# ─────────────────────────────────────────────────────────────────────────────

class CombatDetector(nn.Module if _TORCH else object):
    """
    Tiny anchor-free object detector for combat scene understanding.

    Detects: player · enemies · projectiles · pickups
    Architecture: lightweight backbone → 4 detection heads (one per class)
    Each head predicts a (H/8 × W/8) heatmap + (x,y,w,h) offsets per cell.

    Input:  (B, 3, 224, 224)
    Output: dict with keys 'player','enemy','projectile','pickup'
            each value: (B, 5, H, W) — [conf, cx, cy, w, h]
    """

    CLASSES = ["player", "enemy", "projectile", "pickup"]

    def __init__(self):
        _require_torch()
        super().__init__()

        # Backbone: 5 layers → 28×28 feature map
        self.backbone = nn.Sequential(
            nn.Conv2d(3,  16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
            _make_depthwise_block(16, 32,  stride=2),   # 56×56
            _make_depthwise_block(32, 64,  stride=2),   # 28×28
            _make_depthwise_block(64, 96),
            _make_depthwise_block(96, 96),
        )
        # Per-class detection heads
        self.heads = nn.ModuleDict({
            cls: nn.Sequential(
                nn.Conv2d(96, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
                nn.Conv2d(32, 5, 1),   # conf + cx + cy + w + h
            )
            for cls in self.CLASSES
        })

    def forward(self, x):
        feat = self.backbone(x)
        return {cls: head(feat) for cls, head in self.heads.items()}

    def decode(self, x, conf_thresh: float = 0.4):
        """Return list of dicts per class: [{cx,cy,w,h,conf}, ...]."""
        with torch.no_grad():
            raw = self.forward(x)        # single-image input expected (B=1)
            results = {}
            for cls, pred in raw.items():
                pred = pred[0]           # (5, H, W)
                conf = torch.sigmoid(pred[0])
                mask = conf > conf_thresh
                ys, xs = mask.nonzero(as_tuple=True)
                boxes = []
                for y, x_ in zip(ys.tolist(), xs.tolist()):
                    c  = float(conf[y, x_])
                    cx = float(torch.sigmoid(pred[1, y, x_]))
                    cy = float(torch.sigmoid(pred[2, y, x_]))
                    bw = float(torch.sigmoid(pred[3, y, x_]))
                    bh = float(torch.sigmoid(pred[4, y, x_]))
                    boxes.append(dict(cx=cx, cy=cy, w=bw, h=bh, conf=c))
                results[cls] = boxes
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. State Encoder  (CNN feature extractor for RL state)
# ─────────────────────────────────────────────────────────────────────────────

STATE_DIM = 256   # dimension of the encoded state vector

class StateEncoder(nn.Module if _TORCH else object):
    """
    Converts a raw game frame (and optional temporal stack) into a compact
    state vector for downstream RL policies.

    Input:  (B, C, H, W)  where C = 3*num_frames (stacked grayscale frames)
    Output: (B, STATE_DIM)
    """

    def __init__(self, num_frames: int = 4):
        _require_torch()
        super().__init__()
        in_ch = num_frames * 3
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, stride=4, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Lazy-resolve output dim
        dummy_h, dummy_w = 84, 84
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, dummy_h, dummy_w)
            cnn_out = self.cnn(dummy).shape[1]
        self.fc = nn.Linear(cnn_out, STATE_DIM)

    def forward(self, x):
        return F.relu(self.fc(self.cnn(x)))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dodge Policy  (PPO Actor-Critic for combat movement)
# ─────────────────────────────────────────────────────────────────────────────

DODGE_ACTIONS = [
    "idle",
    "up", "down", "left", "right",
    "up_left", "up_right", "down_left", "down_right",
]
NUM_DODGE_ACTIONS = len(DODGE_ACTIONS)


class DodgePolicy(nn.Module if _TORCH else object):
    """
    PPO Actor-Critic for real-time combat movement.

    Input:  state vector (STATE_DIM,) from StateEncoder
            + structured combat features (hp, enemy_count, nearest_enemy_dir×2,
              projectile_density, pickup_nearby)  → 6 extra scalars

    Output: action logits (NUM_DODGE_ACTIONS,) + value scalar
    """

    EXTRA_FEATURES = 6

    def __init__(self, state_dim: int = STATE_DIM):
        _require_torch()
        super().__init__()
        in_dim = state_dim + self.EXTRA_FEATURES
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256),   nn.ReLU(),
        )
        self.actor  = nn.Linear(256, NUM_DODGE_ACTIONS)
        self.critic = nn.Linear(256, 1)

    def forward(self, state, extras):
        x    = torch.cat([state, extras], dim=-1)
        feat = self.shared(x)
        return self.actor(feat), self.critic(feat)

    def act(self, state, extras, greedy: bool = False):
        with torch.no_grad():
            logits, value = self.forward(state, extras)
            dist   = torch.distributions.Categorical(logits=logits)
            action = dist.probs.argmax() if greedy else dist.sample()
        return int(action), float(dist.log_prob(action)), float(value)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Selection Policy  (ranks skill cards using learned priorities)
# ─────────────────────────────────────────────────────────────────────────────

class SelectionPolicy(nn.Module if _TORCH else object):
    """
    Scores available skill cards and outputs a pick index.

    Inputs per card:
      - skill_id  (int, embedded)
      - ocr_conf  (float)
      - tmpl_conf (float)
      - db_priority (float, normalised 0-1)
      - rarity_one_hot (4,)

    Context:
      - current hp ratio
      - wave number (normalised)
      - previously chosen skills (multi-hot, len=num_skills)
    """

    def __init__(self, num_skills: int, max_cards: int = 3, embed_dim: int = 32):
        _require_torch()
        super().__init__()
        self.embed = nn.Embedding(num_skills + 1, embed_dim, padding_idx=0)
        card_in = embed_dim + 2 + 1 + 4          # embed + confs + prio + rarity
        ctx_in  = 2 + num_skills                  # hp + wave + history
        self.card_enc  = nn.Sequential(nn.Linear(card_in, 64), nn.ReLU(), nn.Linear(64, 32))
        self.ctx_enc   = nn.Sequential(nn.Linear(ctx_in, 64),  nn.ReLU(), nn.Linear(64, 32))
        self.scorer    = nn.Linear(32 + 32, 1)

    def forward(self, card_ids, card_feats, context):
        emb   = self.embed(card_ids)                          # (B, N, E)
        card  = self.card_enc(torch.cat([emb, card_feats], dim=-1))   # (B, N, 32)
        ctx   = self.ctx_enc(context).unsqueeze(1).expand_as(card)    # (B, N, 32)
        score = self.scorer(torch.cat([card, ctx], dim=-1)).squeeze(-1)  # (B, N)
        return score

    def pick(self, card_ids, card_feats, context):
        with torch.no_grad():
            scores = self.forward(card_ids, card_feats, context)
        return int(scores.argmax(dim=-1))
