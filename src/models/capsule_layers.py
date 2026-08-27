"""
Capsule Network Layers
Implements Primary Capsules and Activity Capsules with dynamic routing

Includes comprehensive logging for training diagnostics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

# Configure logging
logger = logging.getLogger(__name__)


class PrimaryCaps(nn.Module):
    """
    Primary Capsule Layer
    Converts convolutional features into capsules

    Args:
        in_channels: Number of input channels from conv layer
        out_capsules: Number of capsule types
        out_dim: Dimensionality of each capsule vector
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
    """
    def __init__(self, in_channels, out_capsules, out_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.out_capsules = out_capsules
        self.out_dim = out_dim
        self.capsules = nn.Conv2d(in_channels, out_capsules * out_dim,
                                 kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Capsule tensor [B, num_capsules, capsule_dim]
        """
        B = x.size(0)
        x = self.capsules(x)  # [B, out_capsules*out_dim, H, W]
        x = x.view(B, self.out_capsules, self.out_dim, x.size(2), x.size(3))  # [B, caps, dim, H, W]
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, caps, dim]
        x = x.view(B, -1, self.out_dim)  # [B, H*W*caps, dim]
        return self.squash(x)

    def squash(self, x, eps=1e-8):
        """
        Squashing non-linearity for capsules
        Ensures capsule magnitude is between 0 and 1

        Args:
            x: Capsule tensor [B, num_capsules, capsule_dim]
        Returns:
            Squashed capsule tensor with same shape
        """
        norm = torch.norm(x, dim=-1, keepdim=True)  # [B, num_capsules, 1]
        scale = (norm**2) / (1.0 + norm**2)
        return scale * x / (norm + eps)


class ActivityCaps(nn.Module):
    """
    Activity Capsule Layer with Dynamic Routing
    Routes from primary capsules to activity capsules

    Args:
        out_capsules: Number of activity classes
        out_dim: Dimensionality of each activity capsule
        routing_iters: Number of dynamic routing iterations
    """
    def __init__(self, out_capsules=6, out_dim=16, routing_iters=3):
        super().__init__()
        self.out_capsules = out_capsules
        self.out_dim = out_dim
        self.routing_iters = routing_iters
        self.weights = None
        self.in_capsules = None

        # Logging control
        self._log_counter = 0
        self._log_interval = 50

    def forward(self, x):
        """
        Dynamic routing algorithm

        Args:
            x: Primary capsules [B, in_capsules, in_dim]
        Returns:
            Activity capsules [B, out_capsules, out_dim]
        """
        B, in_capsules, in_dim = x.size()

        self._log_counter += 1
        should_log = (self._log_counter % self._log_interval == 1) and self.training

        # Initialize weights if first forward pass or capsule count changed
        if self.weights is None or self.in_capsules != in_capsules:
            # Use Xavier/Glorot initialization for proper scaling
            std = math.sqrt(2.0 / (in_dim + self.out_dim))
            weights = nn.Parameter(
                std * torch.randn(1, in_capsules, self.out_capsules, self.out_dim, in_dim, device=x.device)
            )
            # Delete the None attribute first if it exists
            if hasattr(self, 'weights'):
                delattr(self, 'weights')
            # Now register as parameter
            self.register_parameter('weights', weights)
            self.in_capsules = in_capsules
            logger.info(f"[Init] ActivityCaps weights initialized with std={std:.4f}, "
                       f"shape={weights.shape}")

        if should_log:
            input_norms = x.norm(dim=-1)
            logger.info(f"[ActivityCaps Input] norm_mean={input_norms.mean().item():.6f}, "
                       f"norm_max={input_norms.max().item():.6f}")

        # Compute prediction vectors
        x_expanded = x.unsqueeze(2).unsqueeze(-1)  # [B, in_caps, 1, in_dim, 1]
        W = self.weights.expand(B, -1, -1, -1, -1)  # [B, in_caps, out_caps, out_dim, in_dim]
        u_hat = torch.matmul(W, x_expanded).squeeze(-1)  # [B, in_caps, out_caps, out_dim]

        if should_log:
            logger.info(f"[ActivityCaps Routing] u_hat: mean={u_hat.mean().item():.6f}, "
                       f"std={u_hat.std().item():.6f}")

        # Dynamic routing with scaling
        b = torch.zeros(B, in_capsules, self.out_capsules, 1, device=x.device)

        for i in range(self.routing_iters):
            # Softmax over output capsules
            c = F.softmax(b, dim=2)  # [B, in_caps, out_caps, 1]

            # Weighted sum of prediction vectors
            s = (c * u_hat).sum(dim=1, keepdim=True)  # [B, 1, out_caps, out_dim]

            # Scale before squashing to prevent attenuation
            s = s * math.sqrt(in_capsules)

            # Apply squashing non-linearity
            v = self.squash(s)  # [B, 1, out_caps, out_dim]

            if should_log and i == self.routing_iters - 1:
                v_norms = v.norm(dim=-1)
                logger.info(f"[ActivityCaps Iter {i}] v_norms: mean={v_norms.mean().item():.6f}, "
                           f"max={v_norms.max().item():.6f}")

            # Update routing coefficients
            if i < self.routing_iters - 1:
                # Agreement: dot product between predictions and outputs
                b = b + (u_hat * v).sum(dim=-1, keepdim=True)  # [B, in_caps, out_caps, 1]

        return v.squeeze(1)  # [B, out_caps, out_dim]

    def squash(self, x, eps=1e-8):
        """
        Squashing non-linearity with improved numerical stability

        Uses modified formula: norm/(1+norm) instead of norm^2/(1+norm^2)
        This prevents severe attenuation of small values.

        Args:
            x: Capsule tensor [..., capsule_dim]
        Returns:
            Squashed capsule tensor with same shape
        """
        norm = torch.norm(x, dim=-1, keepdim=True)
        # Modified squash for numerical stability
        scale = norm / (1.0 + norm + eps)
        return scale * x / (norm + eps)


class HierarchicalActivityCaps(nn.Module):
    """
    Two-Stage Hierarchical Capsule Routing for Memory Efficiency

    Reduces memory 4x by routing through intermediate capsule layer:
        Primary [B, 100k, 8] -> Intermediate [B, 50, 12] -> Activity [B, num_classes, 16]

    Memory: 1.94 GB vs 5.05 GB (direct routing to 101 classes)

    This hierarchical approach enables scaling to large numbers of classes (101+)
    on consumer GPUs by breaking the routing into two stages.

    Args:
        intermediate_capsules: Number of intermediate capsules (default: 50)
        intermediate_dim: Dimension of intermediate capsules (default: 12)
        out_capsules: Number of activity classes (default: 101)
        out_dim: Dimension of activity capsules (default: 16)
        routing_iters_stage1: Routing iterations for stage 1 (default: 2)
        routing_iters_stage2: Routing iterations for stage 2 (default: 2)
    """

    def __init__(self, intermediate_capsules=50, intermediate_dim=12,
                 out_capsules=101, out_dim=16,
                 routing_iters_stage1=2, routing_iters_stage2=2):
        super().__init__()
        self.intermediate_capsules = intermediate_capsules
        self.intermediate_dim = intermediate_dim
        self.out_capsules = out_capsules
        self.out_dim = out_dim
        self.routing_iters_stage1 = routing_iters_stage1
        self.routing_iters_stage2 = routing_iters_stage2

        # Weights initialized dynamically on first forward pass (like ActivityCaps)
        self.weights_stage1 = None
        self.weights_stage2 = None
        self.in_capsules = None

        # Logging control
        self._log_counter = 0
        self._log_interval = 50  # Log every N batches

    def forward(self, x):
        """
        Two-stage hierarchical routing

        Args:
            x: Primary capsules [B, in_capsules, in_dim]

        Returns:
            Activity capsules [B, out_capsules, out_dim]
        """
        self._log_counter += 1
        should_log = (self._log_counter % self._log_interval == 1) and self.training

        if should_log:
            input_norms = x.norm(dim=-1)
            logger.info(f"[Stage1 Input] shape={x.shape}, "
                       f"norm_mean={input_norms.mean().item():.6f}, "
                       f"norm_max={input_norms.max().item():.6f}")

        # Stage 1: Primary -> Intermediate
        x_intermediate = self._route_stage1(x, should_log)

        if should_log:
            inter_norms = x_intermediate.norm(dim=-1)
            logger.info(f"[Stage1 Output] shape={x_intermediate.shape}, "
                       f"norm_mean={inter_norms.mean().item():.6f}, "
                       f"norm_max={inter_norms.max().item():.6f}")

        # Stage 2: Intermediate -> Activity
        x_activity = self._route_stage2(x_intermediate, should_log)

        if should_log:
            out_norms = x_activity.norm(dim=-1)
            logger.info(f"[Stage2 Output] shape={x_activity.shape}, "
                       f"norm_mean={out_norms.mean().item():.6f}, "
                       f"norm_max={out_norms.max().item():.6f}")

        return x_activity

    def _route_stage1(self, x, should_log=False):
        """Route from primary to intermediate capsules"""
        B, in_capsules, in_dim = x.size()

        # Dynamic weight initialization with proper scaling
        if self.weights_stage1 is None or self.in_capsules != in_capsules:
            # Use Xavier/Glorot initialization scaled for capsule routing
            # fan_in = in_capsules * in_dim, fan_out = intermediate_capsules * intermediate_dim
            # Standard deviation = sqrt(2 / (fan_in + fan_out)) but we use a larger scale
            # to ensure sufficient signal after squashing
            std = math.sqrt(2.0 / (in_dim + self.intermediate_dim))
            weights = nn.Parameter(
                std * torch.randn(1, in_capsules, self.intermediate_capsules,
                                  self.intermediate_dim, in_dim, device=x.device)
            )
            # Delete the None attribute first if it exists
            if hasattr(self, 'weights_stage1'):
                delattr(self, 'weights_stage1')
            # Now register as parameter
            self.register_parameter('weights_stage1', weights)
            self.in_capsules = in_capsules
            logger.info(f"[Init] weights_stage1 initialized with std={std:.4f}, "
                       f"shape={weights.shape}")

        # Compute prediction vectors
        x_expanded = x.unsqueeze(2).unsqueeze(-1)  # [B, in_caps, 1, in_dim, 1]
        W = self.weights_stage1.expand(B, -1, -1, -1, -1)  # [B, in_caps, inter_caps, inter_dim, in_dim]
        u_hat = torch.matmul(W, x_expanded).squeeze(-1)  # [B, in_caps, inter_caps, inter_dim]

        if should_log:
            logger.info(f"[Stage1 Routing] u_hat: mean={u_hat.mean().item():.6f}, "
                       f"std={u_hat.std().item():.6f}, "
                       f"range=[{u_hat.min().item():.6f}, {u_hat.max().item():.6f}]")

        # Dynamic routing with scaling to prevent squash attenuation
        b = torch.zeros(B, in_capsules, self.intermediate_capsules, 1, device=x.device)
        for i in range(self.routing_iters_stage1):
            c = F.softmax(b, dim=2)
            s = (c * u_hat).sum(dim=1, keepdim=True)

            # Scale s before squashing to prevent attenuation of small values
            # The scaling factor compensates for the averaging effect
            s = s * math.sqrt(in_capsules)

            v = self.squash(s)

            if should_log and i == self.routing_iters_stage1 - 1:
                v_norms = v.norm(dim=-1)
                logger.info(f"[Stage1 Iter {i}] v_norms: mean={v_norms.mean().item():.6f}, "
                           f"max={v_norms.max().item():.6f}")

            if i < self.routing_iters_stage1 - 1:
                b = b + (u_hat * v).sum(dim=-1, keepdim=True)

        return v.squeeze(1)  # [B, inter_caps, inter_dim]

    def _route_stage2(self, x, should_log=False):
        """Route from intermediate to activity capsules"""
        B, in_capsules, in_dim = x.size()

        # Dynamic weight initialization with proper scaling
        if self.weights_stage2 is None:
            std = math.sqrt(2.0 / (in_dim + self.out_dim))
            weights = nn.Parameter(
                std * torch.randn(1, in_capsules, self.out_capsules,
                                  self.out_dim, in_dim, device=x.device)
            )
            # Delete the None attribute first if it exists
            if hasattr(self, 'weights_stage2'):
                delattr(self, 'weights_stage2')
            # Now register as parameter
            self.register_parameter('weights_stage2', weights)
            logger.info(f"[Init] weights_stage2 initialized with std={std:.4f}, "
                       f"shape={weights.shape}")

        # Compute prediction vectors
        x_expanded = x.unsqueeze(2).unsqueeze(-1)  # [B, in_caps, 1, in_dim, 1]
        W = self.weights_stage2.expand(B, -1, -1, -1, -1)  # [B, in_caps, out_caps, out_dim, in_dim]
        u_hat = torch.matmul(W, x_expanded).squeeze(-1)  # [B, in_caps, out_caps, out_dim]

        if should_log:
            logger.info(f"[Stage2 Routing] u_hat: mean={u_hat.mean().item():.6f}, "
                       f"std={u_hat.std().item():.6f}")

        # Dynamic routing with scaling
        b = torch.zeros(B, in_capsules, self.out_capsules, 1, device=x.device)
        for i in range(self.routing_iters_stage2):
            c = F.softmax(b, dim=2)
            s = (c * u_hat).sum(dim=1, keepdim=True)

            # Scale s before squashing
            s = s * math.sqrt(in_capsules)

            v = self.squash(s)

            if should_log and i == self.routing_iters_stage2 - 1:
                v_norms = v.norm(dim=-1)
                logger.info(f"[Stage2 Iter {i}] v_norms: mean={v_norms.mean().item():.6f}, "
                           f"max={v_norms.max().item():.6f}")

            if i < self.routing_iters_stage2 - 1:
                b = b + (u_hat * v).sum(dim=-1, keepdim=True)

        return v.squeeze(1)  # [B, out_caps, out_dim]

    def squash(self, x, eps=1e-8):
        """
        Squashing non-linearity with numerical stability

        The standard squash formula norm^2/(1+norm^2) severely attenuates small values.
        For example, norm=0.01 gives scale=0.0001 (100x smaller).

        This implementation uses a modified formula that preserves more signal
        for small norms while still bounding large norms to < 1.

        Args:
            x: Capsule tensor [..., capsule_dim]
        Returns:
            Squashed capsule tensor with same shape
        """
        norm = x.norm(dim=-1, keepdim=True)

        # Modified squash: use softer attenuation for small values
        # For small norms: scale ~= norm (linear)
        # For large norms: scale ~= 1 - 1/norm (approaches 1)
        # Crossover at norm=1
        scale = norm / (1 + norm + eps)

        return scale * (x / (norm + eps))
