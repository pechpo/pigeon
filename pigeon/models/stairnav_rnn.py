from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union, Optional

import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete

@dataclass
class WrappedStairNavHBPolicy:
    """Minimal Habitat-Baselines PointNav wrapper.

    - Loads a PointNavResNetPolicy checkpoint saved by habitat-baselines.
    - Maintains hidden state per instance for single-env inference.
    - Exposes act() that takes depth and an explicit pointgoal (rho, theta).
    """

    ckpt_path: str
    device: Union[str, torch.device] = "cuda"

    def __post_init__(self) -> None:
        from .stairnav_policy import ModPointNavResNetPolicy
        from habitat_baselines.common.tensor_dict import TensorDict
        from habitat_baselines.rl.ppo.policy import PolicyActionData

        self._TensorDict = TensorDict  # type: ignore[attr-defined]
        self._PolicyActionData = PolicyActionData  # type: ignore[attr-defined]
        self._PointNavResNetPolicy = ModPointNavResNetPolicy

        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        # Build observation/action space to construct the policy
        obs_space = SpaceDict(
            {
                "depth": spaces.Box(low=0.0, high=1.0, shape=(224, 224, 1), dtype=np.float32),
                "stairgoal": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "pointgoal_with_gps_compass": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        action_space = Discrete(4)

        from .checkpoint_compat import install_vlfm_checkpoint_aliases

        install_vlfm_checkpoint_aliases()
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        # Newer habitat-baselines checkpoints typically contain a nested config
        self.policy = self._PointNavResNetPolicy.from_config(ckpt["config"], obs_space, action_space)
        self.policy.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
        self.policy.to(self.device)
        self.policy.eval()

        # Hidden and prev action buffers
        self._hidden = torch.zeros(
            1,
            self.policy.net.num_recurrent_layers,
            512,
            device=self.device,
        )
        self._prev_action = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        self._step = 0

    def reset(self) -> None:
        # must use zeros_like to avoid error
        self._hidden = torch.zeros_like(self._hidden)
        self._prev_action = torch.zeros_like(self._prev_action)
        self._step = 0

    def get_state(self) -> Dict[str, Any]:
        """Return a snapshot of the current recurrent state for external storage.

        Returns:
            Dict[str, Any]: Contains 'hidden', 'prev_action' tensors and 'step' int.
        """
        return {
            "hidden": self._hidden.clone(),
            "prev_action": self._prev_action.clone(),
            "step": int(self._step),
        }

    def set_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore a previously saved recurrent state. If None, resets state."""
        if state is None:
            self.reset()
            return
        # Move to correct device and assign
        self._hidden = state["hidden"].to(self.device).clone()
        self._prev_action = state["prev_action"].to(self.device).clone()
        self._step = int(state.get("step", 0))

    @torch.inference_mode()
    def act(self, depth: np.ndarray, pointgoal_rho_theta: np.ndarray, deterministic: bool = False) -> int:
        """pointgoal_rho_theta should be np.array([1, 0]) if go upstairs, otherwise np.array([-1, 0])
        """
        # Depth preprocessing to [1, 224, 224, 1] in [0,1]
        d = depth.astype(np.float32)
        # Normalize by a cap to [0,1]
        denom = float(np.nanmax(d) or 5.0)
        denom = max(denom, 1e-6)
        d = np.clip(d / denom, 0.0, 1.0)
        if d.ndim == 2:
            d = d[..., None]
        # Resize to 224x224 if needed
        if d.shape[0] != 224 or d.shape[1] != 224:
            import cv2

            d = cv2.resize(d, (224, 224), interpolation=cv2.INTER_AREA)
            if d.ndim == 2:
                d = d[..., None]

        from habitat_baselines.common.tensor_dict import TensorDict

        obs = TensorDict(
            {
                "depth": torch.from_numpy(d).to(self.device).unsqueeze(0),  # [1,224,224,1]
                "stairgoal": torch.from_numpy(pointgoal_rho_theta.astype(np.float32)).to(self.device).view(1, 2),
            }
        )

        masks = torch.tensor([[0 if self._step == 0 else 1]], dtype=torch.bool, device=self.device)
        self._step += 1

        out = self.policy.act(  # type: ignore[attr-defined]
            obs, self._hidden, self._prev_action, masks, deterministic=deterministic
        )
        # PolicyActionData on recent habitat-baselines
        try:
            action_tensor = out.actions  # type: ignore[attr-defined]
            self._hidden = out.rnn_hidden_states  # type: ignore[attr-defined]
        except Exception:
            # Older API compatibility (value, action, logprobs, hidden)
            _, action_tensor, _, self._hidden = out

        self._prev_action = action_tensor.clone().to(self.device)
        return int(action_tensor.view(-1)[0].item())
