###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Hardware-in-the-loop channel layer.

Mirrors the Channel-Access layer of accelerator control systems (EPICS PVs):
a thin ``get``/``put`` interface decouples the *policy* from its *plant*.  A
:class:`SimulatedChannel` wraps one of the package environments (no external
dependency) so trained policies can be exercised through the same interface
that a real EPICS/LLRF backend (e.g. ``pyepics``) would provide; the optional
:class:`EPICSChannel` simply guards the ``epics`` import and is only usable
when the user has ``pyepics`` installed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..registry import make

__all__ = ["Channel", "SimulatedChannel", "EPICSChannel", "emulate_policy_loop"]


@dataclass
class Channel:
    """Abstract channel protocol (a single scalar or complex PV).

    Attributes:
        getter: Read callback returning the current value.
        setter: Optional write callback.
    """

    getter: Callable[[], complex]
    setter: Callable[[complex], None] | None = None

    def get(self) -> complex:
        """Read the current channel value."""
        return self.getter()

    def put(self, value: complex) -> None:
        """Write a channel value (no-op when ``setter`` is ``None``)."""
        if self.setter is not None:
            self.setter(value)

    def __call__(self) -> complex:
        """Convenience alias of :meth:`get`."""
        return self.get()


class SimulatedChannel:
    """A channel view of one package environment.

    Args:
        env_id:      Environment id (constructed lazily).
        get_attr:    Dotted attribute path read by :meth:`get` (default
                     ``"plant.vc"``, i.e. the cavity voltage).
        put_attr:    Optional dotted attribute path written by :meth:`put`.
        env_factory: Optional factory overriding ``env_id`` creation.
    """

    def __init__(
        self,
        env_id: str,
        get_attr: str = "plant.vc",
        put_attr: str | None = None,
        env_factory: Callable | None = None,
    ) -> None:
        self.env_factory = env_factory or (lambda: make(env_id))
        self.get_attr = get_attr
        self.put_attr = put_attr
        self.env = self.env_factory()

    def reset(self, seed: int | None = 0) -> None:
        """Reset the wrapped environment."""
        self.env.reset(seed=seed)

    def _resolve(self, path: str):
        obj = self.env
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj

    def get(self) -> complex:
        """Read the configured attribute (complex-converted)."""
        return complex(self._resolve(self.get_attr))

    def put(self, value: complex) -> None:
        """Write the configured attribute when ``put_attr`` is set."""
        if self.put_attr is None:
            return
        obj = self.env
        parts = self.put_attr.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def step(self, action: np.ndarray) -> dict:
        """Advance the wrapped environment one sample with ``action``.

        Args:
            action: Environment action.

        Returns:
            The ``info`` dict of the step.
        """
        *_, info = self.env.step(np.asarray(action, dtype=np.float32))
        return info


class EPICSChannel:
    """Optional EPICS Channel-Access wrapper (guards the ``epics`` import).

    Args:
        pv_name: EPICS process-variable name.
        dtype:   Value conversion (``"float"`` / ``"complex"``).

    Raises:
        ImportError: If the ``epics`` (pyepics) package is not installed.
    """

    def __init__(self, pv_name: str, dtype: str = "float") -> None:
        try:
            from epics import PV
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "EPICSChannel requires 'pyepics' (pip install pyepics). "
                "Use SimulatedChannel for dependency-free channel emulation."
            ) from exc
        self.pv = PV(pv_name)
        self.dtype = dtype

    def get(self) -> complex:
        """Read the process variable."""
        value = self.pv.get()
        if self.dtype == "complex":
            re_ = float(self.pv.get())
            return complex(re_, 0.0)  # EPICS floats carry a single component
        return complex(float(value), 0.0)

    def put(self, value: complex) -> None:
        """Write the process variable."""
        self.pv.put(float(value.real))


def emulate_policy_loop(
    channel: SimulatedChannel,
    action_fn: Callable[[np.ndarray], np.ndarray],
    steps: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Run a policy through a channel for a few samples (HIL emulation).

    The environment's observation is read from the channel's wrapped env and
    the (scalar) returned trajectory is stored.

    Args:
        channel:   A :class:`SimulatedChannel`.
        action_fn: Policy mapping an observation to an action.
        steps:     Number of emulated samples.
        seed:      Environment seed.

    Returns:
        Dict with ``value`` (per-sample channel values), ``reward`` and
        ``length``.
    """
    channel.reset(seed=seed)
    env = channel.env
    obs, _ = env.reset(seed=seed)
    values = np.zeros(steps, dtype=np.complex128)
    rewards = np.zeros(steps, dtype=np.float64)
    n = 0
    for i in range(steps):
        action = np.asarray(action_fn(obs), dtype=np.float32)
        obs, reward, terminated, truncated, _info = env.step(action)
        values[i] = channel.get()
        rewards[i] = float(reward)
        n += 1
        if terminated or truncated:
            break
    return {"value": values[:n], "reward": rewards[:n], "length": float(n)}
