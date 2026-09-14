"""Live viewer window, so you can watch the fly instead of reading numbers.

Wraps `mujoco.viewer.launch_passive`: the window renders whatever is currently
in `mj_data` while our own loop drives the physics. Paced to wall-clock so the
motion plays at real speed rather than as fast as the CPU manages.

Degrades to a no-op if no display is available, so scripts still run headless.
"""

from __future__ import annotations

import time


class Viewer:
    """Passive MuJoCo viewer with real-time pacing and a camera on the fly."""

    def __init__(self, sim, *, enabled: bool = True, track: str | None = "fly/c_thorax",
                 speed: float = 1.0, distance: float = 12.0):
        self.sim = sim
        self.speed = speed
        self.handle = None
        self._t0 = None
        self._sim_t0 = None

        if not enabled:
            return
        try:
            import mujoco
            import mujoco.viewer

            self.handle = mujoco.viewer.launch_passive(
                sim.mj_model, sim.mj_data, show_left_ui=False, show_right_ui=False
            )
            cam = self.handle.cam
            cam.distance = distance
            cam.elevation = -20.0
            cam.azimuth = 135.0
            if track:
                bid = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, track)
                if bid >= 0:
                    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    cam.trackbodyid = bid
        except Exception as exc:                      # no display, no viewer
            print(f"  (viewer unavailable: {type(exc).__name__}: {exc})")
            self.handle = None

    @property
    def active(self) -> bool:
        return self.handle is not None and self.handle.is_running()

    def sync(self, sim_time_s: float) -> bool:
        """Redraw and sleep so playback tracks wall-clock. False if closed."""
        if self.handle is None:
            return True
        if not self.handle.is_running():
            return False

        if self._t0 is None:
            self._t0, self._sim_t0 = time.time(), sim_time_s

        self.handle.sync()
        target = (sim_time_s - self._sim_t0) / max(self.speed, 1e-6)
        lag = target - (time.time() - self._t0)
        if lag > 0:
            time.sleep(min(lag, 0.1))
        return True

    def close(self):
        """Shut the window down before MuJoCo frees the model.

        The viewer renders on its own thread holding references to mj_model
        and mj_data. Letting the interpreter tear those down while that thread
        is still alive segfaults, so close explicitly and give it a moment.
        """
        if self.handle is None:
            return
        handle, self.handle = self.handle, None
        try:
            handle.close()
        except Exception:
            pass
        time.sleep(0.2)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
