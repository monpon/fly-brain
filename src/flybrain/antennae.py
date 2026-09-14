"""Feeling the walls: antennal mechanosensation, and what to do about it.

A walking Drosophila runs into things constantly, and the sensor it uses is
the antenna. Bristles on the pedicel and funiculus, plus the Johnston's organ
measuring how far the funiculus is deflected, report contact and its
direction; the signal goes to the antennal mechanosensory and motor centre and
out through descending neurons that turn the fly. Blind, antennaless flies
stop wall-following. This is a genuine, separate sensory channel from
olfaction that happens to live on the same organ.

Here it is read as the contact force on the antennal collision proxies --
`body.add_wall_proxies` puts a small sphere on each funiculus, and those
spheres reach further forward than any other part of the fly, so the antenna
meets a wall before the head does. FlyGym's own
`get_bodysegment_contact_forces` does not see these contacts (it reports zero
while MuJoCo's contact list plainly shows `wallproxy_l_funiculus` against
`maze_wall_2`), so the contact array is read directly. The reflex it drives *preempts* odour steering rather
than being added to it, which is the right structure -- a fly with its
antenna against a wall turns away first and resumes tracking afterwards. The
odour gradient has nothing useful to say about a wall, because the wall is not
made of smell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEFT_PROXY = "wallproxy_l_funiculus"
RIGHT_PROXY = "wallproxy_r_funiculus"
HEAD_PROXY = "wallproxy_c_head"


@dataclass
class ReflexParams:
    threshold: float = 1e-4    # contact force counting as a touch
    tau_ms: float = 40.0       # smoothing, roughly the mechanosensory lag
    turn: float = 0.9          # how hard to turn away from a touched wall
    drive: float = 0.35        # slow down while in contact -- see below
    hold_ms: float = 250.0     # keep turning briefly after contact is lost


class WallReflex:
    """Antennal contact -> a turn away from the wall.

    Slowing down during the reflex is not cosmetic. This body turns at about
    75 deg/s, so at its full 15 mm/s the turning circle is ~23 mm across --
    wider than a maze corridor. It physically cannot round a corner at speed;
    it has to slow down, exactly as a real fly does when it turns sharply.
    """

    def __init__(self, sim, fly_name: str = "fly",
                 params: ReflexParams | None = None):
        import mujoco

        self.sim = sim
        self.fly_name = fly_name
        self.p = params or ReflexParams()
        self.left = 0.0
        self.right = 0.0
        self.hold = 0.0
        self.hold_sign = 0.0
        self.touching = False
        self._buf = np.zeros(6)

        def gid(name):
            return mujoco.mj_name2id(
                sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"{fly_name}/{name}"
            )

        # A head-on contact registers on both sides, which is what makes the
        # reflex commit to a direction instead of dithering.
        self.side_of = {}
        for name, sides in ((LEFT_PROXY, ("l",)), (RIGHT_PROXY, ("r",)),
                            (HEAD_PROXY, ("l", "r"))):
            i = gid(name)
            if i >= 0:
                self.side_of[i] = sides

    def _raw(self) -> tuple[float, float]:
        import mujoco

        d = self.sim.mj_data
        left = right = 0.0
        for c in range(d.ncon):
            sides = (self.side_of.get(int(d.contact.geom1[c]))
                     or self.side_of.get(int(d.contact.geom2[c])))
            if sides is None:
                continue
            mujoco.mj_contactForce(self.sim.mj_model, d, c, self._buf)
            f = float(np.linalg.norm(self._buf[:3]))
            if "l" in sides:
                left += f
            if "r" in sides:
                right += f
        return left, right

    def sense(self, dt_ms: float) -> tuple[float, float]:
        alpha = float(np.exp(-dt_ms / max(self.p.tau_ms, 1e-6)))
        raw_l, raw_r = self._raw()
        self.left = alpha * self.left + (1 - alpha) * raw_l
        self.right = alpha * self.right + (1 - alpha) * raw_r
        return self.left, self.right

    def override(self, dt_ms: float) -> tuple[float, float] | None:
        """A (turn, drive) pair if the reflex is active, else None."""
        left, right = self.sense(dt_ms)
        p = self.p
        hit_l, hit_r = left > p.threshold, right > p.threshold
        self.touching = hit_l or hit_r

        if hit_l or hit_r:
            if hit_l and hit_r:
                # Nose-on into a dead end. Commit to one side rather than
                # dithering: whichever antenna is pressed less hard.
                sign = 1.0 if left >= right else -1.0
            else:
                sign = 1.0 if hit_l else -1.0   # positive turn steers right
            self.hold = p.hold_ms
            self.hold_sign = sign
            return sign * p.turn, p.drive

        if self.hold > 0.0:
            self.hold -= dt_ms
            return self.hold_sign * p.turn * 0.6, min(1.0, p.drive * 2.0)
        return None
