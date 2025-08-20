from __future__ import annotations

from typing import Any, Dict, Tuple

import logging
import numpy as np
import sacn
from requests import exceptions as req_exc

from .wled_client import WLED

logger = logging.getLogger(__name__)


class SacnLight:
    def __init__(self, conf: Dict[str, Any]):
        sacn_conf = conf["sacn"]
        self.universe = int(sacn_conf["universe"])
        self.dest_ip = sacn_conf["destination_ip"]
        self.rgb = sacn_conf["rgb_channels"]
        self.idle_level = int(sacn_conf["idle_level"])
        self.peak_level = int(sacn_conf["peak_level"])
        self.sender = sacn.sACNsender()
        self.sender.start()
        self.sender.activate_output(self.universe)
        self.sender[self.universe].multicast = False
        self.sender[self.universe].destination = self.dest_ip
        self.frame = [0] * 512
        self.idle()

    def set_rgb(self, r: int, g: int, b: int) -> None:
        r = int(np.clip(r, 0, 255))
        g = int(np.clip(g, 0, 255))
        b = int(np.clip(b, 0, 255))
        self.frame[self.rgb[0] - 1] = r
        self.frame[self.rgb[1] - 1] = g
        self.frame[self.rgb[2] - 1] = b
        self.sender[self.universe].dmx_data = tuple(self.frame)

    def idle(self) -> None:
        self.set_rgb(self.idle_level, self.idle_level, self.idle_level)

    def blackout(self) -> None:
        self.set_rgb(0, 0, 0)

    def stop(self) -> None:
        try:
            self.sender.stop()
        except Exception:
            pass


class WledLight:
    def __init__(self, conf: Dict[str, Any]):
        host = conf["wled"]["host"]
        self.w = WLED(host)
        self.base_rgb = (180, 180, 200)
        try:
            self.w.set_color(*self.base_rgb, brightness=40)
        except req_exc.RequestException as exc:
            logger.warning("WLED set_color failed: %s", exc)

    def set_base_rgb(self, rgb: Tuple[int, int, int]) -> None:
        self.base_rgb = tuple(int(x) for x in rgb)

    def pulse(self, level: float) -> None:
        try:
            self.w.pulse_by_level(level, base_rgb=self.base_rgb)
        except req_exc.RequestException as exc:
            logger.warning("WLED pulse failed: %s", exc)

    def idle(self) -> None:
        try:
            self.w.set_color(*self.base_rgb, brightness=30)
        except req_exc.RequestException as exc:
            logger.warning("WLED set_color failed: %s", exc)

    def blackout(self) -> None:
        try:
            self.w.set_color(0, 0, 0, brightness=0)
        except req_exc.RequestException as exc:
            logger.warning("WLED set_color failed: %s", exc)

    def stop(self) -> None:
        pass


def color_from_text(text: str, palettes: Dict[str, Dict[str, Any]]) -> Tuple[int, int, int]:
    t = text.lower()
    for kw, cfg in palettes.items():
        if kw in t:
            return tuple(cfg["rgb"])
    return (180, 180, 200)

