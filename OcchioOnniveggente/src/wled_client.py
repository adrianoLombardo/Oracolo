import requests

class WLED:
    def __init__(self, host: str):
        self.base = f"http://{host}"

    def set_color(self, r, g, b, brightness=128):
        payload = {"on": True, "bri": int(brightness), "seg": [{"id": 0, "col": [[r, g, b]]}]}
        requests.post(self.base + "/json/state", json=payload, timeout=1.0)

    def pulse_by_level(self, level_0_1, base_rgb=(180,180,200), min_bri=20, max_bri=255):
        bri = int(min_bri + (max_bri - min_bri) * max(0.0, min(1.0, level_0_1)))
        r, g, b = base_rgb
        payload = {"on": True, "bri": bri, "seg":[{"id":0, "col":[[r, g, b]]}]}
        requests.post(self.base + "/json/state", json=payload, timeout=0.5)
