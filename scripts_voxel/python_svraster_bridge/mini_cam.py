import math
import torch

class MiniCam:
    def __init__(self, width, height, w2c, c2w, tanfovx, tanfovy, cx, cy, cam_mode="persp"):
        self.image_width = width
        self.image_height = height
        self.w2c = w2c
        self.c2w = c2w
        self.tanfovx = tanfovx
        self.tanfovy = tanfovy
        self.cx = cx
        self.cy = cy
        self.cam_mode = cam_mode
        self.frame_id = -1
        # ---------- NEW ----------
        # world‑space camera centre ⟵ 4th column of c2w
        self.position = self.c2w[:3, 3]
        # **+Z** column of c2w is the *forward* (look‑at) direction
        self.lookat   = self.c2w[:3, 2]
        self.pix_size = 2 * self.tanfovx / self.image_width