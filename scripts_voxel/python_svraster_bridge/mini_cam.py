import math
import torch

class MiniCam:
    def __init__(self, width, height, w2c, c2w, tanfovx, tanfovy, cx, cy, cam_mode="persp"):
        # --- coercion to *plain* Python scalars ---------------------------
        self.image_width  = int(width)
        self.image_height = int(height)
        self.cx           = float(cx)
        self.cy           = float(cy)
        self.tanfovx      = float(tanfovx)
        self.tanfovy      = float(tanfovy)
        # ------------------------------------------------------------------
        self.fovx = 2 * math.atan(self.tanfovx)
        self.fovy = 2 * math.atan(self.tanfovy)
        # matrices stay tensors
        self.w2c = w2c
        self.c2w = c2w
        self.cam_mode = cam_mode
        self.frame_id = -1