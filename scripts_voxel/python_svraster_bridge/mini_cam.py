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
        # self.fovx = 2 * math.atan(tanfovx)  
        # self.fovy = 2 * math.atan(tanfovy)  
        self.cx = cx
        self.cy = cy
        self.cam_mode = cam_mode
        self.frame_id = -1