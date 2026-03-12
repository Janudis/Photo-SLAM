import torch
import svraster_cuda

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

    @property
    def down(self):
        return self.c2w[:3, 1]

    @property
    def right(self):
        return self.c2w[:3, 0]

    def compute_rd(self, wh=None, cxcy=None, device=None):
        if wh is None:
            wh = (self.image_width, self.image_height)
        if cxcy is None:
            cxcy = (self.cx * wh[0] / self.image_width, self.cy * wh[1] / self.image_height)

        rd = svraster_cuda.utils.compute_rd(
            width=wh[0],
            height=wh[1],
            cx=cxcy[0],
            cy=cxcy[1],
            tanfovx=self.tanfovx,
            tanfovy=self.tanfovy,
            c2w_matrix=self.c2w,
        )
        if device is not None:
            rd = rd.to(device)
        return rd

    def project(self, pts, return_depth=False):
        cam_pts = pts @ self.w2c[:3, :3].T + self.w2c[:3, 3]
        depth = cam_pts[:, [2]]
        cam_uv = cam_pts[:, :2] / depth
        cam_uv[:, 0] = cam_uv[:, 0] / self.tanfovx + (2 * self.cx / self.image_width - 1)
        cam_uv[:, 1] = cam_uv[:, 1] / self.tanfovy + (2 * self.cy / self.image_height - 1)
        if return_depth:
            return cam_uv, depth
        return cam_uv

    def depth2pts(self, depth):
        device = depth.device
        h, w = depth.shape[-2:]
        rd = self.compute_rd(wh=(w, h), device=device)
        return self.position.view(3, 1, 1).to(device) + rd * depth

    def depth2normal(self, depth, ks=3, tol_cos=-1):
        assert ks % 2 == 1
        pad = ks // 2
        ks_1 = ks - 1
        pts = self.depth2pts(depth)
        normal_pseudo = torch.zeros_like(pts)
        dx = pts[:, pad:-pad, ks_1:] - pts[:, pad:-pad, :-ks_1]
        dy = pts[:, ks_1:, pad:-pad] - pts[:, :-ks_1, pad:-pad]
        normal_pseudo[:, pad:-pad, pad:-pad] = torch.nn.functional.normalize(
            torch.cross(dx, dy, dim=0), dim=0
        )

        if tol_cos > 0:
            with torch.no_grad():
                pts_dir = torch.nn.functional.normalize(
                    pts - self.position.view(3, 1, 1).to(depth.device), dim=0
                )
                dot = (normal_pseudo * pts_dir).sum(0)
                mask = dot > tol_cos
            normal_pseudo = normal_pseudo * mask

        return normal_pseudo
