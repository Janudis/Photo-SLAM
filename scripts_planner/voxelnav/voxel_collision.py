import time
import numpy as np
import torch
from scipy.spatial import KDTree

# --------- low-level collision tests ----------

def sphere_intersects_aabb(p: torch.Tensor, c: torch.Tensor, h: torch.Tensor, r: float) -> torch.Tensor:
    """
    Correct sphere-AABB intersection test.

    Sphere centered at p with radius r intersects an AABB centered at c with half-size h
    iff distance(p, AABB(c,h)) <= r.

    p: [3]
    c: [K,3]
    h: [K] (half-size, scalar per voxel)
    returns: [K] bool
    """
    # Box min/max
    h3 = h.view(-1, 1)
    bmin = c - h3
    bmax = c + h3

    # Clamp p to the box to find closest point on/in box
    p3 = p.view(1, 3)
    q = torch.maximum(bmin, torch.minimum(p3, bmax))  # [K,3]

    # Squared distance from p to closest point q
    d2 = torch.sum((q - p3) ** 2, dim=-1)

    return d2 <= (float(r) * float(r))


def segment_intersects_aabb(p0: torch.Tensor, p1: torch.Tensor, c: torch.Tensor, h: torch.Tensor, r: float) -> torch.Tensor:
    """
    Segment-AABB intersection (slab method) with Minkowski expansion by sphere radius r.

    Segment p(t)=p0 + t*(p1-p0), t in [0,1]
    Box is centered at c with half-size h, expanded by r.
    """
    he = h.view(-1, 1) + float(r)
    mn = c - he
    mx = c + he

    d = (p1 - p0).view(1, 3)  # [1,3]
    p0v = p0.view(1, 3)       # [1,3]

    eps = 1e-12
    inv = 1.0 / torch.where(torch.abs(d) < eps, torch.ones_like(d), d)

    t1 = (mn - p0v) * inv
    t2 = (mx - p0v) * inv

    tmin = torch.minimum(t1, t2)
    tmax = torch.maximum(t1, t2)

    # If d==0 for an axis, need p0 within slab; otherwise no hit.
    parallel = (torch.abs(d) < eps)
    within = (p0v >= mn) & (p0v <= mx)
    axis_ok = torch.where(parallel, within, torch.ones_like(within, dtype=torch.bool))

    t_enter = torch.max(tmin, dim=-1).values
    t_exit  = torch.min(tmax, dim=-1).values

    hit = (t_exit >= t_enter) & (t_exit >= 0.0) & (t_enter <= 1.0) & torch.all(axis_ok, dim=-1)
    return hit


# --------- OMPL wrappers (SplatNav style) ----------

class VoxelEVC:
    """
    State validity checker:
    valid iff robot sphere does NOT intersect any nearby occupied voxel AABB.
    """
    def __init__(self, voxmap, robot_radius: float, device: torch.device):
        self.device = device
        self.robot_radius = float(robot_radius)

        self.centers = voxmap.centers
        self.half_sizes = voxmap.half_sizes

        self.kdtree: KDTree = voxmap.kdtree

        # IMPORTANT: radius to collect candidate voxels should cover the AABB's bounding sphere.
        # AABB bounding sphere radius = sqrt(3) * h
        self.search_radius = float(np.sqrt(3.0) * voxmap.max_half_size + self.robot_radius)

        self.point_count = 0
        self.times_point = []

    def is_valid_point(self, p_np: np.ndarray) -> bool:
        self.point_count += 1

        neigh = self.kdtree.query_ball_point(p_np, self.search_radius)
        if len(neigh) == 0:
            return True

        idx = torch.as_tensor(neigh, device=self.device, dtype=torch.long)
        c = self.centers.index_select(0, idx)
        h = self.half_sizes.index_select(0, idx)

        p = torch.tensor(p_np, device=self.device, dtype=torch.float32)

        t0 = time.time()
        hit = sphere_intersects_aabb(p, c, h, self.robot_radius)
        self.times_point.append(time.time() - t0)

        return not bool(torch.any(hit).item())


class VoxelEMV:
    def __init__(self, voxmap, robot_radius: float, device: torch.device):
        self.device = device
        self.robot_radius = float(robot_radius)

        self.centers = voxmap.centers
        self.half_sizes = voxmap.half_sizes

        self.kdtree: KDTree = voxmap.kdtree
        self.base_radius = float(np.sqrt(3.0) * voxmap.max_half_size + self.robot_radius)

        self.line_count = 0
        self.times_line = []

        # NEW: sampling step along segment for candidate collection
        self.sample_step = float(2.0 * voxmap.max_half_size)  # ~ one voxel diagonal scale

    def is_valid_segment(self, p0_np: np.ndarray, p1_np: np.ndarray) -> bool:
        self.line_count += 1

        seg = p1_np - p0_np
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-9:
            # fallback to point check-ish
            neigh = self.kdtree.query_ball_point(p0_np, self.base_radius)
            if len(neigh) == 0:
                return True
            idx = torch.as_tensor(neigh, device=self.device, dtype=torch.long)
            c = self.centers.index_select(0, idx)
            h = self.half_sizes.index_select(0, idx)
            p0 = torch.tensor(p0_np, device=self.device, dtype=torch.float32)
            p1 = torch.tensor(p1_np, device=self.device, dtype=torch.float32)
            hit = segment_intersects_aabb(p0, p1, c, h, self.robot_radius)
            return not bool(torch.any(hit).item())

        # number of samples along segment (including endpoints)
        n = int(np.ceil(seg_len / self.sample_step)) + 1
        n = max(2, min(n, 64))  # cap to avoid pathological long edges
        ts = np.linspace(0.0, 1.0, n, dtype=np.float32)

        # gather candidate voxel indices locally around sampled points
        neigh_set = set()
        r = self.base_radius
        for t in ts:
            pt = p0_np + t * seg
            neigh = self.kdtree.query_ball_point(pt, r)
            neigh_set.update(neigh)

        if len(neigh_set) == 0:
            return True

        idx = torch.as_tensor(list(neigh_set), device=self.device, dtype=torch.long)
        c = self.centers.index_select(0, idx)
        h = self.half_sizes.index_select(0, idx)

        p0 = torch.tensor(p0_np, device=self.device, dtype=torch.float32)
        p1 = torch.tensor(p1_np, device=self.device, dtype=torch.float32)

        t0 = time.time()
        hit = segment_intersects_aabb(p0, p1, c, h, self.robot_radius)
        self.times_line.append(time.time() - t0)

        return not bool(torch.any(hit).item())

