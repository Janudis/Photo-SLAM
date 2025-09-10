#!/usr/bin/env python3
import os, sys, torch
import numpy as np
import matplotlib.pyplot as plt

# --- make svraster importable ---
SVRASTER = "/home/dimitris/svraster"
if SVRASTER not in sys.path:
    sys.path.insert(0, SVRASTER)

# --- imports (svraster only) ---
from src.sparse_voxel_model import SparseVoxelModel            # high-level init
from src.utils import octree_utils as ou                       # low-level utils
import svraster_cuda.utils as cu                               # packed key helpers (ijk<->octpath)

# --- device ---
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if not use_cuda:
    print("[WARN] CUDA not available; this still runs, but SVM is meant for CUDA.")

# --- toy points (spread in all directions) ---
xyz_np = np.array([
    [-0.20,  0.00,  0.00],
    [ 0.20,  0.00,  0.00],
    [ 0.00, -0.20,  0.00],
    [ 0.00,  0.20,  0.00],
    [ 0.00,  0.00,  0.20],
    [ 0.00,  0.00, -0.20],
    [ 0.15,  0.10,  0.05],
    [-0.12, -0.08,  0.06],
], dtype=np.float32)
xyz = torch.from_numpy(xyz_np).to(device)

# simple dummy appearance (not used for the key demo)
rgb = torch.ones_like(xyz, device=device) * 0.8

# --- FRAME A: initial bound + level ---
centerA = torch.tensor([0.0, 0.0, 0.0], device=device)
extentA = torch.tensor([0.408], device=device)   # pick something small
L = 6                                           # initial octree level
vox_size_A = ou.level_2_vox_size(extentA, torch.tensor([[L]], device=device)).squeeze()

svmA = SparseVoxelModel(sh_degree=1, black_background=True)
svmA.points_init(
    scene_center=centerA,
    scene_extent=extentA,
    xyz=xyz,
    expected_vox_size=vox_size_A,   # keep level L
    rgb=rgb,
    shs=0.0,
    density=-10.0
)

octpathA = svmA.octpath.detach().view(-1).cpu()        # [N]
octlevelA= svmA.octlevel.detach().view(-1).cpu()       # [N]
centersA = svmA.vox_center.detach().cpu().numpy()      # [N,3]
sizesA   = svmA.vox_size.detach().view(-1).cpu().numpy()
ijkA     = cu.octpath_2_ijk(svmA.octpath, svmA.octlevel).detach().cpu().numpy()

def hex64(t):
    v = int(t.item())
    return f"0x{v:016x}"

print("\n=== FRAME A (initial) ===")
for i in range(len(xyz_np)):
    print(f"{i:2d}  {xyz_np[i,0]: .2f}  {xyz_np[i,1]: .2f}  {xyz_np[i,2]: .2f}  "
          f"L={int(octlevelA[i]):2d}  {hex64(octpathA[i])}  "
          f"cx={centersA[i,0]: .6f}  cy={centersA[i,1]: .6f}  cz={centersA[i,2]: .6f}  "
          f"sz={sizesA[i]: .6f}  ijk={ijkA[i]}")

# --- FRAME B1: expand bound, keep SAME level L ---
extentB = extentA * 3.0
vox_size_B_sameL = ou.level_2_vox_size(extentB, torch.tensor([[L]], device=device)).squeeze()

svmB1 = SparseVoxelModel(sh_degree=1, black_background=True)
svmB1.points_init(
    scene_center=centerA,             # same center (for clarity)
    scene_extent=extentB,             # expanded bound
    xyz=xyz,
    expected_vox_size=vox_size_B_sameL,   # this keeps L == 6
    rgb=rgb,
    shs=0.0,
    density=-10.0
)
octpathB1 = svmB1.octpath.detach().view(-1).cpu()
octlevelB1= svmB1.octlevel.detach().view(-1).cpu()
centersB1 = svmB1.vox_center.detach().cpu().numpy()
sizesB1   = svmB1.vox_size.detach().view(-1).cpu().numpy()

print("\n=== FRAME B1 (expanded bound, SAME L) ===")
for i in range(len(xyz_np)):
    print(f"{i:2d}  {xyz_np[i,0]: .2f}  {xyz_np[i,1]: .2f}  {xyz_np[i,2]: .2f}  "
          f"L={int(octlevelB1[i]):2d}  {hex64(octpathB1[i])}  "
          f"cx={centersB1[i,0]: .6f}  cy={centersB1[i,1]: .6f}  cz={centersB1[i,2]: .6f}  "
          f"sz={sizesB1[i]: .6f}")

preserved_sameL = ((octpathA == octpathB1) & (octlevelA == octlevelB1)).sum().item()
print(f"\n[Match A→B1 (same L)] preserved: {preserved_sameL}/{len(xyz_np)}")
for i in range(len(xyz_np)):
    if not (octpathA[i] == octpathB1[i] and octlevelA[i] == octlevelB1[i]):
        print(f"  i={i}: A {hex64(octpathA[i])} -> B1 {hex64(octpathB1[i])}")

# --- FRAME B2: expand bound, keep SAME physical voxel size (=> L changes) ---
svmB2 = SparseVoxelModel(sh_degree=1, black_background=True)
svmB2.points_init(
    scene_center=centerA,
    scene_extent=extentB,             # expanded bound
    xyz=xyz,
    expected_vox_size=vox_size_A,     # same physical size as A -> different L
    rgb=rgb,
    shs=0.0,
    density=-10.0
)
octpathB2 = svmB2.octpath.detach().view(-1).cpu()
octlevelB2= svmB2.octlevel.detach().view(-1).cpu()
centersB2 = svmB2.vox_center.detach().cpu().numpy()
sizesB2   = svmB2.vox_size.detach().view(-1).cpu().numpy()

print("\n=== FRAME B2 (expanded bound, SAME physical size => re-round L) ===")
for i in range(len(xyz_np)):
    print(f"{i:2d}  {xyz_np[i,0]: .2f}  {xyz_np[i,1]: .2f}  {xyz_np[i,2]: .2f}  "
          f"L={int(octlevelB2[i]):2d}  {hex64(octpathB2[i])}  "
          f"cx={centersB2[i,0]: .6f}  cy={centersB2[i,1]: .6f}  cz={centersB2[i,2]: .6f}  "
          f"sz={sizesB2[i]: .6f}")

preserved_rr = ((octpathA == octpathB2) & (octlevelA == octlevelB2)).sum().item()
print(f"\n[Match A→B2 (same physical size)] preserved: {preserved_rr}/{len(xyz_np)}")
for i in range(len(xyz_np)):
    if not (octpathA[i] == octpathB2[i] and octlevelA[i] == octlevelB2[i]):
        print(f"  i={i}: A L={int(octlevelA[i])}, {hex64(octpathA[i])} "
              f"-> B2 L={int(octlevelB2[i])}, {hex64(octpathB2[i])}")

np.set_printoptions(precision=6, suppress=True)
print("\ncentersA:\n", centersA)
print("centersB1 (same L):\n", centersB1)
print("centersB2 (same physical size):\n", centersB2)

# --- Quick 2D visualization of centers (A vs B1/B2) ---
fig, axs = plt.subplots(1, 2, figsize=(10,4))
axs[0].scatter(centersA[:,0], centersA[:,1], s=40, label="A")
axs[0].scatter(centersB1[:,0], centersB1[:,1], s=40, marker='x', label="B1 (same L)")
axs[0].axis('equal'); axs[0].grid(True); axs[0].legend(); axs[0].set_title("A vs B1")

axs[1].scatter(centersA[:,0], centersA[:,1], s=40, label="A")
axs[1].scatter(centersB2[:,0], centersB2[:,1], s=40, marker='x', label="B2 (same size)")
axs[1].axis('equal'); axs[1].grid(True); axs[1].legend(); axs[1].set_title("A vs B2")

plt.tight_layout()
plt.show()
