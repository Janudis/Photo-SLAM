#include <torch/extension.h>    

// forward & backward declarations (implemented in .cu)
torch::Tensor voxel_rasterize_forward(
        torch::Tensor vox,        // M × (pos, level, sh, density…)
        torch::Tensor K,          // 3×3
        torch::Tensor Twc,        // 4×4
        int H,  int W);

std::vector<torch::Tensor> voxel_rasterize_backward(
        torch::Tensor dL_dI,      // grad from loss w.r.t rendered image
        torch::Tensor vox,
        torch::Tensor K,
        torch::Tensor Twc,
        int H, int W);

// autograd glue
class VoxelRasterizeFunction : public torch::autograd::Function<VoxelRasterizeFunction> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor vox,
                               torch::Tensor K,
                               torch::Tensor Twc,
                               int H, int W) {
      ctx->save_for_backward({vox, K, Twc});
      ctx->saved_data["H"] = H;
      ctx->saved_data["W"] = W;
      return voxel_rasterize_forward(vox, K, Twc, H, W);
  }
  static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext *ctx,
                                             std::vector<torch::Tensor> grad_outputs) {
      auto saved = ctx->get_saved_variables();
      auto vox = saved[0];
      auto K   = saved[1];
      auto Twc = saved[2];
      int H = ctx->saved_data["H"].toInt();
      int W = ctx->saved_data["W"].toInt();
      auto grads = voxel_rasterize_backward(grad_outputs[0], vox, K, Twc, H, W);
      // return same number of tensors as inputs
      return {grads[0], torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

// ---------- wrappers compatible with C++17 -----------------
static torch::Tensor forward_wrapper(
    torch::Tensor vox,
    torch::Tensor K,
    torch::Tensor Twc,
    int64_t       H,
    int64_t       W)
{
// VoxelRasterizeFunction::apply expects exactly those five args
return VoxelRasterizeFunction::apply(vox, K, Twc, int(H), int(W));
}

/* we do NOT need a manual backward wrapper – the autograd Function
takes care of it.  Simply remove the whole backward_wrapper block. */

TORCH_LIBRARY(voxel_rasterizer, m)
{
    m.def("forward(Tensor vox, Tensor K, Tensor Twc, "
          "int h, int w) -> Tensor",
          TORCH_FN(forward_wrapper));
}

at::Tensor voxel_rasterizer_forward(
    torch::Tensor vox,
    torch::Tensor K,
    torch::Tensor Twc,
    int64_t H,
    int64_t W)
{
    return forward_wrapper(vox, K, Twc, H, W);
}
