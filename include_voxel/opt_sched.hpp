#pragma once
#include <torch/torch.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace sv { namespace optim {

// -------- Param group & state (mirrors Py) --------
struct ParamState {
  int64_t      step = 0;
  torch::Tensor exp_avg;    // same shape as param
  torch::Tensor exp_avg_sq; // same shape as param
};

struct ParamGroup {
  // exactly like Py: each group has a list of tensors and its own hyperparams
  std::vector<torch::Tensor*> params;
  double lr    = 1e-3;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double eps   = 1e-15;
  bool   biased = false;  // match SVRaster flag
  bool   sparse = false;  // present for parity; dense update here
};

// -------- SparseAdam (C++) --------
class SparseAdam {
public:
  SparseAdam() = default;

  void add_param_group(const ParamGroup& g) { groups_.push_back(g); }

  // Mirrors optimizer.zero_grad(set_to_none=True)
  void zero_grad(bool set_to_none = true) {
    for (auto& g : groups_) {
      for (auto* p : g.params) {
        if (!p || !p->defined()) continue;
        if (set_to_none) p->mutable_grad() = torch::Tensor();
        else if (p->grad().defined()) p->grad().zero_();
      }
    }
  }

  // Mirrors SparseAdam.step(): handles biased/unbiased paths
  void step() {
    torch::NoGradGuard ng;

    for (auto& g : groups_) {
      const double lr = g.lr, b1 = g.beta1, b2 = g.beta2, eps = g.eps;

      for (auto* P : g.params) {
        if (!P || !P->defined()) continue;
        auto& param = *P;
        auto grad = param.grad();
        if (!grad.defined()) continue;

        auto* key = param.unsafeGetTensorImpl();
        auto& st  = state_[key];

        // --- ALWAYS ensure buffers are defined (even if step > 0) ---
        if (!st.exp_avg.defined() || !st.exp_avg_sq.defined()) {
          st.exp_avg    = torch::zeros_like(param, torch::MemoryFormat::Preserve);
          st.exp_avg_sq = torch::zeros_like(param, torch::MemoryFormat::Preserve);
        }

        // First-time visit? (keep this too; it's harmless now)
        if (st.step == 0) {
          // already initialized above; nothing extra needed
        }
        st.step += 1;

        // m = b1*m + (1-b1)*g
        st.exp_avg.mul_(b1).add_(grad, 1.0 - b1);
        // v = b2*v + (1-b2)*g*g
        st.exp_avg_sq.mul_(b2).addcmul_(grad, grad, 1.0 - b2);

        if (g.biased) {
          auto denom = st.exp_avg_sq.sqrt_().add_(eps);
          param.addcdiv_(st.exp_avg, denom, -lr);
        } else {
          const double bc1 = 1.0 - std::pow(b1, (double)st.step);
          const double bc2 = 1.0 - std::pow(b2, (double)st.step);
          auto m_hat = st.exp_avg    / bc1;
          auto v_hat = st.exp_avg_sq / bc2;
          auto denom = v_hat.sqrt_().add_(eps);
          param.addcdiv_(m_hat, denom, -lr);
        }
      }
    }
  }

  // Called when you concatenate rows onto a param tensor (subdivision, etc.)
  void on_param_concatenated(torch::Tensor& old_param,
                            const torch::Tensor& add_rows,
                            torch::Tensor& new_param) {
    torch::NoGradGuard ng;

    auto old_key = old_param.unsafeGetTensorImpl();
    auto it = state_.find(old_key);
    if (it == state_.end()) {
      // No state yet → nothing to extend (this mirrors the Py check “had_state”).
      return;
    }
    auto& st = it->second;

    // If state exists but was never initialized, initialize now to old_param shape.
    if (!st.exp_avg.defined() || st.exp_avg.numel() == 0) {
      st.exp_avg    = torch::zeros_like(old_param, torch::MemoryFormat::Preserve).detach();
      st.exp_avg_sq = torch::zeros_like(old_param, torch::MemoryFormat::Preserve).detach();
      st.exp_avg.set_requires_grad(false);
      st.exp_avg_sq.set_requires_grad(false);
    }

    // Make sure add_rows matches param’s device & dtype
    auto rows = add_rows.to(old_param.options());     // device + dtype
    auto z    = torch::zeros_like(rows);

    st.exp_avg    = torch::cat({st.exp_avg,    z}, 0).contiguous().detach();
    st.exp_avg_sq = torch::cat({st.exp_avg_sq, z}, 0).contiguous().detach();
    st.exp_avg.set_requires_grad(false);
    st.exp_avg_sq.set_requires_grad(false);

    // Move state mapping to the new tensor object
    auto new_key = new_param.unsafeGetTensorImpl();
    state_.erase(it);
    state_.emplace(new_key, std::move(st));
  }

  // Update group pointers to point at the *new* tensor object after concat
  void rebind_param_pointer(torch::Tensor* old_ptr, torch::Tensor* new_ptr) {
    for (auto& g : groups_) {
      for (auto& p : g.params) if (p == old_ptr) p = new_ptr;
    }
  }

  std::vector<ParamGroup>& param_groups() { return groups_; }

private:
  std::vector<ParamGroup> groups_;
  std::unordered_map<c10::TensorImpl*, ParamState> state_;
};

// -------- MultiStepLR (C++) --------
// Matches torch.optim.lr_scheduler.MultiStepLR default behavior:
// last_epoch starts at -1; step() increments and decays if last_epoch in milestones.

struct MultiStepLRState {
  int64_t last_epoch = -1;
  std::vector<int> milestones;
  double gamma = 1.0;
};

class MultiStepLR {
public:
    MultiStepLR(SparseAdam* opt,
                const std::vector<int>& milestones,
                double gamma)
    : opt_(opt), gamma_(gamma) {
    milestones_.insert(milestones.begin(), milestones.end());
    }

    void step() {
    if (!opt_) return;
    ++last_epoch_;
    if (milestones_.count(last_epoch_) == 0) return;
    for (auto& g : opt_->param_groups()) g.lr *= gamma_;
    }

    int64_t last_epoch() const { return last_epoch_; }

    MultiStepLRState state_dict() const {
    MultiStepLRState st;
    st.last_epoch = last_epoch_;
    st.gamma = gamma_;
    st.milestones.reserve(milestones_.size());
    for (const auto& m : milestones_) st.milestones.push_back(m);
    std::sort(st.milestones.begin(), st.milestones.end());
    return st;
    }

    void load_state_dict(const MultiStepLRState& st) {
    last_epoch_ = st.last_epoch;
    gamma_ = st.gamma;
    milestones_.clear();
    milestones_.insert(st.milestones.begin(), st.milestones.end());
    }

private:
  SparseAdam* opt_ = nullptr;
  double gamma_ = 1.0;
  std::unordered_set<int> milestones_;
  int64_t last_epoch_ = -1;
};

}} // namespace sv::optim
