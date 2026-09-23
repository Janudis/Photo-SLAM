#pragma once

#include <torch/torch.h>

namespace sv {

inline constexpr float kSHC0 = 0.28209479177387814f;

inline constexpr float kInactiveGeoDensifyMaxPixelDist = 1.0f;
inline constexpr float kSdfInitializationOrbRadiusVox = 1.0f;

inline constexpr bool kMonocularRenderedDepthDensify = true;
inline constexpr int kMonocularRenderedDepthPixelStride = 8;
inline constexpr int kMonocularRenderedDepthEvidenceSamples = 4;
inline constexpr float kMonocularRenderedDepthEvidenceTruncVox = 1.0f;
inline constexpr float kMonocularRenderedDepthEvidenceMaxWeight = 64.0f;
inline constexpr int kMonocularRenderedDepthEvidencePromoteMinViews = 2;
inline constexpr float kMonocularRenderedDepthEvidencePromoteMinWeight = 1.0f;
inline constexpr float kMonocularRenderedDepthEvidenceMinBaselineRatio = 0.05f;

inline constexpr int kMonocularMvsWidth = 512;
inline constexpr int kMonocularMvsHeight = 320;
inline constexpr int kMonocularMvsViewNum = 7;
inline constexpr char kMonocularMvsDepthRangeMode[] = "tandem_sparse_quantile";
inline constexpr float kMonocularMvsDepthMinScene = 0.01f;
inline constexpr float kMonocularMvsInverseDepthQuantile = 0.20f;
inline constexpr float kMonocularMvsDepthMaxMultiplier = 3.0f;
inline constexpr float kMonocularMvsDiscardPercentage = 10.0f;
inline constexpr int kMonocularMvsTsdfEvidencePixelStride = 1;
inline constexpr float kMonocularMvsTsdfEvidenceTruncVox = 2.0f;
inline constexpr float kMonocularMvsTsdfEvidenceMaxWeight = 64.0f;
inline constexpr int kMonocularMvsTsdfEvidencePromoteMinViews = 2;
inline constexpr float kMonocularMvsTsdfEvidencePromoteMinWeight = 1.0f;

inline constexpr float kMonocularDepthEndMultiplier = 0.1f;
inline constexpr float kMonocularDepthAlphaMin = 0.5f;
inline constexpr float kMonocularDepthConfidenceMin = 0.0f;
inline constexpr float kLambdaMonocularNormal = 0.0001f;
inline constexpr int kMonocularNormalFrom = 1000;
inline constexpr int kMonocularNormalEnd = 15000;
inline constexpr float kMonocularNormalEndMultiplier = 0.1f;
inline constexpr int kMonocularNormalKernelSize = 3;
inline constexpr float kMonocularNormalToleranceDegrees = 90.0f;
inline constexpr float kMonocularNormalMaxDepthJumpRelative = 0.05f;

inline constexpr int kPruneMvsMinSupportingViews = 2;
inline constexpr int kPruneMvsMinContradictingViews = 2;
inline constexpr float kPruneMvsDepthToleranceVox = 1.5f;
inline constexpr int kSurfaceMinViews = 4;
inline constexpr int kSurfaceViewWindowSize = 8;

inline torch::Tensor rgbToShZero(const torch::Tensor& rgb)
{
    return (rgb - 0.5f) / kSHC0;
}

inline torch::Tensor shZeroToRgb(const torch::Tensor& sh)
{
    return sh * kSHC0 + 0.5f;
}

}
