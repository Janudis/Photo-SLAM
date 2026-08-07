#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <opencv2/core/types.hpp>

#include "include_voxel/voxel_types.h"

namespace sv {

struct RgbdTsdfGridKey {
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t z = 0;

    bool operator==(const RgbdTsdfGridKey& other) const noexcept
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct RgbdTsdfGridKeyHash {
    std::size_t operator()(const RgbdTsdfGridKey& key) const noexcept
    {
        std::size_t seed = static_cast<std::uint32_t>(key.x);
        seed ^= static_cast<std::size_t>(static_cast<std::uint32_t>(key.y)) +
                0x9e3779b9U + (seed << 6U) + (seed >> 2U);
        seed ^= static_cast<std::size_t>(static_cast<std::uint32_t>(key.z)) +
                0x9e3779b9U + (seed << 6U) + (seed >> 2U);
        return seed;
    }
};

struct RgbdTsdfCornerEvidence {
    float distance = 0.0f;
    float weight = 0.0f;
};

struct RgbdTsdfCellEvidence {
    std::vector<camera_id_t> observed_keyframes;
    cv::Vec3f color_sum{0.0f, 0.0f, 0.0f};
    std::uint32_t color_observations = 0;
};

inline std::array<RgbdTsdfGridKey, 8> rgbdTsdfCellCornerKeys(
    const RgbdTsdfGridKey& cell)
{
    return {{
        {cell.x,     cell.y,     cell.z},
        {cell.x,     cell.y,     cell.z + 1},
        {cell.x,     cell.y + 1, cell.z},
        {cell.x,     cell.y + 1, cell.z + 1},
        {cell.x + 1, cell.y,     cell.z},
        {cell.x + 1, cell.y,     cell.z + 1},
        {cell.x + 1, cell.y + 1, cell.z},
        {cell.x + 1, cell.y + 1, cell.z + 1},
    }};
}

}  // namespace sv
