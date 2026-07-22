#include "include_voxel/mapper_depth_registry.h"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <mutex>
#include <unordered_map>

namespace sv {
namespace {

std::mutex g_mapper_depth_mutex;
std::unordered_map<std::string, std::string> g_mapper_depth_paths;

std::string normalizedPath(const std::string& path)
{
    return std::filesystem::path(path).lexically_normal().string();
}

} // namespace

void registerMapperDepthImage(
    const std::string& image_filename,
    const std::string& depth_filename)
{
    std::lock_guard<std::mutex> lock(g_mapper_depth_mutex);
    g_mapper_depth_paths[normalizedPath(image_filename)] =
        normalizedPath(depth_filename);
}

cv::Mat loadMapperDepthImage(const std::string& image_filename)
{
    std::string depth_filename;
    {
        std::lock_guard<std::mutex> lock(g_mapper_depth_mutex);
        const auto it = g_mapper_depth_paths.find(normalizedPath(image_filename));
        if (it == g_mapper_depth_paths.end()) {
            return cv::Mat();
        }
        depth_filename = it->second;
    }
    return cv::imread(depth_filename, cv::IMREAD_UNCHANGED);
}

void clearMapperDepthImages()
{
    std::lock_guard<std::mutex> lock(g_mapper_depth_mutex);
    g_mapper_depth_paths.clear();
}

} // namespace sv
