#pragma once

#include <opencv2/core/mat.hpp>

#include <string>

namespace sv {

// A sequence runner can provide a mapper-only depth image while ORB-SLAM uses
// a denser tracking depth image for the same RGB frame.
void registerMapperDepthImage(
    const std::string& image_filename,
    const std::string& depth_filename);

cv::Mat loadMapperDepthImage(const std::string& image_filename);

void clearMapperDepthImages();

} // namespace sv
