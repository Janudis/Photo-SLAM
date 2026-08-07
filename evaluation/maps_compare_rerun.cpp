#include <rerun.hpp>
#include <rerun/blueprint/archetypes/container_blueprint.hpp>
#include <rerun/blueprint/archetypes/time_panel_blueprint.hpp>
#include <rerun/blueprint/archetypes/view_blueprint.hpp>
#include <rerun/blueprint/archetypes/view_contents.hpp>
#include <rerun/blueprint/archetypes/viewport_blueprint.hpp>
#include <rerun/blueprint/components/container_kind.hpp>
#include <rerun/blueprint/components/grid_columns.hpp>
#include <rerun/blueprint/components/included_content.hpp>
#include <rerun/blueprint/components/panel_state.hpp>
#include <rerun/blueprint/components/query_expression.hpp>
#include <rerun/blueprint/components/root_container.hpp>
#include <rerun/blueprint/components/view_class.hpp>
#include <rerun/blueprint/components/view_origin.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Args {
    fs::path ours;
    fs::path hislam2;
    fs::path out;
    fs::path id_map;
    int stride = 1;
    int max_frames = 0;
    float depth_min = 0.0f;
    float depth_max = 6.0f;
};

struct FrameFiles {
    std::optional<fs::path> ours_rgb;
    std::optional<fs::path> ours_depth;
    std::optional<fs::path> ours_normal;
    std::optional<fs::path> ours_normal_svrecon;
    std::optional<fs::path> gt_rgb;
    std::optional<fs::path> gt_depth;
    std::optional<fs::path> gt_normal;
    std::optional<fs::path> hi_rgb_before;
    std::optional<fs::path> hi_depth_before;
    std::optional<fs::path> hi_normal_before;
    std::optional<fs::path> hi_rgb_after;
    std::optional<fs::path> hi_depth_after;
    std::optional<fs::path> hi_normal_after;
};

struct BlueprintNode {
    std::array<uint8_t, 16> uuid;
    std::string path;
};

void usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0
        << " --ours <our_shutdown_dir> --hislam2 <hislam2_output_dir> --out <maps_compare.rrd>\n"
        << "      [--id_map <ours_id hislam2_id text/csv>] [--stride N] [--max_frames N]\n"
        << "      [--depth_min M] [--depth_max M]\n";
}

bool parseInt(const std::string& text, int& out) {
    char* end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

bool parseFloat(const std::string& text, float& out) {
    char* end = nullptr;
    const float value = std::strtof(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0') {
        return false;
    }
    out = value;
    return true;
}

bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto require_value = [&](std::string& value) -> bool {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << key << "\n";
                return false;
            }
            value = argv[++i];
            return true;
        };

        std::string value;
        if (key == "--ours" && require_value(value)) {
            args.ours = value;
        } else if (key == "--hislam2" && require_value(value)) {
            args.hislam2 = value;
        } else if (key == "--out" && require_value(value)) {
            args.out = value;
        } else if (key == "--id_map" && require_value(value)) {
            args.id_map = value;
        } else if (key == "--stride" && require_value(value)) {
            if (!parseInt(value, args.stride)) {
                return false;
            }
        } else if (key == "--max_frames" && require_value(value)) {
            if (!parseInt(value, args.max_frames)) {
                return false;
            }
        } else if (key == "--depth_min" && require_value(value)) {
            if (!parseFloat(value, args.depth_min)) {
                return false;
            }
        } else if (key == "--depth_max" && require_value(value)) {
            if (!parseFloat(value, args.depth_max)) {
                return false;
            }
        } else if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown or malformed argument: " << key << "\n";
            return false;
        }
    }

    args.stride = std::max(1, args.stride);
    if (args.ours.empty() || args.hislam2.empty() || args.out.empty()) {
        return false;
    }
    if (args.depth_max <= args.depth_min) {
        args.depth_max = args.depth_min + 1.0f;
    }
    return true;
}

std::string shellQuote(const fs::path& path) {
    std::string out = "'";
    for (const char c : path.string()) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

bool mergeRrdFiles(const fs::path& recording, const fs::path& blueprint, const fs::path& out) {
    if (fs::exists(out)) {
        fs::remove(out);
    }
    const std::string command =
        "rerun rrd merge -o " + shellQuote(out) + " " + shellQuote(recording) +
        " " + shellQuote(blueprint);
    const int status = std::system(command.c_str());
    if (status != 0) {
        std::cerr << "[maps_compare_rerun] failed to merge Rerun streams\n";
        return false;
    }
    return fs::exists(out) && fs::file_size(out) > 0;
}

std::optional<int> regexId(const fs::path& path, const std::regex& pattern) {
    std::smatch match;
    const std::string name = path.filename().string();
    if (!std::regex_match(name, match, pattern) || match.size() < 2) {
        return std::nullopt;
    }
    int id = 0;
    if (!parseInt(match[1].str(), id)) {
        return std::nullopt;
    }
    return id;
}

std::vector<fs::path> listFiles(const fs::path& dir) {
    std::vector<fs::path> paths;
    if (!fs::exists(dir)) {
        return paths;
    }
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

std::map<int, FrameFiles> collectOurs(const fs::path& root) {
    std::map<int, FrameFiles> frames;
    const std::regex image_re(R"(^\d+_(\d+)_.*\.(jpg|jpeg|png)$)", std::regex::icase);
    const std::regex image_gt_re(R"(^\d+_(\d+)_.*_gt\.(jpg|jpeg|png)$)", std::regex::icase);
    const std::regex depth_re(R"(^kf_(\d+)_iter_.*\.png$)", std::regex::icase);
    const std::regex depth_gt_re(R"(^kf_(\d+)_gt\.png$)", std::regex::icase);
    const std::regex normal_re(R"(^kf_(\d+)\.png$)", std::regex::icase);
    const std::regex normal_gt_re(R"(^kf_(\d+)_gt_from_depth\.png$)", std::regex::icase);

    for (const auto& path : listFiles(root / "image")) {
        if (auto id = regexId(path, image_re)) {
            frames[*id].ours_rgb = path;
        }
    }
    for (const auto& path : listFiles(root / "image_gt")) {
        if (auto id = regexId(path, image_gt_re)) {
            frames[*id].gt_rgb = path;
        }
    }
    for (const auto& path : listFiles(root / "depth")) {
        if (auto id = regexId(path, depth_re)) {
            frames[*id].ours_depth = path;
        } else if (auto gt_id = regexId(path, depth_gt_re)) {
            frames[*gt_id].gt_depth = path;
        }
    }
    for (const auto& path : listFiles(root / "normal")) {
        if (auto gt_id = regexId(path, normal_gt_re)) {
            frames[*gt_id].gt_normal = path;
        } else if (auto id = regexId(path, normal_re)) {
            frames[*id].ours_normal = path;
        }
    }
    for (const auto& path : listFiles(root / "normals_svrecon")) {
        if (auto id = regexId(path, normal_re)) {
            frames[*id].ours_normal_svrecon = path;
        }
    }
    return frames;
}

void collectHislamFolder(
    std::map<int, FrameFiles>& frames,
    const fs::path& folder,
    const std::string& attr
) {
    const std::regex id_re(R"(^(\d+)\.(jpg|jpeg|png)$)", std::regex::icase);
    for (const auto& path : listFiles(folder)) {
        const auto id = regexId(path, id_re);
        if (!id) {
            continue;
        }
        FrameFiles& f = frames[*id];
        if (attr == "hi_rgb_before") {
            f.hi_rgb_before = path;
        } else if (attr == "hi_depth_before") {
            f.hi_depth_before = path;
        } else if (attr == "hi_normal_before") {
            f.hi_normal_before = path;
        } else if (attr == "hi_rgb_after") {
            f.hi_rgb_after = path;
        } else if (attr == "hi_depth_after") {
            f.hi_depth_after = path;
        } else if (attr == "hi_normal_after") {
            f.hi_normal_after = path;
        }
    }
}

std::map<int, FrameFiles> collectHislam2(const fs::path& root) {
    std::map<int, FrameFiles> frames;
    const fs::path renders = root / "renders";
    collectHislamFolder(frames, renders / "image_before_opt", "hi_rgb_before");
    collectHislamFolder(frames, renders / "depth_before_opt", "hi_depth_before");
    collectHislamFolder(frames, renders / "normal_before_opt", "hi_normal_before");
    collectHislamFolder(frames, renders / "image_after_opt", "hi_rgb_after");
    collectHislamFolder(frames, renders / "depth_after_opt", "hi_depth_after");
    collectHislamFolder(frames, renders / "normal_after_opt", "hi_normal_after");
    return frames;
}

std::map<int, int> readIdMap(const fs::path& path) {
    std::map<int, int> mapping;
    if (path.empty() || !fs::exists(path)) {
        return mapping;
    }

    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        for (char& c : line) {
            if (c == ',') {
                c = ' ';
            }
        }
        std::istringstream iss(line);
        int ours_id = 0;
        int hislam_id = 0;
        if (iss >> ours_id >> hislam_id) {
            mapping[ours_id] = hislam_id;
        }
    }
    return mapping;
}

cv::Mat blackImage(const cv::Size& size) {
    const int width = size.width > 0 ? size.width : 64;
    const int height = size.height > 0 ? size.height : 64;
    return cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
}

cv::Mat readRgb(const std::optional<fs::path>& path, const cv::Size& fallback_size = {}) {
    if (!path || !fs::exists(*path)) {
        return blackImage(fallback_size);
    }
    cv::Mat img = cv::imread(path->string(), cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        return blackImage(fallback_size);
    }

    cv::Mat rgb;
    if (img.channels() == 1) {
        cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, rgb, cv::COLOR_BGRA2RGB);
    } else {
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    }
    if (rgb.depth() != CV_8U) {
        cv::Mat tmp;
        rgb.convertTo(tmp, CV_8U);
        rgb = tmp;
    }
    if (!rgb.isContinuous()) {
        rgb = rgb.clone();
    }
    return rgb;
}

cv::Mat readDepthViz(
    const std::optional<fs::path>& path,
    float min_depth,
    float max_depth,
    const cv::Size& fallback_size = {}
) {
    if (!path || !fs::exists(*path)) {
        return blackImage(fallback_size);
    }
    cv::Mat img = cv::imread(path->string(), cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        return blackImage(fallback_size);
    }
    if (img.channels() == 3 || img.channels() == 4) {
        return readRgb(path, fallback_size);
    }

    cv::Mat depth;
    img.convertTo(depth, CV_32F);
    double raw_max = 0.0;
    cv::minMaxLoc(depth, nullptr, &raw_max);
    if (img.depth() == CV_16U) {
        const std::string p = path->string();
        const bool replica_encoded =
            p.find("HI-SLAM2") != std::string::npos ||
            p.find("outputs/replica") != std::string::npos;
        const float scale = replica_encoded ? (1.0f / 6553.5f) : (1.0f / 1000.0f);
        depth *= scale;
    }

    cv::Mat valid = (depth > min_depth) & (depth < max_depth);
    cv::Mat norm = cv::Mat::zeros(depth.size(), CV_32F);
    const float denom = std::max(max_depth - min_depth, 1.0e-6f);
    for (int y = 0; y < depth.rows; ++y) {
        const float* dptr = depth.ptr<float>(y);
        const uchar* vptr = valid.ptr<uchar>(y);
        float* nptr = norm.ptr<float>(y);
        for (int x = 0; x < depth.cols; ++x) {
            if (vptr[x]) {
                nptr[x] = std::clamp((dptr[x] - min_depth) / denom, 0.0f, 1.0f);
            }
        }
    }

    cv::Mat u8;
    norm.convertTo(u8, CV_8U, 255.0);
    cv::Mat color_bgr;
    cv::applyColorMap(u8, color_bgr, cv::COLORMAP_JET);
    color_bgr.setTo(cv::Scalar(0, 0, 0), ~valid);

    cv::Mat color_rgb;
    cv::cvtColor(color_bgr, color_rgb, cv::COLOR_BGR2RGB);
    if (!color_rgb.isContinuous()) {
        color_rgb = color_rgb.clone();
    }
    return color_rgb;
}

void logImage(rerun::RecordingStream& rec, const std::string& path, const cv::Mat& rgb) {
    cv::Mat contiguous = rgb.isContinuous() ? rgb : rgb.clone();
    std::vector<uint8_t> bytes(
        contiguous.data,
        contiguous.data + contiguous.total() * contiguous.elemSize()
    );
    rec.log(
        path,
        rerun::Image::from_rgb24(
            rerun::Collection<uint8_t>::take_ownership(std::move(bytes)),
            {static_cast<uint32_t>(contiguous.cols), static_cast<uint32_t>(contiguous.rows)}
        )
    );
}

void logMapsCompareBlueprint(rerun::RecordingStream& blueprint) {
    namespace rb = rerun::blueprint;
    using rb::archetypes::ContainerBlueprint;
    using rb::archetypes::TimePanelBlueprint;
    using rb::archetypes::ViewBlueprint;
    using rb::archetypes::ViewContents;
    using rb::archetypes::ViewportBlueprint;
    using rb::components::ContainerKind;
    using rb::components::GridColumns;
    using rb::components::IncludedContent;
    using rb::components::PanelState;
    using rb::components::QueryExpression;
    using rb::components::RootContainer;
    using rb::components::ViewClass;
    using rb::components::ViewOrigin;

    const BlueprintNode tabs{
        {0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00,
         0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01},
        "/container/10000000-0000-4000-8000-000000000001"
    };
    const BlueprintNode rgb_grid{
        {0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00,
         0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01},
        "/container/20000000-0000-4000-8000-000000000001"
    };
    const BlueprintNode depth_grid{
        {0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00,
         0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01},
        "/container/30000000-0000-4000-8000-000000000001"
    };
    const BlueprintNode normal_grid{
        {0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00,
         0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01},
        "/container/40000000-0000-4000-8000-000000000001"
    };

    const auto make_view = [](uint8_t prefix, uint8_t index) -> BlueprintNode {
        char path[48];
        std::snprintf(
            path,
            sizeof(path),
            "/view/%02x000000-0000-4000-8000-0000000000%02x",
            prefix,
            index
        );
        return BlueprintNode{
            {prefix, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00,
             0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, index},
            path
        };
    };

    const std::array<BlueprintNode, 4> rgb_views = {
        make_view(0x51, 0x01),
        make_view(0x51, 0x02),
        make_view(0x51, 0x03),
        make_view(0x51, 0x04),
    };
    const std::array<BlueprintNode, 4> depth_views = {
        make_view(0x52, 0x01),
        make_view(0x52, 0x02),
        make_view(0x52, 0x03),
        make_view(0x52, 0x04),
    };
    const std::array<BlueprintNode, 5> normal_views = {
        make_view(0x53, 0x01),
        make_view(0x53, 0x02),
        make_view(0x53, 0x03),
        make_view(0x53, 0x04),
        make_view(0x53, 0x05),
    };

    const auto log_view =
        [&](const BlueprintNode& node, const std::string& name, const std::string& origin) {
            blueprint.log(
                node.path,
                ViewBlueprint(ViewClass(std::string("2D")))
                    .with_display_name(rerun::components::Name(name))
                    .with_space_origin(ViewOrigin(origin))
            );
            blueprint.log(
                node.path + "/ViewContents",
                ViewContents(rerun::Collection<QueryExpression>{
                    QueryExpression(std::string("$origin/**"))
                })
            );
        };

    const auto log_grid =
        [&](const BlueprintNode& node,
            const std::string& name,
            const std::vector<BlueprintNode>& views) {
            std::vector<IncludedContent> contents;
            contents.reserve(views.size());
            for (const auto& view : views) {
                contents.emplace_back(view.path);
            }
            blueprint.log(
                node.path,
                ContainerBlueprint(ContainerKind::Grid)
                    .with_display_name(rerun::components::Name(name))
                    .with_contents(contents)
                    .with_grid_columns(GridColumns(2))
            );
        };

    log_view(rgb_views[0], "Ours RGB", "maps/rgb/ours");
    log_view(rgb_views[1], "GT RGB", "maps/rgb/gt");
    log_view(rgb_views[2], "HI-SLAM2 Before RGB", "maps/rgb/hislam2_before");
    log_view(rgb_views[3], "HI-SLAM2 After RGB", "maps/rgb/hislam2_after");
    log_grid(rgb_grid, "RGB", {rgb_views.begin(), rgb_views.end()});

    log_view(depth_views[0], "Ours Depth", "maps/depth/ours");
    log_view(depth_views[1], "GT Depth", "maps/depth/gt");
    log_view(depth_views[2], "HI-SLAM2 Before Depth", "maps/depth/hislam2_before");
    log_view(depth_views[3], "HI-SLAM2 After Depth", "maps/depth/hislam2_after");
    log_grid(depth_grid, "Depth", {depth_views.begin(), depth_views.end()});

    log_view(normal_views[0], "Ours Normal", "maps/normal/ours");
    log_view(normal_views[1], "Ours SVRecon Normal", "maps/normal/ours_svrecon");
    log_view(normal_views[2], "GT Normal", "maps/normal/gt");
    log_view(normal_views[3], "HI-SLAM2 Before Normal", "maps/normal/hislam2_before");
    log_view(normal_views[4], "HI-SLAM2 After Normal", "maps/normal/hislam2_after");
    log_grid(normal_grid, "Normal", {normal_views.begin(), normal_views.end()});

    blueprint.log(
        tabs.path,
        ContainerBlueprint(ContainerKind::Tabs)
            .with_contents({
                IncludedContent(rgb_grid.path),
                IncludedContent(depth_grid.path),
                IncludedContent(normal_grid.path),
            })
    );
    blueprint.log(
        "/viewport",
        ViewportBlueprint::update_fields().with_root_container(RootContainer(tabs.uuid))
    );
    blueprint.log(
        "/time_panel",
        TimePanelBlueprint::update_fields().with_state(PanelState::Collapsed)
    );
}

FrameFiles mergeFrame(
    const FrameFiles& ours,
    const std::map<int, FrameFiles>& hislam2,
    int hi_id
) {
    FrameFiles out = ours;
    const auto hi_it = hislam2.find(hi_id);
    if (hi_it != hislam2.end()) {
        out.hi_rgb_before = hi_it->second.hi_rgb_before;
        out.hi_depth_before = hi_it->second.hi_depth_before;
        out.hi_normal_before = hi_it->second.hi_normal_before;
        out.hi_rgb_after = hi_it->second.hi_rgb_after;
        out.hi_depth_after = hi_it->second.hi_depth_after;
        out.hi_normal_after = hi_it->second.hi_normal_after;
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) {
        usage(argv[0]);
        return 1;
    }

    const auto ours = collectOurs(args.ours);
    const auto hislam2 = collectHislam2(args.hislam2);
    const auto id_map = readIdMap(args.id_map);
    if (ours.empty()) {
        std::cerr << "[maps_compare_rerun] no frames found in " << args.ours << "\n";
        return 1;
    }
    if (args.out.has_parent_path()) {
        fs::create_directories(args.out.parent_path());
    }
    if (fs::exists(args.out)) {
        fs::remove(args.out);
    }
    const fs::path blueprint_tmp = args.out.string() + ".blueprint.tmp.rrd";
    const fs::path recording_tmp = args.out.string() + ".recording.tmp.rrd";
    if (fs::exists(blueprint_tmp)) {
        fs::remove(blueprint_tmp);
    }
    if (fs::exists(recording_tmp)) {
        fs::remove(recording_tmp);
    }

    {
        rerun::RecordingStream blueprint(
            "maps_compare",
            "maps_compare",
            rerun::StoreKind::Blueprint
        );
        blueprint.save(blueprint_tmp.string()).exit_on_failure();
        blueprint.set_time_sequence("blueprint", 0);
        logMapsCompareBlueprint(blueprint);
        blueprint.flush_blocking().exit_on_failure();
    }

    int logged = 0;
    {
        rerun::RecordingStream rec("maps_compare", "maps_compare");
        rec.save(recording_tmp.string()).exit_on_failure();

        int ordinal = 0;
        for (const auto& [ours_id, ours_files] : ours) {
            if ((ordinal++ % args.stride) != 0) {
                continue;
            }
            const auto map_it = id_map.find(ours_id);
            const int hi_id = map_it != id_map.end() ? map_it->second : ours_id;
            const FrameFiles files = mergeFrame(ours_files, hislam2, hi_id);

            const cv::Mat gt_rgb = readRgb(files.gt_rgb);
            const cv::Size shape(gt_rgb.cols, gt_rgb.rows);
            const cv::Mat ours_rgb = readRgb(files.ours_rgb, shape);
            const cv::Mat hi_rgb_before = readRgb(files.hi_rgb_before, shape);
            const cv::Mat hi_rgb_after = readRgb(files.hi_rgb_after, shape);

            const cv::Mat gt_depth =
                readDepthViz(files.gt_depth, args.depth_min, args.depth_max, shape);
            const cv::Mat ours_depth =
                readDepthViz(files.ours_depth, args.depth_min, args.depth_max, shape);
            const cv::Mat hi_depth_before =
                readDepthViz(files.hi_depth_before, args.depth_min, args.depth_max, shape);
            const cv::Mat hi_depth_after =
                readDepthViz(files.hi_depth_after, args.depth_min, args.depth_max, shape);

            const cv::Mat gt_normal = readRgb(files.gt_normal, shape);
            const cv::Mat ours_normal = readRgb(files.ours_normal, shape);
            const cv::Mat ours_normal_svrecon = readRgb(files.ours_normal_svrecon, shape);
            const cv::Mat hi_normal_before = readRgb(files.hi_normal_before, shape);
            const cv::Mat hi_normal_after = readRgb(files.hi_normal_after, shape);

            rec.set_time_sequence("keyframe_id", ours_id);
            rec.set_time_sequence("frame", logged);

            logImage(rec, "maps/rgb/ours", ours_rgb);
            logImage(rec, "maps/rgb/gt", gt_rgb);
            logImage(rec, "maps/rgb/hislam2_before", hi_rgb_before);
            logImage(rec, "maps/rgb/hislam2_after", hi_rgb_after);

            logImage(rec, "maps/depth/ours", ours_depth);
            logImage(rec, "maps/depth/gt", gt_depth);
            logImage(rec, "maps/depth/hislam2_before", hi_depth_before);
            logImage(rec, "maps/depth/hislam2_after", hi_depth_after);

            logImage(rec, "maps/normal/ours", ours_normal);
            logImage(rec, "maps/normal/ours_svrecon", ours_normal_svrecon);
            logImage(rec, "maps/normal/gt", gt_normal);
            logImage(rec, "maps/normal/hislam2_before", hi_normal_before);
            logImage(rec, "maps/normal/hislam2_after", hi_normal_after);

            ++logged;
            if (args.max_frames > 0 && logged >= args.max_frames) {
                break;
            }
        }
        rec.flush_blocking().exit_on_failure();
    }

    if (!mergeRrdFiles(recording_tmp, blueprint_tmp, args.out)) {
        return 1;
    }
    fs::remove(blueprint_tmp);
    fs::remove(recording_tmp);

    std::cout << "[maps_compare_rerun] wrote " << logged
              << " frames to " << args.out
              << " (ours_frames=" << ours.size()
              << ", hislam2_frames=" << hislam2.size() << ")\n";
    return 0;
}


// how to run (example):
// ./bin/maps_compare_rerun \
//   --ours results/replica_rgbd_voxel/office0/4854_shutdown \
//   --hislam2 third_party/HI-SLAM2/outputs/replica/office0 \
//   --id_map results/replica_rgbd_voxel/office0/4854_shutdown/kf_frame_id_map.txt \
//   --out results/replica_rgbd_voxel/office0/4854_shutdown/maps_compare.rrd
