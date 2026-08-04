#include "include_voxel/laptop_precheck_profiler.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace sv {
namespace {

constexpr double kBytesPerMb = 1024.0 * 1024.0;

double average(double sum, std::uint64_t count)
{
    return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

std::string jsonEscape(const std::string& value)
{
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20) {
                out << "\\u"
                    << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(ch)
                    << std::dec << std::setfill(' ');
            } else {
                out << static_cast<char>(ch);
            }
        }
    }
    return out.str();
}

std::string csvEscape(const std::string& value)
{
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }
    std::string escaped = "\"";
    for (const char ch : value) {
        escaped += ch == '"' ? "\"\"" : std::string(1, ch);
    }
    escaped += '"';
    return escaped;
}

} // namespace

LaptopPrecheckProfiler::Scope::Scope(
    LaptopPrecheckProfiler* profiler,
    std::string module,
    const std::uint64_t work_items)
    : profiler_(profiler),
      module_(std::move(module)),
      work_items_(work_items),
      start_(Clock::now())
{
    if (profiler_) {
        profiler_->enter(module_);
    }
}

LaptopPrecheckProfiler::Scope::~Scope()
{
    finish();
}

LaptopPrecheckProfiler::Scope::Scope(Scope&& other) noexcept
    : profiler_(other.profiler_),
      module_(std::move(other.module_)),
      work_items_(other.work_items_),
      start_(other.start_)
{
    other.profiler_ = nullptr;
    other.work_items_ = 0;
}

LaptopPrecheckProfiler::Scope& LaptopPrecheckProfiler::Scope::operator=(
    Scope&& other) noexcept
{
    if (this != &other) {
        finish();
        profiler_ = other.profiler_;
        module_ = std::move(other.module_);
        work_items_ = other.work_items_;
        start_ = other.start_;
        other.profiler_ = nullptr;
        other.work_items_ = 0;
    }
    return *this;
}

void LaptopPrecheckProfiler::Scope::finish()
{
    if (!profiler_) {
        return;
    }
    profiler_->leave(module_, start_, work_items_);
    profiler_ = nullptr;
}

LaptopPrecheckProfiler::LaptopPrecheckProfiler(
    const bool enabled,
    const int sample_interval_ms,
    const bool cuda_enabled)
    : enabled_(enabled),
      cuda_enabled_(cuda_enabled),
      sample_interval_ms_(std::max(10, sample_interval_ms))
{
}

LaptopPrecheckProfiler::~LaptopPrecheckProfiler()
{
    stop();
}

void LaptopPrecheckProfiler::start()
{
    if (!enabled_) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (running_) {
            return;
        }
        stop_requested_ = false;
        running_ = true;
        run_start_ = Clock::now();
        run_end_ = run_start_;
    }
    sampleNow();
    sampler_thread_ = std::thread(&LaptopPrecheckProfiler::samplingLoop, this);
}

void LaptopPrecheckProfiler::stop()
{
    if (!enabled_) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_) {
            return;
        }
    }

    sampleNow();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_requested_ = true;
    }
    sample_cv_.notify_all();
    if (sampler_thread_.joinable()) {
        sampler_thread_.join();
    }

    const Clock::time_point now = Clock::now();
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& item : async_calls_) {
        ModuleStats& stats = modules_[item.first];
        for (const AsyncCall& call : item.second) {
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(now - call.start).count();
            ++stats.calls;
            stats.work_items += call.work_items;
            stats.total_wall_ms += elapsed_ms;
            stats.max_wall_ms = std::max(stats.max_wall_ms, elapsed_ms);
        }
    }
    async_calls_.clear();
    active_modules_.clear();
    run_end_ = now;
    running_ = false;
}

bool LaptopPrecheckProfiler::enabled() const
{
    return enabled_;
}

LaptopPrecheckProfiler::Scope LaptopPrecheckProfiler::profile(
    const std::string& module,
    const std::uint64_t work_items)
{
    if (!enabled_ || module.empty()) {
        return {};
    }
    return Scope(this, module, work_items);
}

void LaptopPrecheckProfiler::beginAsync(
    const std::string& module,
    const std::uint64_t work_items)
{
    if (!enabled_ || module.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++active_modules_[module];
        async_calls_[module].push_back({Clock::now(), work_items});
    }
}

void LaptopPrecheckProfiler::endAsync(const std::string& module)
{
    if (!enabled_ || module.empty()) {
        return;
    }
    const Clock::time_point now = Clock::now();
    std::lock_guard<std::mutex> lock(mutex_);
    auto calls = async_calls_.find(module);
    if (calls == async_calls_.end() || calls->second.empty()) {
        return;
    }
    const AsyncCall call = calls->second.back();
    calls->second.pop_back();
    if (calls->second.empty()) {
        async_calls_.erase(calls);
    }

    ModuleStats& stats = modules_[module];
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(now - call.start).count();
    ++stats.calls;
    stats.work_items += call.work_items;
    stats.total_wall_ms += elapsed_ms;
    stats.max_wall_ms = std::max(stats.max_wall_ms, elapsed_ms);

    auto active = active_modules_.find(module);
    if (active != active_modules_.end()) {
        if (active->second <= 1) {
            active_modules_.erase(active);
        } else {
            --active->second;
        }
    }
}

void LaptopPrecheckProfiler::MemoryAggregate::add(
    const MemorySnapshot& snapshot)
{
    ++samples;
    if (snapshot.process_rss_mb >= 0.0) {
        process_rss_sum_mb += snapshot.process_rss_mb;
        process_rss_peak_mb = std::max(process_rss_peak_mb, snapshot.process_rss_mb);
    }
    if (snapshot.cuda_allocated_mb >= 0.0) {
        cuda_allocated_sum_mb += snapshot.cuda_allocated_mb;
        cuda_allocated_peak_mb =
            std::max(cuda_allocated_peak_mb, snapshot.cuda_allocated_mb);
    }
    if (snapshot.cuda_reserved_mb >= 0.0) {
        cuda_reserved_sum_mb += snapshot.cuda_reserved_mb;
        cuda_reserved_peak_mb =
            std::max(cuda_reserved_peak_mb, snapshot.cuda_reserved_mb);
    }
    if (snapshot.cuda_device_used_mb >= 0.0) {
        cuda_device_used_sum_mb += snapshot.cuda_device_used_mb;
        cuda_device_used_peak_mb =
            std::max(cuda_device_used_peak_mb, snapshot.cuda_device_used_mb);
    }
}

void LaptopPrecheckProfiler::enter(const std::string& module)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++active_modules_[module];
    }
}

void LaptopPrecheckProfiler::leave(
    const std::string& module,
    const Clock::time_point start,
    const std::uint64_t work_items)
{
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::lock_guard<std::mutex> lock(mutex_);
    ModuleStats& stats = modules_[module];
    ++stats.calls;
    stats.work_items += work_items;
    stats.total_wall_ms += elapsed_ms;
    stats.max_wall_ms = std::max(stats.max_wall_ms, elapsed_ms);

    auto active = active_modules_.find(module);
    if (active != active_modules_.end()) {
        if (active->second <= 1) {
            active_modules_.erase(active);
        } else {
            --active->second;
        }
    }
}

LaptopPrecheckProfiler::MemorySnapshot
LaptopPrecheckProfiler::readMemorySnapshot() const
{
    MemorySnapshot snapshot;

    std::ifstream statm("/proc/self/statm");
    long total_pages = 0;
    long resident_pages = 0;
    if (statm >> total_pages >> resident_pages) {
        const long page_size = ::sysconf(_SC_PAGESIZE);
        if (page_size > 0) {
            snapshot.process_rss_mb =
                static_cast<double>(resident_pages) *
                static_cast<double>(page_size) / kBytesPerMb;
        }
    }

    if (!cuda_enabled_) {
        return snapshot;
    }
    try {
        namespace allocator = c10::cuda::CUDACachingAllocator;
        const allocator::DeviceStats stats = allocator::getDeviceStats(0);
        const int aggregate =
            static_cast<int>(allocator::StatType::AGGREGATE);
        snapshot.cuda_allocated_mb =
            static_cast<double>(stats.allocated_bytes[aggregate].current) /
            kBytesPerMb;
        snapshot.cuda_reserved_mb =
            static_cast<double>(stats.reserved_bytes[aggregate].current) /
            kBytesPerMb;
    } catch (...) {
        snapshot.cuda_allocated_mb = -1.0;
        snapshot.cuda_reserved_mb = -1.0;
    }

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
        total_bytes >= free_bytes) {
        snapshot.cuda_device_used_mb =
            static_cast<double>(total_bytes - free_bytes) / kBytesPerMb;
    }
    return snapshot;
}

void LaptopPrecheckProfiler::sampleNow()
{
    if (!enabled_) {
        return;
    }
    const MemorySnapshot snapshot = readMemorySnapshot();
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
        return;
    }
    global_memory_.add(snapshot);
    for (const auto& active : active_modules_) {
        if (active.second > 0) {
            modules_[active.first].memory.add(snapshot);
        }
    }
}

void LaptopPrecheckProfiler::samplingLoop()
{
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_requested_) {
        if (sample_cv_.wait_for(
                lock,
                std::chrono::milliseconds(sample_interval_ms_),
                [this]() { return stop_requested_; })) {
            break;
        }
        lock.unlock();
        sampleNow();
        lock.lock();
    }
}

void LaptopPrecheckProfiler::writeReports(
    const std::filesystem::path& output_dir,
    const LaptopPrecheckMetadata& metadata) const
{
    if (!enabled_) {
        return;
    }
    std::filesystem::create_directories(output_dir);

    MemoryAggregate global;
    std::map<std::string, ModuleStats> modules;
    Clock::time_point start;
    Clock::time_point end;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        global = global_memory_;
        modules = modules_;
        start = run_start_;
        end = run_end_;
    }
    const double profiled_seconds =
        std::chrono::duration<double>(end - start).count();

    const auto rss_mean = average(global.process_rss_sum_mb, global.samples);
    const auto cuda_alloc_mean =
        average(global.cuda_allocated_sum_mb, global.samples);
    const auto cuda_reserved_mean =
        average(global.cuda_reserved_sum_mb, global.samples);
    const auto cuda_device_mean =
        average(global.cuda_device_used_sum_mb, global.samples);

    std::ofstream json(output_dir / "laptop_precheck.json");
    if (!json) {
        throw std::runtime_error("Failed to write laptop_precheck.json");
    }
    json << std::fixed << std::setprecision(6);
    json << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"config_file\": \""
         << jsonEscape(metadata.config_file.string()) << "\",\n"
         << "  \"sensor\": \"" << jsonEscape(metadata.sensor) << "\",\n"
         << "  \"device\": \"" << jsonEscape(metadata.device) << "\",\n"
         << "  \"pipeline\": \"" << jsonEscape(metadata.pipeline) << "\",\n"
         << "  \"frames\": " << metadata.frames << ",\n"
         << "  \"keyframes\": " << metadata.keyframes << ",\n"
         << "  \"voxels\": " << metadata.voxels << ",\n"
         << "  \"iterations\": " << metadata.iterations << ",\n"
         << "  \"mapping_seconds\": " << metadata.mapping_seconds << ",\n"
         << "  \"mapping_fps_hz\": "
         << (metadata.mapping_seconds > 0.0
                 ? static_cast<double>(metadata.frames) /
                       metadata.mapping_seconds
                 : 0.0)
         << ",\n"
         << "  \"profiled_wall_seconds\": " << profiled_seconds << ",\n"
         << "  \"map_path\": \""
         << jsonEscape(metadata.map_path.string()) << "\",\n"
         << "  \"map_size_mb\": " << metadata.map_size_mb << ",\n"
         << "  \"sample_interval_ms\": " << sample_interval_ms_ << ",\n"
         << "  \"cuda_allocator_available\": "
         << (cuda_enabled_ ? "true" : "false") << ",\n"
         << "  \"global_process_memory\": {\n"
         << "    \"samples\": " << global.samples << ",\n"
         << "    \"mean_process_rss_mb\": " << rss_mean << ",\n"
         << "    \"peak_process_rss_mb\": " << global.process_rss_peak_mb << ",\n"
         << "    \"mean_cuda_allocated_mb\": " << cuda_alloc_mean << ",\n"
         << "    \"peak_cuda_allocated_mb\": " << global.cuda_allocated_peak_mb << ",\n"
         << "    \"mean_cuda_reserved_mb\": " << cuda_reserved_mean << ",\n"
         << "    \"peak_cuda_reserved_mb\": " << global.cuda_reserved_peak_mb << ",\n"
         << "    \"mean_cuda_device_used_mb\": " << cuda_device_mean << ",\n"
         << "    \"peak_cuda_device_used_mb\": " << global.cuda_device_used_peak_mb << "\n"
         << "  },\n"
         << "  \"modules\": [\n";

    std::size_t module_index = 0;
    for (const auto& item : modules) {
        const ModuleStats& stats = item.second;
        const std::uint64_t samples = stats.memory.samples;
        json << "    {\n"
             << "      \"name\": \"" << jsonEscape(item.first) << "\",\n"
             << "      \"calls\": " << stats.calls << ",\n"
             << "      \"work_items\": " << stats.work_items << ",\n"
             << "      \"total_wall_ms\": " << stats.total_wall_ms << ",\n"
             << "      \"mean_wall_ms_per_call\": "
             << average(stats.total_wall_ms, stats.calls) << ",\n"
             << "      \"mean_wall_ms_per_work_item\": "
             << average(stats.total_wall_ms, stats.work_items) << ",\n"
             << "      \"max_wall_ms_per_call\": " << stats.max_wall_ms << ",\n"
             << "      \"memory_samples_while_active\": " << samples << ",\n"
             << "      \"mean_process_rss_mb_while_active\": "
             << average(stats.memory.process_rss_sum_mb, samples) << ",\n"
             << "      \"peak_process_rss_mb_while_active\": "
             << stats.memory.process_rss_peak_mb << ",\n"
             << "      \"mean_cuda_allocated_mb_while_active\": "
             << average(stats.memory.cuda_allocated_sum_mb, samples) << ",\n"
             << "      \"peak_cuda_allocated_mb_while_active\": "
             << stats.memory.cuda_allocated_peak_mb << ",\n"
             << "      \"mean_cuda_reserved_mb_while_active\": "
             << average(stats.memory.cuda_reserved_sum_mb, samples) << ",\n"
             << "      \"peak_cuda_reserved_mb_while_active\": "
             << stats.memory.cuda_reserved_peak_mb << ",\n"
             << "      \"mean_cuda_device_used_mb_while_active\": "
             << average(stats.memory.cuda_device_used_sum_mb, samples) << ",\n"
             << "      \"peak_cuda_device_used_mb_while_active\": "
             << stats.memory.cuda_device_used_peak_mb << "\n"
             << "    }";
        json << (++module_index < modules.size() ? ",\n" : "\n");
    }
    json << "  ],\n"
         << "  \"notes\": [\n"
         << "    \"Module wall time is host-observed; asynchronous CUDA work can overlap other modules.\",\n"
         << "    \"Module memory is whole-process memory sampled while that module is active, not exclusive ownership.\",\n"
         << "    \"CUDA allocated/reserved values cover PyTorch's caching allocator; CUDA device used also includes other CUDA libraries and other processes.\",\n"
         << "    \"On Jetson, CPU and GPU share physical memory, so RSS and CUDA values must not be added as independent capacities.\"\n"
         << "  ]\n"
         << "}\n";

    std::ofstream csv(output_dir / "laptop_precheck_modules.csv");
    if (!csv) {
        throw std::runtime_error("Failed to write laptop_precheck_modules.csv");
    }
    csv << std::fixed << std::setprecision(6);
    csv << "module,calls,work_items,total_wall_ms,mean_wall_ms_per_call,"
           "mean_wall_ms_per_work_item,max_wall_ms_per_call,memory_samples_while_active,"
           "mean_process_rss_mb_while_active,peak_process_rss_mb_while_active,"
           "mean_cuda_allocated_mb_while_active,peak_cuda_allocated_mb_while_active,"
           "mean_cuda_reserved_mb_while_active,peak_cuda_reserved_mb_while_active,"
           "mean_cuda_device_used_mb_while_active,peak_cuda_device_used_mb_while_active\n";
    for (const auto& item : modules) {
        const ModuleStats& stats = item.second;
        const std::uint64_t samples = stats.memory.samples;
        csv << csvEscape(item.first) << ','
            << stats.calls << ',' << stats.work_items << ','
            << stats.total_wall_ms << ','
            << average(stats.total_wall_ms, stats.calls) << ','
            << average(stats.total_wall_ms, stats.work_items) << ','
            << stats.max_wall_ms << ',' << samples << ','
            << average(stats.memory.process_rss_sum_mb, samples) << ','
            << stats.memory.process_rss_peak_mb << ','
            << average(stats.memory.cuda_allocated_sum_mb, samples) << ','
            << stats.memory.cuda_allocated_peak_mb << ','
            << average(stats.memory.cuda_reserved_sum_mb, samples) << ','
            << stats.memory.cuda_reserved_peak_mb << ','
            << average(stats.memory.cuda_device_used_sum_mb, samples) << ','
            << stats.memory.cuda_device_used_peak_mb << '\n';
    }

    std::ofstream txt(output_dir / "laptop_precheck.txt");
    if (!txt) {
        throw std::runtime_error("Failed to write laptop_precheck.txt");
    }
    txt << std::fixed << std::setprecision(2);
    txt << "Laptop Precheck\n"
        << "===============\n"
        << "Config: " << metadata.config_file << '\n'
        << "Sensor/device: " << metadata.sensor << " / " << metadata.device << '\n'
        << "Pipeline: " << metadata.pipeline << '\n'
        << "Frames/keyframes/iterations/voxels: "
        << metadata.frames << " / " << metadata.keyframes << " / "
        << metadata.iterations << " / " << metadata.voxels << '\n'
        << "Mapping time/FPS: " << metadata.mapping_seconds << " s / "
        << (metadata.mapping_seconds > 0.0
                ? static_cast<double>(metadata.frames) / metadata.mapping_seconds
                : 0.0)
        << " Hz\n"
        << "Profiled wall time: " << profiled_seconds << " s\n"
        << "Map size: " << metadata.map_size_mb << " MB\n"
        << "Process RSS mean/peak: " << rss_mean << " / "
        << global.process_rss_peak_mb << " MB\n"
        << "PyTorch CUDA allocated mean/peak: " << cuda_alloc_mean << " / "
        << global.cuda_allocated_peak_mb << " MB\n"
        << "PyTorch CUDA reserved mean/peak: " << cuda_reserved_mean << " / "
        << global.cuda_reserved_peak_mb << " MB\n"
        << "Whole-device CUDA used mean/peak: " << cuda_device_mean << " / "
        << global.cuda_device_used_peak_mb << " MB\n\n"
        << "Module statistics\n"
        << "-----------------\n";
    txt << std::left << std::setw(38) << "module"
        << std::right << std::setw(9) << "calls"
        << std::setw(13) << "total ms"
        << std::setw(13) << "mean ms"
        << std::setw(13) << "RSS mean"
        << std::setw(13) << "RSS peak"
        << std::setw(13) << "CUDA mean"
        << std::setw(13) << "CUDA peak" << '\n';
    for (const auto& item : modules) {
        const ModuleStats& stats = item.second;
        const std::uint64_t samples = stats.memory.samples;
        txt << std::left << std::setw(38) << item.first
            << std::right << std::setw(9) << stats.calls
            << std::setw(13) << stats.total_wall_ms
            << std::setw(13) << average(stats.total_wall_ms, stats.calls)
            << std::setw(13)
            << average(stats.memory.process_rss_sum_mb, samples)
            << std::setw(13) << stats.memory.process_rss_peak_mb
            << std::setw(13)
            << average(stats.memory.cuda_allocated_sum_mb, samples)
            << std::setw(13) << stats.memory.cuda_allocated_peak_mb << '\n';
    }
    txt << "\nInterpretation\n"
        << "--------------\n"
        << "Times are host-observed and nested/asynchronous modules can overlap.\n"
        << "Memory is whole-process memory while a module is active, not exclusive module ownership.\n"
        << "CUDA allocated/reserved is PyTorch-only; whole-device CUDA usage also includes other CUDA libraries and processes.\n"
        << "Jetson uses unified memory, so RSS and CUDA columns overlap and must not be added together.\n";
}

} // namespace sv
