#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sv {

struct LaptopPrecheckMetadata
{
    std::filesystem::path config_file;
    std::filesystem::path map_path;
    std::string sensor;
    std::string device;
    std::string pipeline;
    int frames = 0;
    int keyframes = 0;
    int voxels = 0;
    int iterations = 0;
    double mapping_seconds = 0.0;
    double map_size_mb = 0.0;
};

// Low-overhead process profiler used to estimate whether a mapper configuration
// is suitable for a resource-constrained target. Memory is sampled for the
// complete process while a module is active; it is not exclusive ownership.
class LaptopPrecheckProfiler
{
public:
    class Scope
    {
    public:
        Scope() = default;
        Scope(
            LaptopPrecheckProfiler* profiler,
            std::string module,
            std::uint64_t work_items = 1);
        ~Scope();

        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&& other) noexcept;

        void finish();

    private:
        LaptopPrecheckProfiler* profiler_ = nullptr;
        std::string module_;
        std::uint64_t work_items_ = 0;
        std::chrono::steady_clock::time_point start_;
    };

    LaptopPrecheckProfiler(
        bool enabled,
        int sample_interval_ms,
        bool cuda_enabled);
    ~LaptopPrecheckProfiler();

    LaptopPrecheckProfiler(const LaptopPrecheckProfiler&) = delete;
    LaptopPrecheckProfiler& operator=(const LaptopPrecheckProfiler&) = delete;

    void start();
    void stop();
    bool enabled() const;

    Scope profile(
        const std::string& module,
        std::uint64_t work_items = 1);
    void beginAsync(
        const std::string& module,
        std::uint64_t work_items = 1);
    void endAsync(const std::string& module);

    void writeReports(
        const std::filesystem::path& output_dir,
        const LaptopPrecheckMetadata& metadata) const;

private:
    friend class Scope;

    using Clock = std::chrono::steady_clock;

    struct MemorySnapshot
    {
        double process_rss_mb = -1.0;
        double cuda_allocated_mb = -1.0;
        double cuda_reserved_mb = -1.0;
        double cuda_device_used_mb = -1.0;
    };

    struct MemoryAggregate
    {
        std::uint64_t samples = 0;
        double process_rss_sum_mb = 0.0;
        double process_rss_peak_mb = 0.0;
        double cuda_allocated_sum_mb = 0.0;
        double cuda_allocated_peak_mb = 0.0;
        double cuda_reserved_sum_mb = 0.0;
        double cuda_reserved_peak_mb = 0.0;
        double cuda_device_used_sum_mb = 0.0;
        double cuda_device_used_peak_mb = 0.0;

        void add(const MemorySnapshot& snapshot);
    };

    struct ModuleStats
    {
        std::uint64_t calls = 0;
        std::uint64_t work_items = 0;
        double total_wall_ms = 0.0;
        double max_wall_ms = 0.0;
        MemoryAggregate memory;
    };

    struct AsyncCall
    {
        Clock::time_point start;
        std::uint64_t work_items = 1;
    };

    void enter(const std::string& module);
    void leave(
        const std::string& module,
        Clock::time_point start,
        std::uint64_t work_items);
    void sampleNow();
    void samplingLoop();
    MemorySnapshot readMemorySnapshot() const;

    bool enabled_ = false;
    bool cuda_enabled_ = false;
    int sample_interval_ms_ = 50;

    mutable std::mutex mutex_;
    std::condition_variable sample_cv_;
    std::thread sampler_thread_;
    bool running_ = false;
    bool stop_requested_ = false;
    Clock::time_point run_start_;
    Clock::time_point run_end_;

    MemoryAggregate global_memory_;
    std::map<std::string, ModuleStats> modules_;
    std::unordered_map<std::string, std::uint64_t> active_modules_;
    std::unordered_map<std::string, std::vector<AsyncCall>> async_calls_;
};

} // namespace sv
