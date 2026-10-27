#include "memory_stats.h"

#include <ros/console.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace memory_stats {

void Snapshot::add(const std::string& key, std::uint64_t payload,
                   std::uint64_t capacity, std::uint64_t count,
                   std::uint64_t overhead) {
  Component& result = components[key];
  result.payload_bytes += payload;
  result.capacity_bytes += capacity;
  result.overhead_estimate_bytes += overhead;
  result.objects += count;
}

bool Snapshot::object(const std::string& key, const void* identity,
                      std::size_t bytes) {
  Component& result = components[key];
  if (!identity) return false;
  if (!seen_objects_.insert(identity).second) {
    ++result.shared_references;
    return false;
  }
  add(key, bytes, bytes, 1);
  return true;
}

bool Snapshot::allocation(const std::string& key, const void* identity,
                          std::size_t payload, std::size_t capacity,
                          const std::string& domain) {
  Component& result = components[key];
  if (!identity) return false;
  if (!seen_allocations_.insert({domain, identity}).second) {
    ++result.shared_references;
    return false;
  }
  add(key, payload, capacity);
  return true;
}

void Snapshot::mat(const std::string& key, const cv::Mat& value) {
  if (value.u) {
    // UMatData::size is the retained base allocation, even for ROI headers.
    allocation(key, value.u, value.u->size, value.u->size, "cv::Mat.storage");
  } else if (value.datastart && value.dataend) {
    // Externally owned Mat: only its known data span is observable, and it may
    // not be heap-owned. The coverage note makes this lower-bound case explicit.
    const std::size_t bytes = static_cast<std::size_t>(value.dataend - value.datastart);
    allocation(key, value.datastart, bytes, bytes, "cv::Mat.external");
  } else {
    add(key, 0, 0);
  }
}

void Snapshot::text(const std::string& key, const std::string& value) {
  const std::uintptr_t data = reinterpret_cast<std::uintptr_t>(value.data());
  const std::uintptr_t owner = reinterpret_cast<std::uintptr_t>(&value);
  if (data >= owner && data - owner < sizeof(value)) {
    add(key, 0, 0);  // SSO bytes are already inside the string/object header.
    return;
  }
  allocation(key, value.data(), value.size() + 1, value.capacity() + 1);
}

Component Snapshot::total() const {
  Component result;
  for (const auto& entry : components) {
    result.payload_bytes += entry.second.payload_bytes;
    result.capacity_bytes += entry.second.capacity_bytes;
    result.overhead_estimate_bytes += entry.second.overhead_estimate_bytes;
    result.objects += entry.second.objects;
    result.shared_references += entry.second.shared_references;
  }
  return result;
}

namespace {
using Clock = std::chrono::steady_clock;
using Values = std::array<long double, 6>;
using Components = std::map<std::string, Component>;
const char* const kTotal = "__accounted_total__";

Values values(const Component& value) {
  return {{static_cast<long double>(value.payload_bytes),
           static_cast<long double>(value.capacity_bytes),
           static_cast<long double>(value.overhead_estimate_bytes),
           static_cast<long double>(value.objects),
           static_cast<long double>(value.shared_references),
           static_cast<long double>(value.capacity_bytes) + value.overhead_estimate_bytes}};
}

bool equal(const Component& a, const Component& b) {
  return a.payload_bytes == b.payload_bytes && a.capacity_bytes == b.capacity_bytes &&
         a.overhead_estimate_bytes == b.overhead_estimate_bytes &&
         a.objects == b.objects && a.shared_references == b.shared_references;
}

bool equivalent(const Components& a, const Components& b) {
  // Categories omitted from a later snapshot mean released (zero) storage.
  const Component zero;
  for (const auto& entry : a) {
    const auto found = b.find(entry.first);
    if (!equal(entry.second, found == b.end() ? zero : found->second)) return false;
  }
  for (const auto& entry : b) {
    const auto found = a.find(entry.first);
    if (!equal(entry.second, found == a.end() ? zero : found->second)) return false;
  }
  return true;
}

std::string csv(const std::string& value) {
  std::string result = "\"";
  for (char ch : value) {
    if (ch == '"') result += '"';
    result += ch;
  }
  result += '"';
  return result;
}

void number(std::ostream& stream, double value) {
  if (std::isfinite(value)) stream << value;
  // Missing/nonfinite timestamps are blank, never a fabricated zero.
}

struct ProcessMemory {
  // VmRSS, VmHWM, RssAnon, RssFile, RssShmem, VmSwap; absent fields stay blank.
  std::array<std::optional<std::uint64_t>, 6> bytes;
};

ProcessMemory processMemory() {
  ProcessMemory result;
#ifdef __linux__
  std::ifstream status("/proc/self/status");
  const std::array<std::string, 6> names =
      {{"VmRSS:", "VmHWM:", "RssAnon:", "RssFile:", "RssShmem:", "VmSwap:"}};
  std::string line;
  while (std::getline(status, line)) {
    std::istringstream field(line);
    std::string name;
    field >> name;
    const auto found = std::find(names.begin(), names.end(), name);
    if (found == names.end()) continue;
    std::uint64_t amount = 0;
    std::string unit;
    if (field >> amount >> unit) {
      if (unit == "kB" && amount <= std::numeric_limits<std::uint64_t>::max() / 1024) {
        result.bytes[static_cast<std::size_t>(found - names.begin())] = amount * 1024;
      }
    }
  }
#endif
  return result;
}

std::string mib(const std::optional<std::uint64_t>& bytes) {
  if (!bytes) return "unavailable";
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(2) << static_cast<double>(*bytes) / 1048576.0;
  return stream.str();
}

struct SampleLocation {
  std::size_t sample_id;
  double wall;
  double data_time;
  int frame;
  std::string stage;
  bool statistical = true;
};

struct Accumulator {
  Values current{};
  Values peak{};
  Values sum{};
  Values integral{};
  std::optional<SampleLocation> accounted_peak_location;
  void update(const Component& component, double previous_hold_sec,
              const SampleLocation& location) {
    const Values next = values(component);
    if (!accounted_peak_location || next[5] > peak[5]) accounted_peak_location = location;
    for (std::size_t i = 0; i < current.size(); ++i) {
      integral[i] += current[i] * previous_hold_sec;
      current[i] = next[i];
      peak[i] = std::max(peak[i], next[i]);
      sum[i] += next[i];
    }
  }
};

// Unlike a released business component, an unreadable OS field is unknown,
// never zero. Means therefore use valid observations/hold intervals only.
struct OptionalAccumulator {
  std::optional<long double> current;
  std::optional<long double> peak;
  std::optional<SampleLocation> peak_location;
  std::size_t valid_count = 0;
  long double sum = 0;
  long double integral = 0;
  double valid_duration = 0;

  void update(std::optional<long double> next, double previous_hold_sec,
              const SampleLocation& location) {
    if (current && previous_hold_sec > 0) {
      integral += *current * previous_hold_sec;
      valid_duration += previous_hold_sec;
    }
    current = next;
    if (!next) return;
    ++valid_count;
    sum += *next;
    observePeak(next, location);
  }

  void observeFinal(std::optional<long double> next, const SampleLocation& location) {
    current = next;
    observePeak(next, location);
  }

  void observePeak(std::optional<long double> next, const SampleLocation& location) {
    if (!next) return;
    if (!peak || *next > *peak) {
      peak = next;
      peak_location = location;  // Ties retain the earliest observed peak.
    }
  }

  std::optional<long double> mean() const {
    if (valid_count == 0) return std::nullopt;
    return sum / static_cast<long double>(valid_count);
  }

  std::optional<long double> timeMean() const {
    if (valid_duration <= 0) return std::nullopt;
    return integral / valid_duration;
  }
};

const std::array<const char*, 7> kProcessMetrics =
    {{"VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem", "VmSwap", "collector_ms"}};

std::string metricText(std::optional<long double> value, long double divisor = 1) {
  if (!value) return "unavailable";
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(3) << *value / divisor;
  return stream.str();
}

const char* const kCoverage = R"COVERAGE(cake_slam memory statistics

Scope and byte semantics
------------------------
These are capacity snapshots of traversed business objects and retained buffers.
payload_bytes is live content; capacity_bytes INCLUDES payload and retained spare
capacity. overhead_estimate_bytes is additional estimated node/bucket overhead.
Do not add payload and capacity together. capacity + overhead_estimate is the
reported accounted estimate, NOT a measurement of the complete heap.
accounted_estimate_bytes records capacity + overhead_estimate at each sample;
its peak/mean are computed on that combined value, not by summing metric peaks.
Object headers include inline/fixed Eigen members; the Eigen helper only adds
dynamic coefficients (unbounded dynamic storage). Strings exclude SSO storage;
heap string capacity includes its terminating character. vector uses size and
capacity. Ordered node overhead assumes three pointers, color and max_align_t
padding; unordered nodes assume next/hash words plus bucket pointers. These are
ABI-dependent estimates and exclude allocator metadata. list adds two link words
per live element (alignment/sentinel/allocator overhead may be missing).
deque reports LIVE ELEMENT BYTES ONLY as a capacity lower bound; spare blocks and
its private pointer map are unknown, NOT zero measured overhead. Dynamic Eigen
bytes are its current coefficient-storage lower bound, excluding alignment/slack.
Mat storage uses UMatData identity and its whole retained allocation size, even
for ROIs. External Mat storage uses its observable base/data-end span; ownership,
spare capacity, and cross-aliasing with external container buffers are unknown.
Custom allocators, over-alignment and library private reserves may add storage.
objects means collected objects or container entries as appropriate for that row.

Deduplication and attribution
----------------------------
Within each snapshot, object addresses are globally deduplicated. Dynamic storage
is globally deduplicated by (domain, identity). Shared references add no bytes.
The FIRST collector visiting storage owns its bytes. VIO reference images should
be traversed before shared frame/camera views; attribution reflects traversal
priority, not an independent ownership proof. Nested object headers contained in
an already-counted owner must not be counted a second time by collectors.
Snapshot registries/statistics and CSV formatting also consume process memory;
their own allocations are not attributed to business components.
process_samples.csv collector_ms is the caller-measured complete collector wall
time; it excludes reporter aggregation, /proc reads, formatting and file writes.
Diagnostics are not zero-overhead. RSS includes the diagnostics themselves and
component capacity is not RSS. Run timing benchmarks separately with statistics
disabled/enabled as appropriate; do not confuse their cost with SLAM stage cost.

Sampling and summaries
----------------------
All sample/print intervals and weighted means use steady-clock WALL time, not
data/simulation time. First sample_id is 0; every accepted sample is flushed.
memory_samples.csv is long-form: one component per sample, plus the reserved
__accounted_total__ row. An absent known component is zero (released storage).
Before a category first appears, its value is also zero for summary statistics.
New category rows are not retroactively appended to old sample records.
memory_summary.csv is rewritten/flushed after EVERY sample so an interrupted
run retains a current summary. Its statistic rows are current, sampled_peak,
arithmetic_mean, time_weighted_mean, and final (when finalized).
Arithmetic mean = sum of accepted sample values / sample_count (equal weight).
Time-weighted mean = sum(value[i] * (wall[i+1]-wall[i])) / (wall[last]-wall[0]):
a left-continuous/previous-sample hold over the observed sample interval. It is
blank when fewer than two distinct wall times exist. There is no extrapolation
after the last statistical sample. The observation window starts at the first
accepted sample and ends at the last accepted statistical sample: startup waits,
input idle and pauses INSIDE that window are included in wall-time weighting;
time before the first sample is excluded. A component's sampled_peak may miss short
peaks between sampling stages. Each peak column is a separate metric maximum,
so capacity and overhead maxima need not occur in the same snapshot. Use the
accounted_estimate_bytes sampled_peak for their combined sampled peak.
memory_summary.csv peak_basis/peak_* identify the earliest observed maximum of
accounted_estimate_bytes specifically; latest data_time/frame/stage identify the
latest snapshot and are NOT peak timestamps. Other metric peaks may be at other
times. Ties for the combined peak retain the earliest sample location.
A final record bypasses interval throttling. If its data_time, frame, and values
match the previous sample (ignoring omitted zero categories), business final
metadata/summary are updated without changing sample_count or means. Otherwise
it is one new statistical sample. finalize() marks the latest business snapshot
final without taking another business sample or extending the weighted-mean
observation interval. Both duplicate final and finalize() refresh process fields
as an extra final observation (see below), so final RSS does not reuse stale RSS.
Wall time is elapsed since reporter creation; data_time/frame/stage identify the
SLAM state. File errors disable file output with a named ROS error; they must not
terminate SLAM. A disabled reporter opens no files and allocates no sampler state.

Process memory and limits
-------------------------
process_samples.csv is a separate Linux /proc/self/status observation: VmRSS,
VmHWM, RssAnon, RssFile, RssShmem and VmSwap, converted from kB to bytes. VmHWM is
the kernel process RSS high-water mark; component sampled_peak is not VmHWM.
Unsupported or missing process fields are blank, not fabricated zero values.
process_summary.csv is rewritten/flushed after each accepted sample and on
finalization. For each OS metric and collector_ms it contains current,
sampled_peak (VmHWM instead names this kernel_high_water_max), arithmetic_mean,
time_weighted_mean, and final rows. A missing latest field leaves current/final
blank. Arithmetic means divide by valid_sample_count, excluding missing samples.
Time means use the same previous-sample hold, exclude intervals whose starting
sample has a missing field, and divide by valid_wall_duration_sec. Missing and
valid sample counts/valid wall duration expose coverage; no value extrapolates
past the last accepted sample. With no valid values/intervals those means are
blank. An extra final process observation refreshes current/final and observed
peaks but DOES NOT add to statistical sums, valid_sample_count, total_sample_count
or wall-weighted intervals; means retain their original sampling window. Its
process_samples.csv row has is_statistical_sample=0 and reuses the last business
sample_id with its own wall time. Regular accepted observations have this flag=1.
process_summary.csv final_observation_wall_elapsed_sec records its actual time;
final_observation_is_extra distinguishes this refresh from a regular final sample.
Peak metadata also records whether that observation is statistical. A fallback
finalize() has no new collector run, so collector_ms remains the latest measured
collector duration, while /proc fields are read again.
VmRSS sampled_peak is the maximum RSS observed by this diagnostic. VmHWM
kernel_high_water_max is the maximum OBSERVED kernel high-water value, which
can include a process peak before collection began or between samples.
peak_observed_* fields identify the earliest sample observing each metric's
maximum (ties keep the first). For VmHWM this is the observation time, NOT the
unknown actual time at which the kernel's historical RSS peak occurred.
RSS contains ROS queues, OpenCV/PCL/Eigen/other library private heaps, stacks,
allocator reserves/fragmentation, mapped files/shared pages and these diagnostics.
Private ROS/library allocations and function-local transient heaps require a
heap profiler such as heaptrack for allocation/call-stack attribution. Short
transient peaks may be completely absent from these capacity snapshots.
NO unclassified module is derived by subtracting accounted bytes from RSS:
those measurements have different coverage and physical/reserved semantics.
)COVERAGE";
}  // namespace

struct Reporter::Impl {
  explicit Impl(Config value) : config(std::move(value)), started(Clock::now()) {
    if (!std::isfinite(config.sample_interval_sec) || config.sample_interval_sec < 0)
      config.sample_interval_sec = 1.0;
    if (!std::isfinite(config.print_interval_sec) || config.print_interval_sec < 0)
      config.print_interval_sec = 10.0;
  }

  Config config;
  Clock::time_point started;
  Clock::time_point last_sample;
  Clock::time_point last_print;
  bool sampled = false;
  bool printed = false;
  bool finalized = false;
  bool files_enabled = false;
  std::size_t count = 0;
  double first_wall = 0;
  double last_wall = 0;
  double data_time = std::numeric_limits<double>::quiet_NaN();
  double collection_ms = 0;
  int frame = -1;
  std::string stage;
  Components current;
  std::map<std::string, Accumulator> stats;
  Accumulator total_stats;
  ProcessMemory process;
  std::array<OptionalAccumulator, 7> process_stats;
  std::optional<double> final_process_wall;
  bool extra_final_process_observation = false;
  std::filesystem::path directory;
  std::ofstream samples_file;
  std::ofstream process_file;

  double elapsed(Clock::time_point point) const {
    return std::chrono::duration<double>(point - started).count();
  }

  void closeFiles() {
    files_enabled = false;
    samples_file.close();
    process_file.close();
  }

  void openFiles() {
    try {
      directory = config.output_dir.empty() ? "." : config.output_dir;
      std::filesystem::create_directories(directory);
      samples_file.open(directory / "memory_samples.csv", std::ios::out | std::ios::trunc);
      process_file.open(directory / "process_samples.csv", std::ios::out | std::ios::trunc);
      std::ofstream coverage(directory / "coverage.txt", std::ios::out | std::ios::trunc);
      if (!samples_file || !process_file || !coverage) throw std::runtime_error("cannot open output files");
      coverage << kCoverage;
      coverage.flush();
      samples_file << "sample_id,wall_elapsed_sec,data_time_sec,frame,stage,is_final,component,payload_bytes,capacity_bytes,overhead_estimate_bytes,objects,shared_references,accounted_estimate_bytes\n";
      process_file << "sample_id,wall_elapsed_sec,data_time_sec,frame,stage,is_final,is_statistical_sample,collector_ms,VmRSS_bytes,VmHWM_bytes,RssAnon_bytes,RssFile_bytes,RssShmem_bytes,VmSwap_bytes\n";
      samples_file.flush();
      process_file.flush();
      if (!coverage || !samples_file || !process_file) throw std::runtime_error("cannot write output headers");
      files_enabled = true;
      writeSummary();
      ROS_INFO_STREAM_NAMED("memory_stats", "Memory statistics output: " << directory.string());
    } catch (const std::exception& error) {
      closeFiles();
      ROS_ERROR_STREAM_NAMED("memory_stats", "Memory statistics file output disabled: " << error.what());
    }
  }

  void prefix(std::ostream& stream, std::size_t sample_id, bool final,
              std::optional<double> observed_wall = std::nullopt) const {
    stream << std::setprecision(17) << sample_id << ',' << observed_wall.value_or(last_wall) << ',';
    number(stream, data_time);
    stream << ',' << frame << ',' << csv(stage) << ',' << (final ? 1 : 0);
  }

  void writeSample(bool final) {
    if (!files_enabled) return;
    for (const auto& entry : current) {
      prefix(samples_file, count - 1, final);
      samples_file << ',' << csv(entry.first);
      for (long double value : values(entry.second)) samples_file << ',' << value;
      samples_file << '\n';
    }
    prefix(samples_file, count - 1, final);
    samples_file << ',' << csv(kTotal);
    for (long double value : total_stats.current) samples_file << ',' << value;
    samples_file << '\n';
    samples_file.flush();
    writeProcessObservation(final, true);
    if (!samples_file) throw std::runtime_error("cannot flush memory sample file");
  }

  void writeProcessObservation(bool final, bool statistical,
                               std::optional<double> observed_wall = std::nullopt) {
    if (!files_enabled) return;
    prefix(process_file, count - 1, final, observed_wall);
    process_file << ',' << (statistical ? 1 : 0) << ',';
    number(process_file, collection_ms);
    for (const auto& value : process.bytes) {
      process_file << ',';
      if (value) process_file << *value;
    }
    process_file << '\n';
    process_file.flush();
    if (!process_file) throw std::runtime_error("cannot flush process sample file");
  }

  void observeProcess(double hold_sec, const SampleLocation& location) {
    process = processMemory();
    for (std::size_t index = 0; index < process.bytes.size(); ++index) {
      std::optional<long double> value;
      if (process.bytes[index]) value = *process.bytes[index];
      if (location.statistical) process_stats[index].update(value, hold_sec, location);
      else process_stats[index].observeFinal(value, location);
    }
    std::optional<long double> collector_value;
    if (std::isfinite(collection_ms) && collection_ms >= 0) collector_value = collection_ms;
    if (location.statistical) process_stats[6].update(collector_value, hold_sec, location);
    else process_stats[6].observeFinal(collector_value, location);
    if (!location.statistical) {
      final_process_wall = location.wall;
      extra_final_process_observation = true;
    }
  }

  void summaryRows(std::ostream& stream, const std::string& key,
                   const Accumulator& accumulator) const {
    auto row = [&](const char* statistic, const Values& data, bool available) {
      stream << csv(key) << ',' << statistic;
      for (long double value : data) {
        stream << ',';
        if (available) stream << value;
      }
      stream << ',' << count << ',' << first_wall << ',' << last_wall << ',';
      number(stream, data_time);
      stream << ',' << frame << ',' << csv(stage) << ',' << (finalized ? 1 : 0)
             << ",accounted_estimate_bytes";
      if (accumulator.accounted_peak_location) {
        const SampleLocation& peak = *accumulator.accounted_peak_location;
        stream << ',' << peak.sample_id << ',' << peak.wall << ',';
        number(stream, peak.data_time);
        stream << ',' << peak.frame << ',' << csv(peak.stage);
      } else {
        stream << ",,,,,";
      }
      stream << '\n';
    };
    row("current", accumulator.current, true);
    row("sampled_peak", accumulator.peak, true);
    Values mean{};
    Values weighted{};
    const double duration = last_wall - first_wall;
    for (std::size_t i = 0; i < mean.size(); ++i) {
      mean[i] = accumulator.sum[i] / static_cast<long double>(count);
      if (duration > 0) weighted[i] = accumulator.integral[i] / duration;
    }
    row("arithmetic_mean", mean, true);
    row("time_weighted_mean", weighted, duration > 0);
    if (finalized) row("final", accumulator.current, true);
  }

  void writeSummary() {
    if (!files_enabled) return;
    std::ofstream summary(directory / "memory_summary.csv", std::ios::out | std::ios::trunc);
    summary << std::setprecision(17);
    summary << "component,statistic,payload_bytes,capacity_bytes,overhead_estimate_bytes,objects,shared_references,accounted_estimate_bytes,sample_count,first_wall_elapsed_sec,last_wall_elapsed_sec,data_time_sec,frame,stage,is_final,peak_basis,peak_sample_id,peak_wall_elapsed_sec,peak_data_time_sec,peak_frame,peak_stage\n";
    if (sampled) {
      for (const auto& entry : stats) summaryRows(summary, entry.first, entry.second);
      summaryRows(summary, kTotal, total_stats);
    }
    summary.flush();
    if (!summary) throw std::runtime_error("cannot update memory_summary.csv");
    writeProcessSummary();
  }

  void writeProcessSummary() {
    std::ofstream summary(directory / "process_summary.csv", std::ios::out | std::ios::trunc);
    summary << std::setprecision(17);
    summary << "metric,unit,statistic,value,total_sample_count,valid_sample_count,missing_sample_count,valid_wall_duration_sec,first_wall_elapsed_sec,last_wall_elapsed_sec,data_time_sec,frame,stage,is_final,peak_observed_sample_id,peak_observed_wall_elapsed_sec,peak_observed_data_time_sec,peak_observed_frame,peak_observed_stage,peak_observed_is_statistical_sample,final_observation_wall_elapsed_sec,final_observation_is_extra\n";
    if (sampled) {
      for (std::size_t index = 0; index < process_stats.size(); ++index) {
        const OptionalAccumulator& accumulator = process_stats[index];
        auto row = [&](const char* statistic, std::optional<long double> value) {
          summary << csv(kProcessMetrics[index]) << ',' << (index == 6 ? "ms" : "bytes")
                  << ',' << statistic << ',';
          if (value) summary << *value;
          summary << ',' << count << ',' << accumulator.valid_count << ','
                  << count - accumulator.valid_count << ',' << accumulator.valid_duration
                  << ',' << first_wall << ',' << last_wall << ',';
          number(summary, data_time);
          summary << ',' << frame << ',' << csv(stage) << ',' << (finalized ? 1 : 0);
          if (accumulator.peak_location) {
            const SampleLocation& peak = *accumulator.peak_location;
            summary << ',' << peak.sample_id << ',' << peak.wall << ',';
            number(summary, peak.data_time);
            summary << ',' << peak.frame << ',' << csv(peak.stage) << ',' << (peak.statistical ? 1 : 0);
          } else {
            summary << ",,,,,,";
          }
          summary << ',';
          if (final_process_wall) summary << *final_process_wall;
          summary << ',' << (extra_final_process_observation ? 1 : 0) << '\n';
        };
        row("current", accumulator.current);
        row(index == 1 ? "kernel_high_water_max" : "sampled_peak", accumulator.peak);
        row("arithmetic_mean", accumulator.mean());
        row("time_weighted_mean", accumulator.timeMean());
        if (finalized) row("final", accumulator.current);
      }
    }
    summary.flush();
    if (!summary) throw std::runtime_error("cannot update process_summary.csv");
  }

  void print(bool final, Clock::time_point now) {
    if (!sampled) return;
    if (!final && printed &&
        std::chrono::duration<double>(now - last_print).count() < config.print_interval_sec) return;
    const double duration = last_wall - first_wall;
    ROS_INFO_STREAM_NAMED("memory_stats", "[Memory] " << (final ? "Final memory statistics" : "Memory statistics")
        << " sample=" << count - 1 << " count=" << count << " frame=" << frame
        << " stage=" << stage << " wall=" << std::fixed << std::setprecision(3) << last_wall
        << "s RSS=" << mib(process.bytes[0]) << " MiB VmHWM=" << mib(process.bytes[1])
        << " MiB; component peaks are sampled peaks");
    const OptionalAccumulator& rss = process_stats[0];
    const OptionalAccumulator& hwm = process_stats[1];
    const OptionalAccumulator& collector = process_stats[6];
    ROS_INFO_STREAM_NAMED("memory_stats", "[Memory] Process RSS current=" << metricText(rss.current, 1048576)
        << " MiB sampled_peak=" << metricText(rss.peak, 1048576)
        << " MiB mean=" << metricText(rss.mean(), 1048576)
        << " MiB time_mean=" << metricText(rss.timeMean(), 1048576)
        << " MiB kernel_VmHWM_max=" << metricText(hwm.peak, 1048576)
        << " MiB collector_ms_mean=" << metricText(collector.mean())
        << " collector_ms_peak=" << metricText(collector.peak));
    auto component = [&](const std::string& key, const Accumulator& accumulator) {
      const long double current_estimate = accumulator.current[5];
      // Do not sum independent metric maxima to invent a peak total.
      std::ostringstream stream;
      stream << std::fixed << std::setprecision(3) << key
             << " payload=" << accumulator.current[0] / 1048576.0L
             << " MiB capacity=" << accumulator.current[1] / 1048576.0L
             << " MiB overhead_est=" << accumulator.current[2] / 1048576.0L
             << " MiB accounted_est=" << current_estimate / 1048576.0L
             << " MiB sampled_peak_est=" << accumulator.peak[5] / 1048576.0L
             << " MiB mean_est=" << (accumulator.sum[5] / count) / 1048576.0L;
      if (duration > 0) stream << " MiB time_mean_est=" << (accumulator.integral[5] / duration) / 1048576.0L;
      else stream << " MiB time_mean_est=unavailable";
      stream << " MiB objects=" << accumulator.current[3]
             << " shared_refs=" << accumulator.current[4];
      if (final) stream << " final_est=" << current_estimate / 1048576.0L << " MiB";
      ROS_INFO_STREAM_NAMED("memory_stats", "[Memory] " << stream.str());
    };
    for (const auto& entry : stats) component(entry.first, entry.second);
    component(kTotal, total_stats);
    last_print = now;
    printed = true;
  }
};

Reporter::Reporter(Config config) noexcept {
  if (!config.enabled) return;
  try {
    // This named INFO logger remains visible when the SLAM root logger is WARN.
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME ".memory_stats", ros::console::levels::Info);
    ros::console::notifyLoggerLevelsChanged();
    impl_.reset(new Impl(std::move(config)));
    impl_->openFiles();
  } catch (const std::exception& error) {
    ROS_ERROR_STREAM_NAMED("memory_stats", "Cannot initialize memory statistics: " << error.what());
  } catch (...) {
    ROS_ERROR_NAMED("memory_stats", "Cannot initialize memory statistics: unknown error");
  }
}

Reporter::~Reporter() noexcept { finalize(); }

bool Reporter::due(bool force) const noexcept {
  if (!impl_ || impl_->finalized) return false;
  return force || !impl_->sampled ||
      std::chrono::duration<double>(Clock::now() - impl_->last_sample).count() >=
          impl_->config.sample_interval_sec;
}

void Reporter::record(Snapshot snapshot, double data_time, int frame,
                       const std::string& stage, bool final) noexcept {
  if (!impl_ || impl_->finalized || (!final && !due())) return;
  Impl& state = *impl_;
  try {
    const auto now = Clock::now();
    const bool same_data_time = data_time == state.data_time ||
        (std::isnan(data_time) && std::isnan(state.data_time));
    const bool duplicate_final = final && state.sampled && same_data_time &&
        frame == state.frame && equivalent(snapshot.components, state.current);
    if (!duplicate_final) {
      const double wall = state.elapsed(now);
      const double hold_sec = state.sampled ? wall - state.last_wall : 0;
      const SampleLocation location{state.count, wall, data_time, frame, stage};
      // Preserve every known category with zero-valued rows after release.
      for (const auto& previous : state.stats) snapshot.components.emplace(previous.first, Component{});
      state.current = std::move(snapshot.components);
      Component total;
      for (const auto& entry : state.current) {
        state.stats[entry.first].update(entry.second, hold_sec, location);
        total.payload_bytes += entry.second.payload_bytes;
        total.capacity_bytes += entry.second.capacity_bytes;
        total.overhead_estimate_bytes += entry.second.overhead_estimate_bytes;
        total.objects += entry.second.objects;
        total.shared_references += entry.second.shared_references;
      }
      state.total_stats.update(total, hold_sec, location);
      if (!state.sampled) state.first_wall = wall;
      state.last_wall = wall;
      state.last_sample = now;
      state.sampled = true;
      ++state.count;
      state.collection_ms = snapshot.collection_ms;
      state.observeProcess(hold_sec, location);
      if (final) state.final_process_wall = wall;
    } else {
      state.collection_ms = snapshot.collection_ms;
      state.observeProcess(0, {state.count - 1, state.elapsed(now), data_time, frame, stage, false});
    }
    state.data_time = data_time;
    state.frame = frame;
    state.stage = stage;
    state.finalized = final;
    try {
      if (!duplicate_final) state.writeSample(final);
      else state.writeProcessObservation(true, false, state.final_process_wall);
      state.writeSummary();
    } catch (const std::exception& error) {
      state.closeFiles();
      ROS_ERROR_STREAM_NAMED("memory_stats", "Memory statistics file output disabled: " << error.what());
    }
    state.print(final, now);
    if (final) state.closeFiles();
  } catch (const std::exception& error) {
    state.finalized = true;
    state.closeFiles();
    ROS_ERROR_STREAM_NAMED("memory_stats", "Memory statistics disabled after sample error: " << error.what());
  } catch (...) {
    state.finalized = true;
    state.closeFiles();
    ROS_ERROR_NAMED("memory_stats", "Memory statistics disabled after sample error: unknown error");
  }
}

void Reporter::finalize() noexcept {
  if (!impl_ || impl_->finalized) return;
  Impl& state = *impl_;
  state.finalized = true;
  try {
    if (state.sampled) {
      const auto now = Clock::now();
      state.observeProcess(0, {state.count - 1, state.elapsed(now), state.data_time,
                              state.frame, state.stage, false});
      state.writeProcessObservation(true, false, state.final_process_wall);
    }
    state.writeSummary();
  } catch (const std::exception& error) {
    state.closeFiles();
    ROS_ERROR_STREAM_NAMED("memory_stats", "Cannot finalize memory statistics: " << error.what());
  } catch (...) {
    state.closeFiles();
    ROS_ERROR_NAMED("memory_stats", "Cannot finalize memory statistics: unknown error");
  }
  try {
    state.print(true, Clock::now());
  } catch (...) {
    ROS_ERROR_NAMED("memory_stats", "Cannot print final memory statistics");
  }
  state.closeFiles();
}

}  // namespace memory_stats
