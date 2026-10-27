#pragma once

#include <Eigen/Core>
#include <opencv2/core/mat.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace memory_stats {

// capacity_bytes includes payload_bytes; overhead_estimate_bytes is additional.
// Container counts are element counts; object() counts concrete objects. These
// counts are diagnostic and need not describe the same kind of object per row.
struct Component {
  std::uint64_t payload_bytes = 0;
  std::uint64_t capacity_bytes = 0;
  std::uint64_t overhead_estimate_bytes = 0;
  std::uint64_t objects = 0;
  std::uint64_t shared_references = 0;
};

namespace detail {
template <typename T, typename = void>
struct HasBuckets : std::false_type {};
template <typename T>
struct HasBuckets<T, std::void_t<decltype(std::declval<const T&>().bucket_count())>>
    : std::true_type {};
}  // namespace detail

// A snapshot owns its deduplication registry. The first collector to encounter
// an object/allocation receives its bytes; later references receive only a
// shared_references increment, including references from another component.
// Object headers and dynamic allocations occupy separate identity domains.
struct Snapshot {
  std::map<std::string, Component> components;
  double collection_ms = 0.0;

  void add(const std::string& key, std::uint64_t payload,
           std::uint64_t capacity, std::uint64_t count = 0,
           std::uint64_t overhead = 0);
  bool object(const std::string& key, const void* identity, std::size_t bytes);
  bool allocation(const std::string& key, const void* identity,
                  std::size_t payload, std::size_t capacity,
                  const std::string& domain = "buffer");

  // Only dynamic storage is included by these helpers. Count the owning
  // object's inline header separately with object() when appropriate.
  template <typename T, typename A>
  void vector(const std::string& key, const std::vector<T, A>& value) {
    static_assert(!std::is_same<T, bool>::value,
                  "vector<bool> has no portable allocation identity");
    if (allocation(key, value.data(), value.size() * sizeof(T),
                   value.capacity() * sizeof(T))) {
      components[key].objects += value.size();
    }
  }

  template <typename AssociativeContainer>
  void map(const std::string& key, const AssociativeContainer& value) {
    using Value = typename AssociativeContainer::value_type;
    const std::size_t bytes = value.size() * sizeof(Value);
    // Container address identifies its private node allocation set. Node and
    // bucket overhead is an ABI-dependent estimate, not allocator measurement.
    if (!allocation(key, &value, bytes, bytes, "associative.nodes")) return;
    components[key].objects += value.size();
    if constexpr (detail::HasBuckets<AssociativeContainer>::value) {
      // next pointer + possible cached hash; buckets contain node pointers.
      // An empty default container can use a shared singleton bucket.
      const std::size_t buckets = value.empty() && value.bucket_count() <= 1
                                      ? 0 : value.bucket_count();
      components[key].overhead_estimate_bytes +=
          (2 * value.size() + buckets) * sizeof(void*);
    } else {
      // Typical tree node: parent/left/right plus color and alignment padding.
      constexpr std::size_t alignment = alignof(std::max_align_t);
      constexpr std::size_t raw = sizeof(Value) + 3 * sizeof(void*) + sizeof(int);
      constexpr std::size_t node = ((raw + alignment - 1) / alignment) * alignment;
      components[key].overhead_estimate_bytes += value.size() * (node - sizeof(Value));
    }
  }

  template <typename T, typename A>
  void sequence(const std::string& key, const std::list<T, A>& value) {
    const std::size_t bytes = value.size() * sizeof(T);
    if (!allocation(key, &value, bytes, bytes, "list.nodes")) return;
    components[key].objects += value.size();
    components[key].overhead_estimate_bytes += value.size() * 2 * sizeof(void*);
  }

  template <typename T, typename A>
  void sequence(const std::string& key, const std::deque<T, A>& value) {
    const std::size_t bytes = value.size() * sizeof(T);
    if (allocation(key, &value, bytes, bytes, "deque.live")) {
      components[key].objects += value.size();
    }
    // deque's spare blocks, internal pointer map and allocator metadata are
    // unavailable through its public API. capacity_bytes is a live-byte lower
    // bound here; zero overhead estimate does NOT assert zero actual overhead.
  }

  template <typename Matrix>
  void eigen(const std::string& key, const Matrix& value) {
    // Fixed-size and bounded inline Eigen storage is already in object().
    if constexpr (Matrix::SizeAtCompileTime == Eigen::Dynamic &&
                  Matrix::MaxSizeAtCompileTime == Eigen::Dynamic) {
      const std::size_t bytes = static_cast<std::size_t>(value.size()) *
                                sizeof(typename Matrix::Scalar);
      allocation(key, value.data(), bytes, bytes);
    }
  }

  void mat(const std::string& key, const cv::Mat& value);
  void text(const std::string& key, const std::string& value);
  Component total() const;

 private:
  struct AllocationIdentity {
    std::string domain;
    const void* pointer;
    bool operator<(const AllocationIdentity& other) const {
      if (domain != other.domain) return domain < other.domain;
      return std::less<const void*>()(pointer, other.pointer);
    }
  };
  std::set<const void*, std::less<const void*>> seen_objects_;
  std::set<AllocationIdentity> seen_allocations_;
};

struct Config {
  bool enabled = false;
  double sample_interval_sec = 1.0;
  double print_interval_sec = 10.0;
  std::string output_dir;
};

// No global instance or background worker. Call from the SLAM owner thread at
// safe stages. Disabled reporters allocate no Impl and open no output files.
// Times used for intervals/weighted means are steady-clock wall times.
class Reporter {
 public:
  explicit Reporter(Config config) noexcept;
  ~Reporter() noexcept;
  Reporter(const Reporter&) = delete;
  Reporter& operator=(const Reporter&) = delete;
  bool due(bool force = false) const noexcept;
  void record(Snapshot snapshot, double data_time, int frame,
              const std::string& stage, bool final = false) noexcept;
  void finalize() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace memory_stats
