// Minimal owning fixtures for the actual production methods in the generated
// include. No lifecycle algorithm is reimplemented in this harness.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace
{
void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}
int deleted_points = 0;
int deleted_features = 0;
int deleted_voxels = 0;
}

struct Feature
{
  enum class RefState { CANDIDATE, VALIDATED, RETIRED };
  RefState ref_state_ = RefState::CANDIDATE;
  bool pending_delete_ = false;
  std::uint64_t ref_id_ = 0;
  int camera_id_ = 0;
  int last_test_frame_id_ = -1, last_test_camera_id_ = -1;
  int independent_test_count_ = 0, rejected_test_count_ = 0, consecutive_reject_count_ = 0;
  ~Feature() { ++deleted_features; }
};

struct VisualPoint
{
  enum class State { SEED, CONFIRMED, SUSPECT, RETIRED };
  State state_ = State::SEED;
  bool pending_delete_ = false;
  std::uint64_t point_id_ = 0;
  std::int64_t map_voxel_x_ = 0, map_voxel_y_ = 0, map_voxel_z_ = 0;
  int last_success_frame_ = -1;
  int independent_test_count_ = 0, rejected_test_count_ = 0, accepted_test_count_ = 0;
  std::list<Feature *> obs_;
  Feature *ref_patch = nullptr;
  bool has_ref_patch_ = false;
  std::vector<Feature *> ref_patch_by_camera_;
  std::vector<std::uint8_t> has_ref_patch_by_camera_;
  VisualPoint *challenger_of_ = nullptr;

  ~VisualPoint()
  {
    for (Feature *reference : obs_) delete reference;
    ++deleted_points;
  }
  void deleteFeatureRef(Feature *feature);
  bool hasUsableReference(int camera_id, bool cross_camera_reference, bool allow_candidate) const;
};

using VOXEL_LOCATION = std::tuple<std::int64_t, std::int64_t, std::int64_t>;
struct VOXEL_POINTS
{
  std::vector<VisualPoint *> voxel_points;
  int count = 0;
  ~VOXEL_POINTS()
  {
    for (VisualPoint *point : voxel_points) delete point;
    ++deleted_voxels;
  }
};

struct VIOManager
{
  bool visual_map_manage_en = true, visual_map_retirement_apply_en = true;
  bool visual_map_manage_shadow_en = false, visual_ref_lifecycle_en = true;
  int visual_ref_validate_min_tests = 3, visual_ref_retire_reject_count = 5;
  int visual_point_suspect_reject_count = 8, visual_point_stale_frames = 200;
  struct { int point_retired = 0, ref_retired = 0, seed_tested = 0; } visual_map_manage_stats_;
  std::map<VOXEL_LOCATION, VOXEL_POINTS *> feat_map;
  std::vector<VisualPoint *> retired_visual_points_;
  std::vector<std::pair<VisualPoint *, Feature *>> retired_visual_refs_;
  std::set<std::uint64_t> shadow_retired_point_suggestions_, shadow_retired_ref_suggestions_;
  const VisualPoint *runtime_support_dump_best_point_ = nullptr;

  ~VIOManager() { for (auto &voxel : feat_map) delete voxel.second; }
  void queueReferenceRetirement(VisualPoint &, Feature &);
  void queuePointRetirement(VisualPoint &);
  void flushVisualMapRetirements();
  void queueStaleVisualMapPoints(int frame_id);
  void recordManagedReferenceRejection(VisualPoint &, Feature &, int frame_id, int camera_id);
};

#include "visual_map_retirement_impl.inc"

namespace
{
VisualPoint *addPoint(VIOManager &manager, int voxel = 0)
{
  auto *point = new VisualPoint;
  point->point_id_ = static_cast<std::uint64_t>(voxel + 1);
  point->map_voxel_x_ = voxel;
  auto &owner = manager.feat_map[VOXEL_LOCATION(voxel, 0, 0)];
  if (owner == nullptr) owner = new VOXEL_POINTS;
  owner->voxel_points.push_back(point);
  owner->count = static_cast<int>(owner->voxel_points.size());
  return point;
}

Feature *addReference(VisualPoint &point, std::uint64_t id = 1)
{
  auto *feature = new Feature;
  feature->ref_id_ = id;
  point.obs_.push_back(feature);
  point.ref_patch = feature;
  point.has_ref_patch_ = true;
  point.ref_patch_by_camera_.assign(3, feature);
  point.has_ref_patch_by_camera_.assign(3, 1);
  return feature;
}

void testFiveRejectionsRetireSeedAtNextFlush()
{
  VIOManager manager;
  auto *point = addPoint(manager);
  auto *reference = addReference(*point);
  for (int frame = 1; frame <= 4; ++frame)
    manager.recordManagedReferenceRejection(*point, *reference, frame, 0);
  require(!reference->pending_delete_, "four rejections must not retire the reference");
  manager.recordManagedReferenceRejection(*point, *reference, 5, 0);
  require(reference->pending_delete_ && point->state_ == VisualPoint::State::SEED,
          "regression must reach reference retirement before seed becomes suspect");
  require(point->last_success_frame_ == -1 && point->rejected_test_count_ == 5,
          "regression must exercise a never-successful seed");
  require(manager.retired_visual_points_.empty(), "point must remain alive until the safe boundary");
  manager.runtime_support_dump_best_point_ = point;
  manager.flushVisualMapRetirements();
  require(manager.feat_map.empty(), "last-reference deletion must also remove the orphan seed");
  require(manager.runtime_support_dump_best_point_ == nullptr, "debug alias must be cleared");
  require(deleted_points == 1 && deleted_features == 1 && deleted_voxels == 1,
          "point, reference, and empty voxel must be freed exactly once");
  require(manager.visual_map_manage_stats_.point_retired == 1, "cascade must update retirement stats");
  manager.flushVisualMapRetirements();
  require(deleted_points == 1 && manager.retired_visual_refs_.empty() &&
          manager.retired_visual_points_.empty(), "repeated flush must be harmless");
}

void testRemainingCandidateIsPreserved()
{
  VIOManager manager;
  auto *point = addPoint(manager);
  auto *bad = addReference(*point, 1);
  auto *candidate = addReference(*point, 2);
  manager.queueReferenceRetirement(*point, *bad);
  manager.flushVisualMapRetirements();
  manager.queueStaleVisualMapPoints(1000);
  manager.flushVisualMapRetirements();
  require(point->obs_.size() == 1 && point->obs_.front() == candidate,
          "an unvalidated remaining reference must keep its seed alive");
  require(!point->pending_delete_ && !point->hasUsableReference(0, true, false),
          "no validated reference must not be confused with no observations");
  require(deleted_features == 1 && deleted_points == 0, "only the rejected reference should be freed");
}

void testCascadeClearsBacklinksAndPreservesVoxelNeighbours()
{
  VIOManager manager;
  auto *orphan = addPoint(manager);
  auto *first = addReference(*orphan, 1);
  auto *second = addReference(*orphan, 2);
  auto *challenger = addPoint(manager);  // Same voxel must survive.
  addReference(*challenger, 3);
  challenger->challenger_of_ = orphan;
  auto *other_challenger = addPoint(manager, 1);
  addReference(*other_challenger, 4);
  other_challenger->challenger_of_ = orphan;
  manager.queueReferenceRetirement(*orphan, *first);
  manager.queueReferenceRetirement(*orphan, *second);
  manager.queueReferenceRetirement(*orphan, *second);  // Must not duplicate the queue.
  manager.flushVisualMapRetirements();
  require(challenger->challenger_of_ == nullptr && other_challenger->challenger_of_ == nullptr,
          "all challenger backlinks must include newly cascaded retirements");
  const auto *voxel = manager.feat_map.at(VOXEL_LOCATION(0, 0, 0));
  require(voxel->count == 1 && voxel->voxel_points.front() == challenger,
          "removal must preserve the neighbouring point and update voxel count");
  require(deleted_points == 1 && deleted_features == 2 && deleted_voxels == 0,
          "multiple retired references must trigger just one point deletion");
}

void testExplicitPointAndReferenceRetirementDoNotDoubleDelete()
{
  VIOManager manager;
  auto *point = addPoint(manager);
  auto *first = addReference(*point, 1);
  auto *second = addReference(*point, 2);
  manager.queueReferenceRetirement(*point, *first);
  manager.queueReferenceRetirement(*point, *second);
  manager.queuePointRetirement(*point);
  manager.queuePointRetirement(*point);
  point->map_voxel_x_ = 99;  // Existing fallback lookup must still work.
  manager.flushVisualMapRetirements();
  require(manager.feat_map.empty() && deleted_points == 1 && deleted_features == 2,
          "point destructor must own references when the point was already queued");
}

void testPeriodicSweepFindsEmptyPointsInEveryState()
{
  VIOManager manager;
  for (auto state : {VisualPoint::State::SEED, VisualPoint::State::CONFIRMED,
                     VisualPoint::State::SUSPECT})
  {
    auto *point = addPoint(manager, static_cast<int>(manager.feat_map.size()));
    point->state_ = state;
  }
  manager.queueStaleVisualMapPoints(19);
  require(manager.retired_visual_points_.empty(), "keep the existing sweep cadence");
  manager.queueStaleVisualMapPoints(20);
  require(manager.retired_visual_points_.size() == 3 && deleted_points == 0,
          "queue empty points without requiring successful history or freeing frame aliases");
  manager.flushVisualMapRetirements();
  require(manager.feat_map.empty() && deleted_points == 3, "sweep must recover pre-existing orphans");
}

void testDisabledAndShadowModesDoNotDelete()
{
  for (int mode = 0; mode < 3; ++mode)
  {
    VIOManager manager;
    if (mode == 0) manager.visual_map_manage_en = false;
    if (mode == 1) manager.visual_map_manage_shadow_en = true;
    if (mode == 2) manager.visual_map_retirement_apply_en = false;
    auto *empty = addPoint(manager);
    auto *observed = addPoint(manager, 1);
    auto *reference = addReference(*observed);
    if (mode != 0) manager.queueReferenceRetirement(*observed, *reference);
    const int before = deleted_points;
    manager.queueStaleVisualMapPoints(20);
    manager.queueStaleVisualMapPoints(40);
    manager.flushVisualMapRetirements();
    require(manager.feat_map.size() == 2 && deleted_points == before &&
            !empty->pending_delete_ && !reference->pending_delete_,
            "disabled/shadow modes must not change ownership or mark actual deletion");
    if (mode != 0)
      require(manager.visual_map_manage_stats_.point_retired == 1,
              "shadow suggestions for an empty point must remain deduplicated");
  }
}

void testExistingStaleReferencePolicyIsPreserved()
{
  VIOManager manager;
  auto *stale = addPoint(manager);
  auto *candidate = addReference(*stale, 1);
  (void)candidate;
  stale->state_ = VisualPoint::State::SUSPECT;
  stale->last_success_frame_ = 20;
  auto *validated = addPoint(manager, 1);
  addReference(*validated, 2)->ref_state_ = Feature::RefState::VALIDATED;
  validated->state_ = VisualPoint::State::SUSPECT;
  validated->last_success_frame_ = 20;
  manager.queueStaleVisualMapPoints(200);
  require(manager.retired_visual_points_.empty(), "must retain the configured stale interval");
  manager.queueStaleVisualMapPoints(220);
  manager.flushVisualMapRetirements();
  require(manager.feat_map.size() == 1 && manager.feat_map.begin()->second->voxel_points.front() == validated,
          "existing stale policy must retain points with validated references");
}
}

int main()
{
  const std::pair<const char *, void (*)()> tests[] = {
      {"five rejected observations", testFiveRejectionsRetireSeedAtNextFlush},
      {"remaining candidate", testRemainingCandidateIsPreserved},
      {"cascade backlinks and neighbours", testCascadeClearsBacklinksAndPreservesVoxelNeighbours},
      {"simultaneous point/reference retirement", testExplicitPointAndReferenceRetirementDoNotDoubleDelete},
      {"periodic orphan sweep", testPeriodicSweepFindsEmptyPointsInEveryState},
      {"disabled and shadow modes", testDisabledAndShadowModesDoNotDelete},
      {"existing stale policy", testExistingStaleReferencePolicyIsPreserved},
  };
  try
  {
    for (const auto &test : tests)
    {
      deleted_points = deleted_features = deleted_voxels = 0;
      test.second();
      std::cout << "PASS: " << test.first << '\n';
    }
  }
  catch (const std::exception &error)
  {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
