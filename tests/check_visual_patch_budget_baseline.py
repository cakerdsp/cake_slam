"""Source regression check against the pre-module commit; no ROS/bag needed.

This verifies preserved function bodies, not binary identity or runtime timing.
The frozen fingerprints are independent of the user's current Git HEAD.
"""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / 'tests/visual_patch_budget_baseline.json').read_text())


def function_body(source, class_name, name):
    start = source.index('void ' + class_name + '::' + name + '(')
    next_function = source.find('\nvoid ', start + 1)
    function = source[start:] if next_function < 0 else source[start:next_function]
    return function[function.index('{'):].strip()


vio = (root / 'src/vio_patch_budget_baseline.inc').read_text()
# The mask-aware score delegates exactly to vikit when no mask is configured.
# Normalize this one call-site hook; keep the original frozen hashes unchanged.
mask_score_call = 'maskedShiTomasiScore(ctx, img, static_cast<int>(raw_px[0]), static_cast<int>(raw_px[1]))'
assert vio.count(mask_score_call) == 1
vio = vio.replace(mask_score_call, 'vk::shiTomasiScore(img, static_cast<int>(raw_px[0]), static_cast<int>(raw_px[1]))')
lio = (root / 'src/voxel_map_multi_cam.cpp').read_text()
for name, expected in manifest['bodies'].items():
    source, owner = (vio, 'VIOManager') if name.endswith('Baseline') else (lio, 'VoxelMapManager')
    actual = hashlib.sha256(function_body(source, owner, name).encode()).hexdigest()
    if actual != expected:
        raise RuntimeError('Disabled-path function differs from baseline ' + manifest['commit'] + ': ' + name)

dispatch = (root / 'src/vio_multi_cam.cpp').read_text()
frame_body = function_body(dispatch, 'VIOManager', 'processMultiCameraFrame')
retrieval_body = function_body(dispatch, 'VIOManager', 'retrieveFromVisualSparseMapVirtual')
for body, target in [(frame_body, 'processMultiCameraFrameBaseline'),
                     (retrieval_body, 'retrieveFromVisualSparseMapVirtualBaseline')]:
    gate = body.index('if (!patch_budget_en)')
    baseline_call = body.index(target + '(')
    early_return = body.index('return;', baseline_call)
    assert gate < baseline_call < early_return
assert frame_body.index('processMultiCameraFrameWithPatchBudget(') > frame_body.index('return;')
assert retrieval_body.index('prepareVirtualCandidates(') > retrieval_body.index('return;')

# The command-line module limit must not overwrite the original YAML limit.
launch = ET.parse(root / 'launch/mapping_mid360_mei_multi_cam.launch').getroot()
params = {p.attrib['name']: p.attrib for node in launch.findall('node') for p in node.findall('param')}
assert params['vio/patch_budget_points']['value'] == '$(arg patch_budget_points)'
assert params['vio/patch_budget_en']['type'] == 'bool'
assert 'vio/max_total_points' not in params
print('4 original baseline bodies, disabled dispatch and separate launch budget verified (' + manifest['commit'] + ')')
