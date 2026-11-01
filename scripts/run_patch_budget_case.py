#!/usr/bin/env python3
"""Replay one budget ablation using a freshly compiled, isolated source copy.

All output stays in --root; the bag is opened read-only. The executable's
compile-time ROOT_DIR must be <root>/source/. See --help for the layout.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import xmlrpc.client

import rosbag
import yaml


MODES = {
    "original500": (False, "shi_tomasi"),
    "direction400": (True, "lidar_directional"),
    "isotropic400": (True, "isotropic"),
}


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def stop_owned_process(proc):
    for sig, timeout in [(signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 5)]:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
    raise RuntimeError("Unable to stop owned process " + str(proc.pid))


def prepare_parameters(source, bag_path, mode, budget_log):
    config = yaml.safe_load((source / "config/mid360_mei_multi_cam.yaml").read_text())
    camera = yaml.safe_load((source / "config/camera_mei_multi_cam.yaml").read_text())
    config.update(camera)  # Same order as mapping_mid360_mei_multi_cam.launch.
    with rosbag.Bag(str(bag_path), "r") as bag:
        topics = bag.get_type_and_topic_info().topics
        duration = bag.get_end_time() - bag.get_start_time()
    selected_topics = [config["common"]["lid_topic"], config["common"]["imu_topic"]]
    for index in range(config["common"]["num_cameras"]):
        item = config["cameras"]["camera" + str(index)]
        topic = item["img_topic"]
        if topic not in topics and topic + "/compressed" in topics:
            topic += "/compressed"
        if topic not in topics:
            raise RuntimeError("Image topic missing from bag: " + topic)
        item["img_topic"] = topic
        selected_topics.append(topic)
    missing = set(selected_topics) - set(topics)
    if missing:
        raise RuntimeError("Sensor topics missing from bag: " + str(sorted(missing)))
    enabled, score = MODES[mode]
    config["vio"].update(patch_budget_en=enabled, max_total_points=500,
                         patch_budget_points=400, patch_budget_mode="global",
                         patch_score_mode=score, patch_budget_log_en=budget_log)
    config["evo"].update(seq_name="trajectory", pose_output_en=True)
    # These settings only change this run's parameter copy, never the source YAML.
    config["pcd_save"].update(pcd_save_en=False, colmap_output_en=False, interval=-1)
    config["image_save"]["img_save_en"] = False
    config["debug"].update(memory_stats_en=False, ref_patch_dump_en=False, runtime_support_dump_en=False)
    return config, selected_topics, duration


def wait_until(check, timeout, proc, description):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(description + ": process exited with " + str(proc.returncode))
        if check():
            return
        time.sleep(0.2)
    raise RuntimeError(description + ": timeout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Fresh experiment directory containing source/, build/, devel/")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--budget-log", action="store_true",
                        help="Enable extra selector diagnostics; leave off for timing comparisons")
    args = parser.parse_args()
    root, bag_path = args.root.resolve(strict=True), args.bag.resolve(strict=True)
    if not math.isfinite(args.rate) or args.rate <= 0:
        parser.error("--rate must be finite and positive")
    source = root / "source"
    # Running from the snapshot also avoids accidentally using an older helper.
    if Path(__file__).resolve().parent.parent != source:
        parser.error("Run the script inside <root>/source/scripts/, not the original checkout")
    cache = (root / "build/CMakeCache.txt").read_text()
    if "CMAKE_HOME_DIRECTORY:INTERNAL=" + str(source) + "\n" not in cache:
        parser.error("The build was not configured from this isolated source directory")
    binary = root / "devel/lib/cake_slam/cake_slam_mapping_multi_cam"
    if not binary.is_file():
        parser.error("Build the isolated executable first: " + str(binary))
    for name in ["vio_patch_budget_baseline.inc", "voxel_patch_budget_geometry.inc"]:
        if not (source / "src" / name).is_file():
            parser.error("Incomplete source snapshot; missing " + name)
    lock = (root / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    log_dir = source / "Log"
    if os.path.lexists(log_dir):
        parser.error("Preserving existing output; inspect it before another run: " + str(log_dir))
    bag_stat = bag_path.stat()
    config, topics, duration = prepare_parameters(source, bag_path, args.mode, args.budget_log)
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=args.mode + "_", dir=runs))
    (run / "roslog").mkdir()
    (run / "roshome").mkdir()
    write_new(run / "parameters.yaml", yaml.safe_dump(config, sort_keys=False))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.pop("ROS_HOSTNAME", None)
    env.update(ROS_MASTER_URI="http://127.0.0.1:%d" % port, ROS_IP="127.0.0.1",
               ROS_LOG_DIR=str(run / "roslog"), ROS_HOME=str(run / "roshome"),
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
               LD_LIBRARY_PATH=str(root / "devel/lib") + ":" + env.get("LD_LIBRARY_PATH", ""))
    processes, handles = [], []
    result = {"mode": args.mode, "bag": str(bag_path), "rate": args.rate,
              "completed": False, "budget_log": args.budget_log, "trajectory_stage": "LIO",
              "run_directory": str(run), "source_directory": str(source), "topics": topics}
    start = time.monotonic()

    def spawn(command, filename):
        stream = (run / filename).open("x")
        handles.append(stream)
        proc = subprocess.Popen(command, cwd=run, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((filename, proc))
        return proc

    print("RUN_DIRECTORY=" + str(run), flush=True)
    try:
        master = spawn(["roscore", "-p", str(port)], "roscore.log")
        socket.setdefaulttimeout(3)
        api = xmlrpc.client.ServerProxy(env["ROS_MASTER_URI"], allow_none=True)

        def master_ready():
            try:
                return api.getPid("/patch_budget_runner")[0] == 1
            except (OSError, xmlrpc.client.Error):
                return False

        wait_until(master_ready, 30, master, "ROS master startup")
        if api.setParam("/patch_budget_runner", "/use_sim_time", True)[0] != 1:
            raise RuntimeError("Cannot enable simulation time")
        if api.setParam("/patch_budget_runner", "/laserMapping", config)[0] != 1:
            raise RuntimeError("Cannot load run parameters")
        node = spawn(["stdbuf", "-oL", "-eL", str(binary), "__name:=laserMapping"], "slam.log")

        def subscribers_ready():
            state = api.getSystemState("/patch_budget_runner")
            if state[0] != 1:
                return False
            subscribers = {topic for topic, nodes in state[2][1] if "/laserMapping" in nodes}
            return set(topics).issubset(subscribers)

        wait_until(subscribers_ready, 90, node, "SLAM sensor subscriber startup")
        player = spawn(["rosbag", "play", "--clock", "--quiet", "-r", str(args.rate),
                        "--delay", "2", str(bag_path), "--topics"] + topics, "play.log")
        deadline, next_report = time.monotonic() + duration / args.rate + 120, time.monotonic() + 20
        while player.poll() is None:
            if node.poll() is not None:
                raise RuntimeError("SLAM exited during replay; see slam.log")
            if time.monotonic() > deadline:
                raise RuntimeError("Bag replay timeout")
            if time.monotonic() >= next_report:
                print(args.mode + ": replaying, %.0f s elapsed" % (time.monotonic() - start), flush=True)
                next_report = time.monotonic() + 20
            time.sleep(0.25)
        if player.returncode != 0:
            raise RuntimeError("rosbag play failed; see play.log")

        trajectory = log_dir / "result/trajectory.txt"
        previous_size, quiet_since = -1, time.monotonic()

        def drained():
            nonlocal previous_size, quiet_since
            size = trajectory.stat().st_size if trajectory.exists() else 0
            if size != previous_size:
                previous_size, quiet_since = size, time.monotonic()
            return size > 0 and time.monotonic() - quiet_since >= 5

        wait_until(drained, 120, node, "Drain pending trajectory output")
        timestamps = []
        for line in trajectory.read_text().splitlines():
            row = [float(value) for value in line.split()]
            if len(row) != 8 or not all(math.isfinite(value) for value in row):
                raise RuntimeError("Malformed trajectory output")
            timestamps.append(row[0])
        if len(timestamps) < 10 or timestamps[-1] <= timestamps[0]:
            raise RuntimeError("Insufficient trajectory output")
        result.update(completed=True, trajectory_samples=len(timestamps),
                      first_timestamp=timestamps[0], last_timestamp=timestamps[-1])
    except (Exception, KeyboardInterrupt) as error:
        result["error"] = repr(error)
    finally:
        for _, proc in reversed(processes):
            stop_owned_process(proc)
        for stream in handles:
            stream.close()
        # Move only output created in this new source snapshot, after processes
        # have stopped; never move or rename anything in the original checkout.
        if log_dir.exists():
            if log_dir.is_symlink() or (run / "Log").exists():
                raise RuntimeError("Unexpected output path; preserving it without moving")
            log_dir.rename(run / "Log")
        after = bag_path.stat()
        result["bag_size_mtime_unchanged"] = (bag_stat.st_size, bag_stat.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        result["wall_seconds"] = time.monotonic() - start
        result["exit_codes"] = {name: proc.returncode for name, proc in processes}
        write_new(run / "run.json", json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
