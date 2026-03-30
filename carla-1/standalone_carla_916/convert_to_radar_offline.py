import os
import sys
import glob
import numpy as np
import yaml
import laspy
from tqdm import tqdm

# ============================================================================
# C-Shenron Physics Engine Path Setup
# ============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# FIXED PATH: script is in carla-suite/standalone_carla_916, so we need to go up two folders to hit the root!
SHENRON_TEAM_CODE = os.path.abspath(os.path.join(CURRENT_DIR, '../../C-Shenron/team_code'))
SHENRON_PACKAGE = os.path.join(SHENRON_TEAM_CODE, 'e2e_agent_sem_lidar2shenron_package')

sys.path.insert(0, SHENRON_TEAM_CODE)
sys.path.insert(0, SHENRON_PACKAGE)
sys.path.insert(0, os.path.dirname(SHENRON_PACKAGE))

try:
    from e2e_agent_sem_lidar2shenron_package.lidar import run_lidar
except ImportError as e:
    print(f"[FATAL] Could not connect to C-Shenron simulator: {e}")
    print(f"Make sure {SHENRON_TEAM_CODE} exists!")
    sys.exit(1)


# NOTE: No tag remapping needed — C-Shenron's lidar.py now natively handles
# CARLA 0.9.16 semantic tags (CityObjectLabel) directly.


# ============================================================================
# Main Dataset Processor
# ============================================================================
def process_dataset(dataset_dir):
    print(f"\n[INFO] Starting Radar Offline Gen for Dataset: {dataset_dir}")
    print(f"[INFO] Running on GPU (single process)")

    config_path = os.path.join(SHENRON_PACKAGE, 'simulator_configs.yaml')
    if not os.path.exists(config_path):
        print(f"[ERROR] Could not find {config_path}")
        return

    with open(config_path, 'r') as f:
        sim_config = yaml.safe_load(f)

    scenario_dir = os.path.join(dataset_dir, 'Scenario8', 'Repetition0_Town01')
    routes = sorted(glob.glob(os.path.join(scenario_dir, 'route_*')))

    if not routes:
        print(f"[WARNING] No routes found inside {scenario_dir}")
        return

    print(f"[INFO] Discovered {len(routes)} routes to process.\n")

    total_frames  = 0
    failed_frames = 0

    for route in routes:
        sem_dir       = os.path.join(route, 'semantic_lidar')
        out_front_dir = os.path.join(route, 'radar_data_front_86')
        out_rear_dir  = os.path.join(route, 'radar_data_rear_86')

        os.makedirs(out_front_dir, exist_ok=True)
        os.makedirs(out_rear_dir,  exist_ok=True)

        laz_files = sorted(glob.glob(os.path.join(sem_dir, '*.laz')))
        if not laz_files:
            continue

        already_done = sum(
            1 for f in laz_files
            if os.path.exists(os.path.join(out_front_dir,
               os.path.basename(f).replace('.laz', '.npy')))
            and os.path.exists(os.path.join(out_rear_dir,
               os.path.basename(f).replace('.laz', '.npy')))
        )
        remaining = len(laz_files) - already_done

        print(f"{os.path.basename(route)} | {len(laz_files)} frames "
              f"({already_done} done, {remaining} to process)")

        if remaining == 0:
            print("  Skipping - already complete.\n")
            continue

        route_failed = 0

        for laz_file in tqdm(laz_files, desc="  Simulating Radar"):
            frame_str      = os.path.basename(laz_file).replace('.laz', '')
            out_file_front = os.path.join(out_front_dir, f"{frame_str}.npy")
            out_file_rear  = os.path.join(out_rear_dir,  f"{frame_str}.npy")

            if os.path.exists(out_file_front) and os.path.exists(out_file_rear):
                continue

            try:
                las = laspy.read(laz_file)
                sem_lidar_frame = np.column_stack([
                    las.x, las.y, las.z,
                    las.cosine, las.index, las.sem_tag
                ]).astype(np.float32)

                # No remapping needed — C-Shenron's lidar.py handles 0.9.16 tags natively

                cfg_front = dict(sim_config)
                cfg_front['INVERT_ANGLE'] = 0
                np.save(out_file_front, run_lidar(cfg_front, sem_lidar_frame))

                cfg_rear = dict(sim_config)
                cfg_rear['INVERT_ANGLE'] = 180
                np.save(out_file_rear, run_lidar(cfg_rear, sem_lidar_frame))

                total_frames += 1

            except Exception as e:
                print(f"\n  [ERROR] Failed on {os.path.basename(laz_file)}: {e}")
                route_failed += 1
                failed_frames += 1

        if route_failed > 0:
            print(f"  [WARNING] {route_failed} frames failed in this route\n")
        else:
            print(f"  Done.\n")

    print(f"{'='*50}")
    print(f"[SUCCESS] Radar Generation Complete!")
    print(f"  Total frames processed : {total_frames}")
    print(f"  Failed frames          : {failed_frames}")
    print(f"  Dataset ready at       : {dataset_dir}")
    print(f"{'='*50}")


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/storage/dataset',
                        help='Root path to your dataset (default: /storage/dataset)')
    args = parser.parse_args()
    process_dataset(args.dataset)
