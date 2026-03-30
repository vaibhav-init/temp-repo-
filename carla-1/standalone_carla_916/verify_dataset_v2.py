import os
import glob
import gzip
import json
import argparse

try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False

REQUIRED_DIRS = [
    'rgb', 'rgb_augmented', 'semantics', 'semantics_augmented',
    'depth', 'depth_augmented', 'bev_semantics',
    'bev_semantics_augmented', 'lidar', 'semantic_lidar',
    'boxes', 'measurements'
]

REQUIRED_MEASUREMENT_KEYS = [
    'pos_global', 'theta', 'speed', 'target_speed', 'target_point',
    'target_point_next', 'command', 'next_command', 'aim_wp', 'route',
    'steer', 'throttle', 'brake', 'control_brake', 'junction',
    'vehicle_hazard', 'light_hazard', 'walker_hazard',
    'stop_sign_hazard', 'stop_sign_close', 'walker_close', 'angle',
    'augmentation_translation', 'augmentation_rotation', 'ego_matrix'
]

def check_frame_alignment(route_dir, frames):
    """Ensure all required folders have exactly the same frames."""
    missing_files = []
    for d in REQUIRED_DIRS:
        dir_path = os.path.join(route_dir, d)
        if not os.path.exists(dir_path):
            missing_files.append(f"Directory missing: {d}")
            continue
        
        ext = '.jpg' if 'rgb' in d else '.png' if ('semantics' in d or 'depth' in d) else '.laz' if 'lidar' in d else '.json.gz'
        
        for f in frames:
            file_path = os.path.join(dir_path, f"{f:04d}{ext}")
            if not os.path.exists(file_path):
                missing_files.append(f"Missing: {d}/{f:04d}{ext}")
    return missing_files

def check_measurement_schema(route_dir, frames):
    """Validate the complex JSON schema inside measurements."""
    errors = []
    for f in frames:
        meas_path = os.path.join(route_dir, 'measurements', f"{f:04d}.json.gz")
        if not os.path.exists(meas_path):
            continue
        try:
            with gzip.open(meas_path, 'rt', encoding='utf-8') as fz:
                data = json.load(fz)
            for k in REQUIRED_MEASUREMENT_KEYS:
                if k not in data:
                    errors.append(f"Missing key '{k}' in {meas_path}")
            if 'ego_matrix' in data and len(data['ego_matrix']) != 4:
                errors.append(f"ego_matrix has wrong shape in {meas_path}")
            if 'route' in data and len(data['route']) == 0:
                errors.append(f"route is empty in {meas_path}")
        except Exception as e:
            errors.append(f"Corrupt JSON {meas_path}: {e}")
    return errors

def check_laz_schema(route_dir, frames):
    """Validate Semantic LiDAR has the required ExtraBytes (sem_tag)."""
    if not LASPY_AVAILABLE:
        return ["laspy not installed. Cannot verify .laz schemas."]
    
    errors = []
    if len(frames) > 0:
        # Just check the first frame to save time
        f = frames[0]
        laz_path = os.path.join(route_dir, 'semantic_lidar', f"{f:04d}.laz")
        if os.path.exists(laz_path):
            try:
                with laspy.open(laz_path) as fz:
                    header = fz.header
                    dims = list(header.point_format.extra_dimension_names)
                    if 'sem_tag' not in dims:
                        errors.append(f"Missing 'sem_tag' dimension in {laz_path}.")
                    if 'index' not in dims:
                        errors.append(f"Missing 'index' dimension in {laz_path}.")
            except Exception as e:
                errors.append(f"Corrupt LAZ {laz_path}: {e}")
    return errors

def verify_dataset(dataset_dir):
    print(f"\\n{'='*50}")
    print(f"Verifying Dataset: {dataset_dir}")
    print(f"{'='*50}\\n")
    
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Path does not exist: {dataset_dir}")
        return

    # Recursively find all 'measurements' folders
    search_pattern = os.path.join(dataset_dir, '**', 'measurements')
    measurement_dirs = glob.glob(search_pattern, recursive=True)
    
    if not measurement_dirs:
        print("[ERROR] Found ZERO 'measurements' folders. Is this a valid C-Shenron dataset?")
        return
        
    route_dirs = [os.path.dirname(md) for md in measurement_dirs]
    print(f"Found {len(route_dirs)} active routes.\\n")
    
    total_frames = 0
    passed = True
    
    for route_dir in route_dirs:
        # Check results.json.gz
        results_path = os.path.join(route_dir, 'results.json.gz')
        if not os.path.exists(results_path):
            print(f"[✘] {route_dir} -> Missing results.json.gz!")
            passed = False
        
        # Get frame numbers
        meas_dir = os.path.join(route_dir, 'measurements')
        meas_files = sorted(glob.glob(os.path.join(meas_dir, '*.json.gz')))
        frames = [int(os.path.basename(f).split('.')[0]) for f in meas_files]
        total_frames += len(frames)
        
        if len(frames) == 0:
            print(f"[!] {route_dir} -> 0 frames found. Incomplete route.")
            continue
            
        # 1. Alignment Check
        align_errors = check_frame_alignment(route_dir, frames)
        if align_errors:
            print(f"[✘] {route_dir} -> {len(align_errors)} Missing sensor files (e.g. {align_errors[0]})")
            passed = False
            
        # 2. JSON Schema Check
        json_errors = check_measurement_schema(route_dir, frames)
        if json_errors:
            print(f"[✘] {route_dir} -> {len(json_errors)} JSON Schema Errors (e.g. {json_errors[0]})")
            passed = False
            
        # 3. LAZ Schema Check
        laz_errors = check_laz_schema(route_dir, frames)
        if laz_errors:
            print(f"[✘] {route_dir} -> LAZ Format Error: {laz_errors[0]}")
            passed = False
            
    print(f"\\n{'='*50}")
    if passed:
        print(f"[✔] DATASET VERIFICATION PASSED 100%!")
        print(f"    - Routes Validated: {len(route_dirs)}")
        print(f"    - Frames Validated: {total_frames}")
        print(f"    - Ready for C-Shenron radar generation & training.")
    else:
        print(f"[✘] DATASET VERIFICATION FAILED.")
        print(f"    Review the errors above.")
    print(f"{'='*50}\\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify C-Shenron Dataset Compatibility')
    parser.add_argument('--dir', required=True, help='Path to the dataset root')
    args = parser.parse_args()
    verify_dataset(args.dir)
