"""
Shenron Model Evaluation Script for CARLA 0.9.16
=================================================
Loads a pretrained Shenron model and drives autonomously in CARLA
to evaluate model performance.

Usage:
  1. Start CARLA 0.9.16 server
  2. python evaluate_model.py --model-dir /path/to/pretrained_model_folder

The model directory should contain:
  - args.txt or config.pickle   (training config)
  - model_XXXX.pth              (model weights)

Default paths are set for the IIITD Ubuntu system.
"""

import os
import random
import sys
import time
import math
import json
import argparse
import pickle
from copy import deepcopy
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import carla

# Add the PATCHED C-Shenron team_code to path (contains converter padding,
# num_bev_semantic_classes fix, and depth slicing fixes from training).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..'))
TEAM_CODE_DIR = os.path.join(REPO_ROOT, 'C-Shenron', 'team_code')
assert os.path.isdir(TEAM_CODE_DIR), f"Patched team_code not found at {TEAM_CODE_DIR}"
sys.path.insert(0, TEAM_CODE_DIR)
sys.path.insert(0, os.path.join(TEAM_CODE_DIR, 'e2e_agent_sem_lidar2shenron_package'))

from model import LidarCenterNet
from config import GlobalConfig
from mask import generate_mask
from sim_radar_utils.convert2D_img import convert_sem_lidar_2D_img_func
import transfuser_utils as t_u
from nav_planner import RoutePlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner

# UKF for GPS filtering
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

# PyTorch optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

# Radar mask (matches C-Shenron sensor_agent.py)
mask_for_radar = generate_mask(shape=256, start_angle=35, fov_degrees=110, end_mag=0)

# ============================================================
#  Default paths for the IIITD Ubuntu system
# ============================================================
DEFAULT_MODEL_DIR = '/storage/training_logs/cshenron_town01_radar_v1'

# ============================================================
#  Weather presets
# ============================================================
WEATHER_PRESETS = {
    'ClearNoon':       carla.WeatherParameters.ClearNoon,
    'ClearSunset':     carla.WeatherParameters.ClearSunset,
    'CloudyNoon':      carla.WeatherParameters.CloudyNoon,
    'CloudySunset':    carla.WeatherParameters.CloudySunset,
    'WetNoon':         carla.WeatherParameters.WetNoon,
    'WetSunset':       carla.WeatherParameters.WetSunset,
    'WetCloudyNoon':   carla.WeatherParameters.WetCloudyNoon,
    'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,
    'SoftRainNoon':    carla.WeatherParameters.SoftRainNoon,
    'SoftRainSunset':  carla.WeatherParameters.SoftRainSunset,
    'MidRainSunset':   carla.WeatherParameters.MidRainSunset,
    'HardRainNoon':    carla.WeatherParameters.HardRainNoon,
    'HardRainSunset':  carla.WeatherParameters.HardRainSunset,
}


# ============================================================
#  Config loader — supports config.pickle, args.txt, config.json
# ============================================================
def load_config(model_dir):
    """
    Load the GlobalConfig from the model directory.
    Tries in order:
      1. config.pickle  (pickled GlobalConfig — exact match, best if available)
      2. config.json    (full config dump with all computed values — safest fallback)
      3. args.txt       (training argparse args — needs setting='eval' override)
    """
    pickle_path = os.path.join(model_dir, 'config.pickle')
    config_json_path = os.path.join(model_dir, 'config.json')
    args_txt_path = os.path.join(model_dir, 'args.txt')

    if os.path.isfile(pickle_path):
        print(f"Loading config from config.pickle ...")
        with open(pickle_path, 'rb') as f:
            config = pickle.load(f)
        return config

    if os.path.isfile(config_json_path):
        # config.json is a full dump of GlobalConfig with all computed values.
        # Safest option when config.pickle is not available.
        print(f"Loading config from config.json ...")
        config = GlobalConfig()
        with open(config_json_path, 'r') as f:
            config_dict = json.load(f)
        for k, v in config_dict.items():
            setattr(config, k, v)
        return config

    if os.path.isfile(args_txt_path):
        # args.txt is the JSON dump of training argparse args.
        # We must override 'setting' to 'eval' because the original 'all' setting
        # tries to os.listdir(root_dir) which points to non-existent training data paths.
        print(f"Loading config from args.txt ...")
        config = GlobalConfig()
        with open(args_txt_path, 'r') as f:
            args_dict = json.load(f)
        args_dict['setting'] = 'eval'  # Skip data directory listing
        config.initialize(**args_dict)
        return config

    raise FileNotFoundError(
        f"No config found in {model_dir}. "
        f"Expected one of: config.pickle, config.json, args.txt"
    )


def find_checkpoint(model_dir, checkpoint_name=None):
    """
    Find the model checkpoint (.pth) in the model directory.
    If checkpoint_name is given, use it directly.
    Otherwise, auto-detect: prefer model_XXXX_0.pth (GPU 0 weights).
    """
    if checkpoint_name:
        path = os.path.join(model_dir, checkpoint_name)
        if os.path.isfile(path):
            return path
            
        # Optional fallback for DDP (_0.pth suffix)
        if checkpoint_name.endswith('.pth') and not checkpoint_name.endswith('_0.pth'):
            fallback = checkpoint_name.replace('.pth', '_0.pth')
            path_fallback = os.path.join(model_dir, fallback)
            if os.path.isfile(path_fallback):
                return path_fallback

        # Maybe it's an absolute path
        if os.path.isfile(checkpoint_name):
            return checkpoint_name
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")

    pth_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth') and f.startswith('model_')])
    if not pth_files:
        raise FileNotFoundError(f"No model_*.pth files found in {model_dir}")

    # Prefer _0.pth (GPU 0 copy from DDP training)
    gpu0_files = [f for f in pth_files if f.endswith('_0.pth')]
    if gpu0_files:
        chosen = gpu0_files[-1]  # Latest epoch
    else:
        chosen = pth_files[-1]

    return os.path.join(model_dir, chosen)


def _normalize_state_dict(raw_checkpoint):
    """Support common checkpoint containers and DataParallel prefixes."""
    state_dict = raw_checkpoint
    if isinstance(raw_checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model', 'net'):
            value = raw_checkpoint.get(key)
            if isinstance(value, dict):
                state_dict = value
                break

    normalized = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            normalized[key[len('module.'):]] = value
        else:
            normalized[key] = value
    return normalized


# ============================================================
#  UKF Helper Functions
# ============================================================
def bicycle_model_forward(x, dt, steer, throttle, brake):
    """Simplified bicycle model for state prediction."""
    front_wheel_base = 1.0
    rear_wheel_base = 1.5
    steer_gain = 0.7
    brake_accel = -4.952399730682373
    throttle_accel = 0.5633837735652924

    vel = x[3]
    if brake:
        accel = brake_accel
    else:
        accel = throttle_accel * throttle

    wheel_heading_change = steer_gain * steer
    beta = math.atan(rear_wheel_base / (front_wheel_base + rear_wheel_base) * math.tan(wheel_heading_change))

    new_vel = vel + accel * dt
    new_vel = max(0.0, new_vel)
    new_heading = t_u.normalize_angle(x[2] + new_vel * math.sin(beta) / rear_wheel_base * dt)
    new_x = x[0] + new_vel * math.cos(new_heading) * dt
    new_y = x[1] + new_vel * math.sin(new_heading) * dt

    return np.array([new_x, new_y, new_heading, new_vel])


def bicycle_model_predict(x, dt, steer, throttle, brake):
    """Called by filterpy UKF per sigma point."""
    return bicycle_model_forward(x, dt, steer, throttle, brake)


def measurement_function(x):
    return x


# ============================================================
#  Shenron Evaluation Agent
# ============================================================
class ShenronEvalAgent:
    """
    Evaluation agent that loads a pretrained Shenron model and drives
    using Camera + Radar in CARLA 0.9.16.
    """

    def __init__(self, config, checkpoint_path, device='cuda:0', radar_cat=1):
        """
        Args:
            config:           GlobalConfig object (already loaded)
            checkpoint_path:  Path to model .pth weights file
            device:           CUDA device string
            radar_cat:        0=front only, 1=front+back, 2=all 4 directions
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.step = 0
        self.config = config
        self.config.debug = False
        self.radar_fallback_count = 0

        # Pre-create data helper
        from data import CARLA_Data
        self.data = CARLA_Data(root=[], config=self.config, shared_dict=None)

        # Load model
        print(f"Loading model checkpoint: {checkpoint_path}")
        net = LidarCenterNet(self.config)
        if self.config.sync_batch_norm:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

        raw_checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = _normalize_state_dict(raw_checkpoint)
        load_result = net.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            missing_preview = ', '.join(load_result.missing_keys[:8])
            unexpected_preview = ', '.join(load_result.unexpected_keys[:8])
            raise ValueError(
                'Checkpoint is incompatible with this evaluator model. '\
                f'Missing keys ({len(load_result.missing_keys)}): {missing_preview} | '\
                f'Unexpected keys ({len(load_result.unexpected_keys)}): {unexpected_preview}. '\
                'This usually means architecture/sensor mismatch (for example generic Garage weights vs Shenron radar model).'
            )
        net.to(self.device)
        net.eval()
        self.net = net
        print("Model loaded successfully!")

        # Match sensor_agent / map_agent: brake threshold from env (optional)
        self.config.brake_uncertainty_threshold = float(
            os.environ.get('UNCERTAINTY_THRESHOLD', str(self.config.brake_uncertainty_threshold)))

        self.config.radar_cat = int(radar_cat)

        # Match sensor_agent.py SLOWER: reduce target speeds by 2 m/s to
        # counteract classification overconfidence
        if int(os.environ.get('SLOWER', 1)):
            self.config.target_speeds[2] = self.config.target_speeds[2] - 2.0
            self.config.target_speeds[3] = self.config.target_speeds[3] - 2.0
            print(f"  [SLOWER] target_speeds adjusted to {self.config.target_speeds}")

        # C-Shenron nav_planner.RoutePlanner (set via attach_route_planner before driving)
        self._route_planner = None
        self.last_nav_command = 4

        # Initialize UKF
        points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=self.residual_state_x)
        self.ukf = UKF(dim_x=4, dim_z=4, fx=bicycle_model_predict, hx=measurement_function,
                       dt=1.0 / self.config.carla_fps, points=points, x_mean_fn=self.state_mean,
                       z_mean_fn=self.z_mean, residual_x=self.residual_state_x, residual_z=self.residual_measurement)
        self.ukf.x = np.array([0, 0, 0, 0])
        self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
        self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
        self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])
        self.filter_initialized = False

        # State tracking (matches sensor_agent.py; use data_save_freq from training config)
        self.state_log = deque(maxlen=max((self.config.lidar_seq_len * self.config.data_save_freq), 2))

        self.lidar_buffer = deque(maxlen=self.config.lidar_seq_len * self.config.data_save_freq)
        self.semantic_lidar_buffer = deque(maxlen=self.config.lidar_seq_len * self.config.data_save_freq)

        self.lidar_last = None
        self.semantic_lidar_last = None

        self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
        self.initialized = False

        # Stuck detection
        self.stuck_detector = 0
        self.force_move = 0

        # Target point tracking
        self.target_point_prev = np.array([0, 0])
        self.commands = deque(maxlen=2)
        self.commands.append(4)  # FOLLOW_LANE
        self.commands.append(4)

        # Uncertainty weight
        self.uncertainty_weight = int(os.environ.get('UNCERTAINTY_WEIGHT', 1))

    def attach_route_planner(self, route_planner):
        """Attach C-Shenron ``nav_planner.RoutePlanner`` (same as ``sensor_agent._route_planner``)."""
        self._route_planner = route_planner

    # ============================================================
    #  UKF helper methods
    # ============================================================
    @staticmethod
    def state_mean(sigmas, Wm):
        x = np.zeros(4)
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
        x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
        x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
        x[2] = math.atan2(sum_sin, sum_cos)
        x[3] = np.sum(np.dot(sigmas[:, 3], Wm))
        return x

    @staticmethod
    def z_mean(sigmas, Wm):
        x = np.zeros(4)
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
        x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
        x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
        x[2] = math.atan2(sum_sin, sum_cos)
        x[3] = np.sum(np.dot(sigmas[:, 3], Wm))
        return x

    @staticmethod
    def residual_state_x(a, b):
        y = a - b
        y[2] = t_u.normalize_angle(y[2])
        return y

    @staticmethod
    def residual_measurement(a, b):
        y = a - b
        y[2] = t_u.normalize_angle(y[2])
        return y

    # ============================================================
    #  LiDAR alignment
    # ============================================================
    def align_lidar(self, lidar, x, y, orientation, x_target, y_target, orientation_target):
        pos_diff = np.array([x_target, y_target, 0.0]) - np.array([x, y, 0.0])
        rot_diff = t_u.normalize_angle(orientation_target - orientation)
        rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
                                    [np.sin(orientation_target),  np.cos(orientation_target), 0.0],
                                    [0.0, 0.0, 1.0]])
        pos_diff = rotation_matrix.T @ pos_diff
        return t_u.algin_lidar(lidar, pos_diff, rot_diff)

    def align_semantic_lidar(self, semantic_lidar, x, y, orientation, x_target, y_target, orientation_target):
        pos_diff = np.array([x_target, y_target, 0.0]) - np.array([x, y, 0.0])
        rot_diff = t_u.normalize_angle(orientation_target - orientation)
        rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
                                    [np.sin(orientation_target),  np.cos(orientation_target), 0.0],
                                    [0.0, 0.0, 1.0]])
        pos_diff = rotation_matrix.T @ pos_diff
        return t_u.algin_semantic_lidar(semantic_lidar, pos_diff, rot_diff)

    # ============================================================
    #  Main inference step
    # ============================================================
    @torch.inference_mode()
    def run_step(self, rgb_image, lidar_data, semantic_lidar_data, gps_carla_from_gnss, speed, compass):
        """
        Main inference step (aligned with ``sensor_agent.run_step`` / ``tick``).

        Args:
            rgb_image: (H, W, 3) BGR camera image
            lidar_data: (N, 4) lidar points [x, y, z, intensity]
            semantic_lidar_data: (N, 6) semantic lidar [x, y, z, cosine, index, tag]
            gps_carla_from_gnss: (2,) position from ``RoutePlanner.convert_gps_to_carla(lat, lon)``
            speed: float, vehicle speed m/s
            compass: IMU compass (rad); preprocess with ``preprocess_compass`` before passing
        """
        self.step += 1
        if self.step % 10 == 0:
            print(f"\n--- Step {self.step} | speed={speed:.2f} m/s | lidar_buf={len(self.lidar_buffer)} ---")

        if self._route_planner is None:
            raise RuntimeError("Call agent.attach_route_planner(RoutePlanner) before run_step.")

        # ---- Process RGB (matches sensor_agent.tick) ----
        _, compressed = cv2.imencode('.jpg', rgb_image)
        camera = cv2.imdecode(compressed, cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = torch.from_numpy(rgb).to(self.device, dtype=torch.float32).unsqueeze(0)

        # ---- LiDAR / semantic LiDAR → ego ----
        lidar_data = t_u.lidar_to_ego_coordinate(self.config, [None, lidar_data])
        semantic_lidar_data = t_u.semantic_lidar_to_ego_coordinate(self.config, [None, semantic_lidar_data])

        # ---- UKF (measurements: GNSS→CARLA xy + preprocessed compass) ----
        compass = t_u.normalize_angle(compass)
        if not self.filter_initialized:
            self.ukf.x = np.array([gps_carla_from_gnss[0], gps_carla_from_gnss[1], compass, speed])
            self.filter_initialized = True

        self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
        self.ukf.update(np.array([gps_carla_from_gnss[0], gps_carla_from_gnss[1], compass, speed]))
        filtered_state = self.ukf.x
        self.state_log.append(filtered_state)

        # ---- Route + command (matches sensor_agent.tick) ----
        waypoint_route = self._route_planner.run_step(np.array([filtered_state[0], filtered_state[1]]))
        if len(waypoint_route) > 2:
            target_point, far_command = waypoint_route[1]
        elif len(waypoint_route) > 1:
            target_point, far_command = waypoint_route[1]
        else:
            target_point, far_command = waypoint_route[0]

        if (target_point != self.target_point_prev).all():
            self.target_point_prev = target_point.copy()
            self.commands.append(far_command.value)

        self.last_nav_command = self.commands[-2]
        one_hot_command = t_u.command_to_one_hot(self.last_nav_command)
        command = torch.from_numpy(one_hot_command[np.newaxis]).to(self.device, dtype=torch.float32)

        ego_target = t_u.inverse_conversion_2d(target_point, filtered_state[0:2], filtered_state[2])
        ego_target = torch.from_numpy(ego_target[np.newaxis]).to(self.device, dtype=torch.float32)

        gt_velocity = torch.FloatTensor([speed]).to(self.device, dtype=torch.float32)
        velocity = gt_velocity.reshape(1, 1)

        # ---- First frame (matches sensor_agent) ----
        if not self.initialized:
            self.lidar_last = deepcopy(lidar_data)
            self.semantic_lidar_last = deepcopy(semantic_lidar_data)
            self.initialized = True
            self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            return self.control

        ego_x, ego_y, ego_theta = filtered_state[0], filtered_state[1], filtered_state[2]
        ego_x_last = self.state_log[-2][0]
        ego_y_last = self.state_log[-2][1]
        ego_theta_last = self.state_log[-2][2]

        lidar_last_aligned = self.align_lidar(
            self.lidar_last, ego_x_last, ego_y_last, ego_theta_last, ego_x, ego_y, ego_theta)
        sem_lidar_last_aligned = self.align_semantic_lidar(
            self.semantic_lidar_last, ego_x_last, ego_y_last, ego_theta_last, ego_x, ego_y, ego_theta)

        lidar_full = np.concatenate((lidar_data, lidar_last_aligned), axis=0)
        sem_lidar_full = np.concatenate((semantic_lidar_data, sem_lidar_last_aligned), axis=0)

        self.lidar_buffer.append(lidar_full)
        self.semantic_lidar_buffer.append(sem_lidar_full)

        needed = self.config.lidar_seq_len * self.config.data_save_freq
        if len(self.lidar_buffer) < needed:
            print(f"  [WAIT] Filling lidar buffer: {len(self.lidar_buffer)}/{needed}")
            self.lidar_last = deepcopy(lidar_data)
            self.semantic_lidar_last = deepcopy(semantic_lidar_data)
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
            return self.control

        if self.step % self.config.action_repeat == 1:
            self.lidar_last = deepcopy(lidar_data)
            self.semantic_lidar_last = deepcopy(semantic_lidar_data)
            return self.control

        lidar_indices = [i * self.config.data_save_freq for i in range(self.config.lidar_seq_len)]
        radar_list = []
        lidar_bev_list = []

        for i in lidar_indices:
            lidar_point_cloud = deepcopy(self.lidar_buffer[i])

            if self.config.realign_lidar and self.config.lidar_seq_len > 1:
                curr_x = self.state_log[i][0]
                curr_y = self.state_log[i][1]
                curr_theta = self.state_log[i][2]
                lidar_point_cloud = self.align_lidar(
                    lidar_point_cloud, curr_x, curr_y, curr_theta, ego_x, ego_y, ego_theta)

            lidar_histogram = torch.from_numpy(
                self.data.lidar_to_histogram_features(
                    lidar_point_cloud, use_ground_plane=self.config.use_ground_plane)
            ).unsqueeze(0).to(self.device, dtype=torch.float32)
            lidar_bev_list.append(lidar_histogram)

            raw_radar = deepcopy(self.semantic_lidar_buffer[i])
            try:
                radar_np = convert_sem_lidar_2D_img_func(raw_radar, 0)

                radar_channel = int(os.environ.get('RADAR_CHANNEL', '1'))
                if radar_channel == 2:
                    radar_np_back = convert_sem_lidar_2D_img_func(raw_radar, 180)
                    radar_np = np.stack((radar_np, radar_np_back), axis=0)

                if self.config.radar_cat == 1:
                    radar_np_back = convert_sem_lidar_2D_img_func(raw_radar, 180)
                    radar_np_back = np.rot90(np.rot90(radar_np_back))
                    radar_cat = np.concatenate((radar_np, radar_np_back), axis=0)  # (512, 256)
                    center_x, center_y = radar_cat.shape[1] // 2, radar_cat.shape[0] // 2
                    crop_size = 256
                    radar_np = radar_cat[center_y - crop_size // 2:center_y + crop_size // 2,
                                         center_x - crop_size // 2:center_x + crop_size // 2]

                if self.config.radar_cat == 2:
                    radar_np_back = convert_sem_lidar_2D_img_func(raw_radar, 180)
                    radar_np_left = convert_sem_lidar_2D_img_func(raw_radar, 270)
                    radar_np_right = convert_sem_lidar_2D_img_func(raw_radar, 90)
                    radar_np = radar_np * mask_for_radar
                    radar_np_back = radar_np_back * mask_for_radar
                    radar_np_left = radar_np_left * mask_for_radar
                    radar_np_right = radar_np_right * mask_for_radar
                    radar_np_left = np.rot90(radar_np_left)
                    radar_np_back = np.rot90(np.rot90(radar_np_back))
                    radar_np_right = np.rot90(np.rot90(np.rot90(radar_np_right)))
                    radar_np = radar_np + radar_np_back + radar_np_left + radar_np_right

                if int(os.environ.get('DB_ON', 0)):
                    radar_np = np.log(radar_np + 1e-10)
                if int(os.environ.get('BLACKOUT_RADAR', 0)):
                    radar_np = np.zeros((256, 256))

                if radar_channel <= 1:
                    radar_np_exp = np.expand_dims(radar_np, axis=2)
                    radar_np_exp = np.transpose(radar_np_exp, (2, 0, 1))
                else:
                    radar_np_exp = radar_np
            except torch.OutOfMemoryError:
                self.radar_fallback_count += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if self.radar_fallback_count == 1 or self.radar_fallback_count % 20 == 0:
                    print(f"  [WARN] Radar synthesis OOM (count={self.radar_fallback_count}). Using zero radar fallback.")
                z = np.zeros((256, 256), dtype=np.float32)
                radar_np_exp = np.expand_dims(z, axis=2)
                radar_np_exp = np.transpose(radar_np_exp, (2, 0, 1))
            except RuntimeError as exc:
                if 'out of memory' in str(exc).lower():
                    self.radar_fallback_count += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if self.radar_fallback_count == 1 or self.radar_fallback_count % 20 == 0:
                        print(f"  [WARN] Radar synthesis OOM (count={self.radar_fallback_count}). Using zero radar fallback.")
                    z = np.zeros((256, 256), dtype=np.float32)
                    radar_np_exp = np.expand_dims(z, axis=2)
                    radar_np_exp = np.transpose(radar_np_exp, (2, 0, 1))
                else:
                    raise

            radar = torch.from_numpy(radar_np_exp).to(self.device, dtype=torch.float32).unsqueeze(0)
            radar_list.append(radar)

        radar_tensor = torch.cat(radar_list, dim=1)
        lidar_bev = torch.cat(lidar_bev_list, dim=1)

        # transFuser_cr uses radar instead of lidar_bev (matches sensor_agent.py)
        if self.config.backbone == 'transFuser_cr':
            forward_lidar_bev = None
        else:
            forward_lidar_bev = lidar_bev

        pred_wp, pred_target_speed, pred_checkpoint, \
        pred_semantic, pred_bev_semantic, pred_depth, \
        pred_bb_features, attention_weights, pred_wp_1, \
        selected_path = self.net.forward(
            rgb=rgb,
            lidar_bev=forward_lidar_bev,
            radar=radar_tensor,
            target_point=ego_target,
            ego_vel=velocity,
            command=command)

        pred_angle = 0.0
        target_speed = 0.0

        if self.config.use_wp_gru:
            if self.config.multi_wp_output:
                if F.sigmoid(selected_path)[0].item() > 0.5:
                    pred_wp = pred_wp_1
            self.pred_wp = pred_wp
            self.predicted_waypoints_ego = pred_wp[0, :, :2].cpu().numpy()

        if self.config.use_controller_input_prediction:
            pred_target_speed_probs = F.softmax(pred_target_speed[0], dim=0)
            pred_aim_wp = pred_checkpoint[0][1].detach().cpu().numpy()
            pred_angle = -math.degrees(math.atan2(-pred_aim_wp[1], pred_aim_wp[0])) / 90.0

            if self.uncertainty_weight:
                uncertainty = pred_target_speed_probs.detach().cpu().numpy()
                if self.step % 20 == 0:
                    print(f"  [DEBUG] uncertainty={uncertainty}")
                    print(f"  [DEBUG] target_speeds={self.config.target_speeds}")
                    print(f"  [DEBUG] ego_target={ego_target.cpu().numpy()[0]}, pred_aim_wp={pred_aim_wp}, pred_angle={pred_angle:.3f}")
                if uncertainty[0] > self.config.brake_uncertainty_threshold:
                    target_speed = self.config.target_speeds[0]
                else:
                    target_speed = float(np.sum(uncertainty * np.array(self.config.target_speeds)))
            else:
                idx = int(torch.argmax(pred_target_speed_probs).item())
                target_speed = self.config.target_speeds[idx]

        if self.config.inference_direct_controller and self.config.use_controller_input_prediction:
            steer, throttle, brake = self.net.control_pid_direct(target_speed, pred_angle, gt_velocity)
        elif self.config.use_wp_gru and not self.config.inference_direct_controller:
            steer, throttle, brake = self.net.control_pid(self.pred_wp, gt_velocity)
        else:
            raise ValueError(
                'Control path mismatch (see sensor_agent.py): use '
                '(inference_direct_controller + use_controller_input_prediction) or '
                '(use_wp_gru and not inference_direct_controller).')

        # Stuck + creep (matches sensor_agent.py)
        if gt_velocity.item() < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        if self.stuck_detector > self.config.stuck_threshold:
            self.force_move = int(self.config.creep_duration)

        if self.force_move > 0:
            emergency_stop = False
            if self.config.backbone != 'aim':
                safety_box = deepcopy(self.lidar_buffer[-1])
                safety_box = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
                safety_box = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]
                safety_box = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
                safety_box = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]
                safety_box = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
                safety_box = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]
                emergency_stop = len(safety_box) > 0

            if not emergency_stop:
                print(f'  [STUCK] Detected agent being stuck. Step: {self.step}')
                throttle = max(self.config.creep_throttle, throttle)
                brake = False
                self.force_move -= 1
                self.stuck_detector = 0
            else:
                print('  [STUCK] Creeping stopped by safety box.')
                throttle = 0.0
                brake = True
                self.force_move = int(self.config.creep_duration)

        self.lidar_last = deepcopy(lidar_data)
        self.semantic_lidar_last = deepcopy(semantic_lidar_data)

        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))

        if self.step < self.config.inital_frames_delay:
            control = carla.VehicleControl(0.0, 0.0, 1.0)

        self.control = control
        return control


# ============================================================
#  CARLA Helper Functions
# ============================================================
def setup_carla(host='localhost', port=2000, town='Town01', weather='ClearNoon', fog_density=None):
    """Connect to CARLA, load the world, and set weather.

    If fog_density is set (e.g. 40.0), applies the same fog overlay as optional training-style weather.
    Default None leaves preset weather unchanged (neutral eval).
    """
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world = client.load_world(town)

    # Set weather
    if weather in WEATHER_PRESETS:
        world.set_weather(WEATHER_PRESETS[weather])
        print(f"Weather set to: {weather}")
    else:
        print(f"WARNING: Unknown weather '{weather}', available: {list(WEATHER_PRESETS.keys())}")
        print("Falling back to ClearNoon")
        world.set_weather(carla.WeatherParameters.ClearNoon)

    if fog_density is not None and fog_density > 0:
        current_weather = world.get_weather()
        current_weather.fog_density = float(fog_density)
        current_weather.fog_distance = 50.0
        current_weather.fog_falloff = 1.0
        world.set_weather(current_weather)
        print(f"Fog overlay applied: density={fog_density}%, distance=50m")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)

    # Shorten traffic light durations for faster testing
    world.tick()
    tl_actors = world.get_actors().filter('traffic.traffic_light')
    for tl in tl_actors:
        tl.set_green_time(15.0)
        tl.set_red_time(5.0)
        tl.set_yellow_time(2.0)
    print(f'  Traffic lights: {len(tl_actors)} found, red=5s, yellow=2s, green=15s')

    traffic_manager = client.get_trafficmanager(8100)
    traffic_manager.set_synchronous_mode(True)

    return client, world, traffic_manager


def spawn_vehicle(world, spawn_index=None):
    """Spawn the ego vehicle just before an intersection."""
    import random
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle.lincoln.mkz_2020')[0]

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    chosen_spawn = None

    if spawn_index is not None:
        chosen_spawn = spawn_points[spawn_index % len(spawn_points)]
    else:
        # Search for a spawn point that leads to an intersection soon
        random.shuffle(spawn_points)
        for sp in spawn_points:
            wp = carla_map.get_waypoint(sp.location)
            
            # Look ahead ~30 meters to see if there's a junction
            found_junction = False
            for dist in range(10, 50, 5):
                next_wps = wp.next(dist)
                if next_wps and next_wps[0].is_junction:
                    found_junction = True
                    break
            
            if found_junction:
                chosen_spawn = sp
                break
        
        # Fallback if none found
        if chosen_spawn is None:
            chosen_spawn = spawn_points[0]

    # Snap to exact lane center using the waypoint system
    center_wp = carla_map.get_waypoint(chosen_spawn.location, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
    if center_wp is not None:
        spawn_transform = center_wp.transform
        spawn_transform.location.z += 0.5  # Lift slightly to avoid clipping ground
    else:
        spawn_transform = chosen_spawn
        spawn_transform.location.z += 0.5

    vehicle = world.spawn_actor(vehicle_bp, spawn_transform)
    print(f"Spawned vehicle at lane center: {spawn_transform.location}")
    return vehicle


def attach_sensors(world, vehicle, config, sem_lidar_pps=None):
    """Attach camera, lidar, semantic lidar, IMU, GNSS sensors."""
    bp_lib = world.get_blueprint_library()
    sensors = {}

    # RGB Camera
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(config.camera_width))
    cam_bp.set_attribute('image_size_y', str(config.camera_height))
    cam_bp.set_attribute('fov', str(config.camera_fov))
    cam_transform = carla.Transform(
        carla.Location(x=config.camera_pos[0], y=config.camera_pos[1], z=config.camera_pos[2]),
        carla.Rotation(roll=config.camera_rot_0[0], pitch=config.camera_rot_0[1], yaw=config.camera_rot_0[2])
    )
    sensors['camera'] = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

    # Regular LiDAR
    lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('rotation_frequency', str(config.lidar_rotation_frequency))
    lidar_bp.set_attribute('points_per_second', str(config.lidar_points_per_second))
    lidar_transform = carla.Transform(
        carla.Location(x=config.lidar_pos[0], y=config.lidar_pos[1], z=config.lidar_pos[2]),
        carla.Rotation(roll=config.lidar_rot[0], pitch=config.lidar_rot[1], yaw=config.lidar_rot[2])
    )
    sensors['lidar'] = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    # Semantic LiDAR (for radar simulation)
    sem_lidar_bp = bp_lib.find('sensor.lidar.ray_cast_semantic')
    sem_lidar_bp.set_attribute('rotation_frequency', str(config.lidar_rotation_frequency))
    sem_points_per_second = int(sem_lidar_pps) if sem_lidar_pps is not None else int(min(config.lidar_points_per_second, 120000))
    sem_lidar_bp.set_attribute('points_per_second', str(sem_points_per_second))
    print(f"Semantic LiDAR points_per_second: {sem_points_per_second}")
    sensors['semantic_lidar'] = world.spawn_actor(sem_lidar_bp, lidar_transform, attach_to=vehicle)

    # IMU
    imu_bp = bp_lib.find('sensor.other.imu')
    sensors['imu'] = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)

    # GNSS
    gnss_bp = bp_lib.find('sensor.other.gnss')
    sensors['gnss'] = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=vehicle)

    return sensors


class SensorData:
    """Collects sensor data callbacks."""
    def __init__(self):
        self.rgb = None
        self.lidar = None
        self.semantic_lidar = None
        self.imu = None
        self.gnss = None

    def on_rgb(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        self.rgb = array

    def on_lidar(self, data):
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        self.lidar = points

    def on_semantic_lidar(self, data):
        points = np.frombuffer(data.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('cos', np.float32), ('idx', np.uint32), ('tag', np.uint32)
        ]))
        result = np.column_stack([
            points['x'], points['y'], points['z'],
            points['cos'],
            points['idx'].astype(np.float64),
            points['tag'].astype(np.float64)
        ])
        self.semantic_lidar = result

    def on_imu(self, data):
        self.imu = data

    def on_gnss(self, data):
        self.gnss = data


def spawn_traffic(client, world, tm, num_vehicles=30, num_walkers=20):
    """Spawn NPC vehicles and pedestrians for realistic testing."""
    import random
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_actors = []
    walker_actors = []

    # --- Spawn Vehicles ---
    vehicle_bps = bp_lib.filter('vehicle.*')
    vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute('number_of_wheels')) >= 4]

    random.shuffle(spawn_points)
    num_vehicles = min(num_vehicles, len(spawn_points) - 1)

    batch = []
    for i in range(num_vehicles):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        bp.set_attribute('role_name', 'autopilot')
        batch.append(carla.command.SpawnActor(bp, spawn_points[i + 1]).then(
            carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())))

    results = client.apply_batch_sync(batch, True)
    for result in results:
        if not result.error:
            vehicle_actors.append(result.actor_id)

    print(f"Spawned {len(vehicle_actors)} NPC vehicles")

    # --- Spawn Walkers ---
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    walker_controller_bp = bp_lib.find('controller.ai.walker')

    walker_ids = []
    controller_ids = []

    for _ in range(num_walkers):
        spawn_loc = world.get_random_location_from_navigation()
        if spawn_loc is None:
            continue
        bp = random.choice(walker_bps)
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        try:
            walker = world.spawn_actor(bp, carla.Transform(spawn_loc))
            walker_ids.append(walker.id)
            controller = world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=walker)
            controller_ids.append(controller.id)
        except:
            pass

    world.tick()
    all_controllers = world.get_actors(controller_ids)
    for controller in all_controllers:
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(1.0 + random.random() * 1.5)

    print(f"Spawned {len(walker_ids)} pedestrians")

    walker_actors = walker_ids + controller_ids
    return vehicle_actors, walker_actors


# ============================================================
#  Main Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Shenron Model Evaluation — test a pretrained model in CARLA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (uses /storage/vaibhavDownloads/pretrained_models/all_towns)
  python evaluate_model.py

  # Specify a different model directory
  python evaluate_model.py --model-dir /path/to/model_folder

  # Pick a specific checkpoint
  python evaluate_model.py --checkpoint model_0030_1.pth

  # Different weather or town
  python evaluate_model.py --weather CloudyNoon --town Town03
        """
    )
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR,
                        help=f'Path to model directory containing args.txt/config.pickle + .pth files '
                             f'(default: {DEFAULT_MODEL_DIR})')
    parser.add_argument('--checkpoint', default='model_0019.pth',
                        help='Specific .pth filename to use (default: model_0019.pth)')
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--town', default='Town01', help='CARLA map to load')
    parser.add_argument('--weather', default='ClearNoon',
                        help=f'Weather preset. Options: {", ".join(WEATHER_PRESETS.keys())}')
    parser.add_argument('--radar-cat', type=int, default=1, help='0=front, 1=front+back, 2=all 4')
    parser.add_argument('--duration', type=int, default=600, help='Duration in seconds')
    parser.add_argument('--vehicles', type=int, default=0, help='Number of NPC vehicles (0 to disable)')
    parser.add_argument('--walkers', type=int, default=0, help='Number of pedestrians (0 to disable)')
    parser.add_argument('--spawn-index', type=int, default=None, help='Specific spawn point index')
    parser.add_argument('--sem-lidar-pps', type=int, default=20000,
                        help='Semantic LiDAR points per second used for radar synthesis (lower reduces GPU memory)')
    parser.add_argument('--fog-density', type=float, default=None,
                        help='If set (e.g. 40), apply fog overlay on top of weather preset; default=no extra fog')
    parser.add_argument('--force-green-lights', action='store_true',
                        help='Force all traffic lights green (not training-neutral)')
    parser.add_argument('--uncertainty-threshold', type=float, default=None,
                        help='Override brake uncertainty threshold (same as UNCERTAINTY_THRESHOLD env in sensor_agent)')
    args = parser.parse_args()

    # Validate model directory
    if not os.path.isdir(args.model_dir):
        print(f"ERROR: Model directory not found: {args.model_dir}")
        sys.exit(1)

    # Load config
    config = load_config(args.model_dir)

    # Must set inference_direct_controller=True for models trained with
    # use_controller_input_prediction=1 and use_wp_gru=0, otherwise the
    # control path selection hits a ValueError.
    config.inference_direct_controller = True

    # Find checkpoint
    checkpoint_path = find_checkpoint(args.model_dir, args.checkpoint)
    print(f"Using checkpoint: {checkpoint_path}")

    os.environ['RADAR_CAT'] = str(args.radar_cat)
    os.environ['RADAR_CHANNEL'] = '1'
    os.environ['UNCERTAINTY_WEIGHT'] = '1'
    if args.uncertainty_threshold is not None:
        os.environ['UNCERTAINTY_THRESHOLD'] = str(args.uncertainty_threshold)

    actors = []
    npc_vehicle_ids = []
    walker_ids = []

    try:
        # Print banner
        print()
        print("=" * 60)
        print("  Shenron Model Evaluation")
        print("=" * 60)
        print(f"  Model Dir:  {args.model_dir}")
        print(f"  Checkpoint: {os.path.basename(checkpoint_path)}")
        print(f"  Town:       {args.town}")
        print(f"  Weather:    {args.weather}")
        print(f"  Duration:   {args.duration}s")
        print(f"  Traffic:    {args.vehicles} vehicles, {args.walkers} walkers")
        print("=" * 60)
        print()

        # 1. Connect to CARLA
        print("Connecting to CARLA...")
        client, world, tm = setup_carla(
            args.host, args.port, args.town, args.weather, fog_density=args.fog_density)
        print(f"Connected! Town: {args.town}")

        if args.force_green_lights:
            print("Forcing all traffic lights to Green (--force-green-lights)...")
            traffic_lights = world.get_actors().filter('*traffic_light*')
            for tl in traffic_lights:
                tl.set_state(carla.TrafficLightState.Green)
                tl.set_green_time(99999.0)

        agent = ShenronEvalAgent(config, checkpoint_path, radar_cat=args.radar_cat)

        # 3. Spawn vehicle
        vehicle = spawn_vehicle(world, spawn_index=args.spawn_index)
        actors.append(vehicle)

        # 4. Attach sensors
        sensor_data = SensorData()
        sensors = attach_sensors(world, vehicle, agent.config, sem_lidar_pps=args.sem_lidar_pps)
        sensors['camera'].listen(sensor_data.on_rgb)
        sensors['lidar'].listen(sensor_data.on_lidar)
        sensors['semantic_lidar'].listen(sensor_data.on_semantic_lidar)
        sensors['imu'].listen(sensor_data.on_imu)
        sensors['gnss'].listen(sensor_data.on_gnss)
        actors.extend(sensors.values())

        # 5. Set spectator
        spectator = world.get_spectator()

        # Wait for sensors to initialize
        for _ in range(10):
            world.tick()

        # Spawn NPC traffic
        if args.vehicles > 0 or args.walkers > 0:
            npc_vehicle_ids, walker_ids = spawn_traffic(client, world, tm, args.vehicles, args.walkers)

        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()
        grp = GlobalRoutePlanner(carla_map, 2.0)

        dest = random.choice(spawn_points).location
        init_trace = grp.trace_route(vehicle.get_transform().location, dest)
        if len(init_trace) < 4:
            init_trace = grp.trace_route(spawn_points[0].location, random.choice(spawn_points).location)
        route_planner = RoutePlanner(
            agent.config.route_planner_min_distance, agent.config.route_planner_max_distance)
        route_planner.set_route([(wp.transform, cmd) for wp, cmd in init_trace], False)
        agent.attach_route_planner(route_planner)

        print()
        print("=" * 60)
        print("  Agent is driving! Press Ctrl+C to stop.")
        print("=" * 60)
        print()

        # 8. Main loop
        start_time = time.time()
        fps_counter = 0
        fps_timer = time.time()

        while time.time() - start_time < args.duration:
            world.tick()

            if sensor_data.rgb is None or sensor_data.lidar is None or sensor_data.semantic_lidar is None or sensor_data.imu is None or sensor_data.gnss is None:
                continue

            # Get vehicle state
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

            # Use world coordinates directly (not GNSS) since route waypoints
            # are in raw CARLA world XY.  GNSS→convert_gps_to_carla produces a
            # different reference frame, causing ~655 m offset in ego target.
            gps_carla = np.array([transform.location.x, transform.location.y], dtype=np.float64)
            compass = t_u.preprocess_compass(sensor_data.imu.compass)

            if len(agent._route_planner.route) < 15:
                dest = random.choice(spawn_points).location
                ext_trace = grp.trace_route(transform.location, dest)
                if len(ext_trace) > 3:
                    agent._route_planner.set_route([(wp.transform, cmd) for wp, cmd in ext_trace], False)

            route_list = list(agent._route_planner.route)
            for i in range(min(15, len(route_list))):
                pos, _ = route_list[i]
                pt = carla.Location(float(pos[0]), float(pos[1]), transform.location.z + 2.0)
                world.debug.draw_point(pt, size=0.25, color=carla.Color(0, 255, 0), life_time=0.1)

            control = agent.run_step(
                rgb_image=sensor_data.rgb,
                lidar_data=sensor_data.lidar,
                semantic_lidar_data=sensor_data.semantic_lidar,
                gps_carla_from_gnss=gps_carla,
                speed=speed,
                compass=compass,
            )
            
            # --- Live Render Neural Network Predicted Path (Red) ---
            if hasattr(agent, 'predicted_waypoints_ego'):
                # CARLA yaw is in degrees, positive right-hand rule but left-handed coords
                yaw = math.radians(transform.rotation.yaw)
                cos_y = math.cos(yaw)
                sin_y = math.sin(yaw)
                
                for idx, wp in enumerate(agent.predicted_waypoints_ego):
                    # C-Shenron prediction format: wp[0] is Local Right, wp[1] is Local Forward.
                    local_right = wp[0]
                    local_forward = wp[1]
                    
                    # Transform to global CARLA coordinates
                    gx = transform.location.x + (local_forward * cos_y) - (local_right * sin_y)
                    gy = transform.location.y + (local_forward * sin_y) + (local_right * cos_y)
                    gz = transform.location.z + 2.5
                        
                    world.debug.draw_point(carla.Location(gx, gy, gz), size=0.4, color=carla.Color(255, 0, 0), life_time=0.2)

                # Periodically Print Raw Waypoint Mathematics for the User
                if fps_counter == 2:
                    carla_local_wps = []
                    _rl = list(agent._route_planner.route)
                    for i in range(min(5, len(_rl))):
                        pos = _rl[i][0]
                        dx = float(pos[0]) - transform.location.x
                        dy = float(pos[1]) - transform.location.y
                        lfwd = dx * cos_y + dy * sin_y
                        lrgt = -dx * sin_y + dy * cos_y
                        carla_local_wps.append((lfwd, lrgt))
                    
                    model_local_wps = [(wp[1], wp[0]) for wp in agent.predicted_waypoints_ego[:5]]
                    
                    print("\n  ==== [WAYPOINT COMPARISON] ====")
                    print("  Format: (Forward meters, Right meters)")
                    print(f"  CARLA (Green):  {' | '.join([f'({f:.1f}, {r:.1f})' for f, r in carla_local_wps])}")
                    print(f"  MODEL (Red):    {' | '.join([f'({f:.1f}, {r:.1f})' for f, r in model_local_wps])}")
                    print("  ===============================")

            # Note: TransFuser model outputs correct steer sign for CARLA directly.
            # (The radar-only Shenron variant needed negation, but this model does not.)

            vehicle.apply_control(control)

            # Update spectator (third-person view)
            loc = transform.transform(carla.Location(x=-6.0, z=3.0))
            spectator.set_transform(carla.Transform(loc, carla.Rotation(pitch=-15.0, yaw=transform.rotation.yaw)))

            # Telemetry
            fps_counter += 1
            if time.time() - fps_timer > 2.0:
                fps = fps_counter / (time.time() - fps_timer)
                elapsed = time.time() - start_time
                remaining = args.duration - elapsed
                
                # --- Advanced Data Collector Telemetry Style ---
                import math as _m
                _yaw_deg = transform.rotation.yaw
                _theta_deg = np.rad2deg(compass) # compass is theta in rads
                # Compass now includes -90 deg offset, so we offset back before verifying against yaw
                _theta_diff = abs(_theta_deg + 90.0 - _yaw_deg) % 360
                if _theta_diff > 180:
                    _theta_diff = 360 - _theta_diff
                _theta_ok = "OK" if _theta_diff < 1.0 else f"OFFSET={_theta_diff:.0f}deg"

                # Semantic tag distribution from the LiDAR
                _sem_data = sensor_data.semantic_lidar
                if _sem_data is not None and len(_sem_data) > 0:
                    _tags, _cnts = np.unique(_sem_data[:, 5], return_counts=True)
                    _top5 = sorted(zip(_tags, _cnts), key=lambda x: -x[1])[:5]
                    _tag_names = {0:'NONE', 1:'Roads', 2:'Sidewalks', 3:'Buildings',
                                  4:'Walls', 5:'Fences', 6:'Poles', 7:'TrafLight',
                                  8:'TrafSign', 9:'Vegetation', 10:'Terrain',
                                  11:'Sky', 12:'Pedestrian', 13:'Rider', 14:'Car',
                                  15:'Truck', 16:'Bus', 17:'Train', 18:'Motorcycle',
                                  19:'Bicycle', 20:'Static', 21:'Dynamic',
                                  23:'Water', 24:'RoadLines', 25:'Ground',
                                  26:'Bridge', 27:'RailTrack', 28:'GuardRail'}
                    _sem_str = ' '.join(f"{_tag_names.get(int(t),f'?{int(t)}')}:{c}" for t,c in _top5)
                else:
                    _sem_str = "None"
                    
                _cmd_names = {1:'LEFT', 2:'RIGHT', 3:'STRAIGHT', 4:'FOLLOW'}
                _nc = agent.last_nav_command
                _cmd_str = f"{_nc}({_cmd_names.get(_nc,'?')})"
                
                print(f"\n[{elapsed:.0f}s/{args.duration}s] Speed: {speed:.1f}m/s | "
                      f"Steer: {control.steer:+.2f} | Throttle: {control.throttle:.2f} | "
                      f"Brake: {control.brake:.2f} | FPS: {fps:.1f} | Rem: {remaining:.0f}s")
                print(f"      YAW={_yaw_deg:+.1f}° | THETA={_theta_deg:+.1f}° | theta_check={_theta_ok}")
                
                _rp = agent._route_planner.route
                if len(_rp) > 2:
                    _tp = np.asarray(_rp[1][0], dtype=np.float64)
                elif len(_rp) > 0:
                    _tp = np.asarray(_rp[0][0], dtype=np.float64)
                else:
                    _tp = agent.ukf.x[:2].copy()
                _ego_target = t_u.inverse_conversion_2d(_tp, agent.ukf.x[:2], agent.ukf.x[2])
                print(f"      target_ego=[{_ego_target[0]:+.1f}, {_ego_target[1]:+.1f}] "
                      f"({'AHEAD' if _ego_target[0] > 0 else 'BEHIND!!'})")
                print(f"      sem_tags: {_sem_str}")
                print(f"      cmd={_cmd_str}")

                fps_counter = 0
                fps_timer = time.time()

        print("\n[DONE] Evaluation duration reached!")

    except KeyboardInterrupt:
        print("\n[STOPPED] Agent stopped by user.")
    finally:
        print("Cleaning up...")
        try:
            if npc_vehicle_ids:
                client.apply_batch([carla.command.DestroyActor(x) for x in npc_vehicle_ids])
            if walker_ids:
                client.apply_batch([carla.command.DestroyActor(x) for x in walker_ids])
        except:
            pass
        for actor in reversed(actors):
            try:
                actor.destroy()
            except:
                pass

        # Restore async mode
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except:
            pass
        print("Done!")


if __name__ == '__main__':
    main()
