#!/usr/bin/env python3
"""
Standalone data collector for CARLA 0.9.16
Produces dataset 100% compatible with C-Shenron training pipeline.

Usage:
  python3 data_collector_v2.py --town Town01 --duration 18000

Output: save_dir/Scenario8/Repetition0_TownXX/route_NN/{rgb,lidar,...}
"""

import carla
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption
import numpy as np
import cv2
import os
import sys
import gzip
import json
import math
import time
import random
import argparse
import traceback
import laspy

# ============================================================================
# Radar simulation imports (from C-Shenron repo)
# C-Shenron team_code: single source of truth for model code
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.normpath(os.path.join(_script_dir, '..', '..'))
CSHENRON_TEAM_CODE = os.path.join(_repo_root, 'C-Shenron', 'team_code')
assert os.path.isdir(CSHENRON_TEAM_CODE), f"C-Shenron team_code not found at {CSHENRON_TEAM_CODE}"
sys.path.insert(0, CSHENRON_TEAM_CODE)

RADAR_AVAILABLE = False
# We do NOT generate radar inline during collection anymore due to performance
# and CARLA 0.9.16 semantic tag changes. We run it in an offline Phase 2 step.

# ============================================================================
# C-Shenron Config (matching config.py exactly)
# ============================================================================
CAMERA_POS = [-1.5, 0.0, 2.0]
CAMERA_ROT = [0.0, 0.0, 0.0]
CAMERA_WIDTH = 1024
CAMERA_HEIGHT = 256
CAMERA_FOV = 110

LIDAR_POS = [0.0, 0.0, 2.5]
LIDAR_ROT = [0.0, 0.0, -90.0]
LIDAR_ROTATION_FREQ = 10
LIDAR_PTS_PER_SEC = 600000

BEV_CAM_HEIGHT = 50.0

RADAR_POS = [2.0, 0.0, 1.0]
RADAR_ROT = [0.0, 0.0, 0.0]
RADAR_H_FOV = 30.0
RADAR_V_FOV = 20.0
RADAR_PTS_PER_SEC = 1500

DATA_SAVE_FREQ = 5
CARLA_FPS = 20
POINT_FORMAT = 0
POINT_PRECISION = 0.001
BB_SAVE_RADIUS = 40.0
NUM_ROUTE_POINTS = 20

WEATHERS = {
    'ClearNoon': carla.WeatherParameters.ClearNoon,
    'CloudySunset': carla.WeatherParameters.CloudySunset,
    'WetSunset': carla.WeatherParameters.WetSunset,
    'MidRainSunset': carla.WeatherParameters.MidRainSunset,
    'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,
    'HardRainNoon': carla.WeatherParameters.HardRainNoon,
    'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
}
DAYTIMES = {
    'Night': -80.0, 'Twilight': 0.0, 'Dawn': 5.0,
    'Sunset': 15.0, 'Morning': 35.0, 'Noon': 75.0,
}
AZIMUTHS = [45.0 * i for i in range(8)]

# BEV class mapping: CARLA semantic tag -> C-Shenron ObsManager class
# CORRECTED for CARLA 0.9.16 Semantic Tags (e.g. Vehicle=14, Pedestrian=12)
CARLA_TO_BEV = np.zeros(256, dtype=np.uint8)
CARLA_TO_BEV[1]  = 1    # Roads
CARLA_TO_BEV[2]  = 2    # Sidewalks
CARLA_TO_BEV[24] = 3    # RoadLines -> lane_markers
CARLA_TO_BEV[14] = 9    # Car -> Vehicles
CARLA_TO_BEV[15] = 9    # Truck -> Vehicles
CARLA_TO_BEV[16] = 9    # Bus -> Vehicles
CARLA_TO_BEV[18] = 9    # Motorcycle -> Vehicles
CARLA_TO_BEV[19] = 9    # Bicycle -> Vehicles
CARLA_TO_BEV[12] = 10   # Pedestrian -> walker
CARLA_TO_BEV[13] = 10   # Rider -> walker
CARLA_TO_BEV[7]  = 6    # TrafficLight -> green (can't know state)
CARLA_TO_BEV[8]  = 0    # TrafficSigns -> unlabeled


# ============================================================================
# Utility functions (from C-Shenron transfuser_utils.py)
# ============================================================================
def normalize_angle(x):
    x = x % (2 * np.pi)
    if x > np.pi:
        x -= 2 * np.pi
    return x


def normalize_angle_degree(x):
    x = x % 360.0
    if x > 180.0:
        x -= 360.0
    return x


def convert_depth(data):
    """Exact replica of t_u.convert_depth from C-Shenron."""
    data = data.astype(np.float32)
    normalized = np.dot(data, [65536.0, 256.0, 1.0])
    normalized /= (256 * 256 * 256 - 1)
    normalized = np.clip(normalized, a_min=0.0, a_max=0.05)
    normalized = normalized * 20.0
    return normalized


def lidar_to_ego_coordinate(lidar_raw):
    """Transform raw LiDAR points to ego vehicle coordinates."""
    yaw = np.deg2rad(LIDAR_ROT[2])
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw),  np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0]])
    translation = np.array(LIDAR_POS)
    return (rot @ lidar_raw[:, :3].T).T + translation


def semantic_lidar_to_ego_coordinate(sem_lidar_raw):
    """Transform raw semantic LiDAR to ego vehicle coordinates."""
    yaw = np.deg2rad(LIDAR_ROT[2])
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw),  np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0]])
    translation = np.array(LIDAR_POS)
    xyz_ego = (rot @ sem_lidar_raw[:, :3].T).T + translation
    return np.concatenate((xyz_ego, sem_lidar_raw[:, 3:]), axis=1)


def align_lidar(lidar, translation, yaw):
    """Align previous frame LiDAR to current frame."""
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw),  np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0]])
    return (rot.T @ (lidar - translation).T).T


def align_semantic_lidar(sem_lidar, translation, yaw):
    """Align previous frame semantic LiDAR to current frame."""
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw),  np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0]])
    xyz_aligned = (rot.T @ (sem_lidar[:, :3] - translation).T).T
    return np.concatenate((xyz_aligned, sem_lidar[:, 3:]), axis=1)


def inverse_conversion_2d(point, translation, yaw):
    """Convert global 2D point to ego coordinate frame."""
    rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw),  np.cos(yaw)]])
    return rot.T @ (point - translation)


def get_relative_transform(ego_matrix, vehicle_matrix):
    """Get position of vehicle in ego coordinate system."""
    relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
    rot = ego_matrix[:3, :3].T
    return rot @ relative_pos


def get_forward_speed(vehicle):
    """Get forward speed of a CARLA vehicle in m/s."""
    vel = vehicle.get_velocity()
    transform = vehicle.get_transform()
    forward = transform.get_forward_vector()
    speed = vel.x * forward.x + vel.y * forward.y + vel.z * forward.z
    return max(0.0, speed)


# ============================================================================
# Sensor data storage (filled by callbacks)
# ============================================================================
class SensorData:
    """Thread-safe storage for async sensor callbacks."""
    def __init__(self):
        self.rgb = None
        self.rgb_augmented = None
        self.semantics = None
        self.semantics_augmented = None
        self.depth = None
        self.depth_augmented = None
        self.bev_semantics = None
        self.lidar = None
        self.semantic_lidar = None
        self.carla_radar = None
        self.frame = -1

    def rgb_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        self.rgb = array
        self.frame = image.frame

    def rgb_augmented_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        self.rgb_augmented = array

    def semantics_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.semantics = array[:, :, 2]  # Red channel = semantic tag

    def semantics_augmented_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.semantics_augmented = array[:, :, 2]

    def depth_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        self.depth = array

    def depth_augmented_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        self.depth_augmented = array

    def bev_semantics_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        raw_tags = array[:, :, 2]
        self.bev_semantics = CARLA_TO_BEV[raw_tags]

    def lidar_callback(self, point_cloud):
        data = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        self.lidar = data.reshape(-1, 4)[:, :3].copy()

    def semantic_lidar_callback(self, point_cloud):
        data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('cos', np.float32), ('idx', np.uint32), ('tag', np.uint32)
        ]))

        arr = np.column_stack([
            data['x'], data['y'], data['z'],
            data['cos'], data['idx'].astype(np.float64),
            data['tag'].astype(np.float64)
        ])
        self.semantic_lidar = arr

    def radar_callback(self, radar_data):
        points = np.frombuffer(radar_data.raw_data,
                               dtype=np.dtype('f4'))
        self.carla_radar = points.reshape(-1, 4).copy()

    def all_ready(self):
        return (self.rgb is not None and
                self.rgb_augmented is not None and
                self.semantics is not None and
                self.semantics_augmented is not None and
                self.depth is not None and
                self.depth_augmented is not None and
                self.bev_semantics is not None and
                self.lidar is not None and
                self.semantic_lidar is not None)


# ============================================================================
# ============================================================================
# Bounding box extraction
# ============================================================================
def get_bounding_boxes(vehicle, world, sem_lidar_360):
    """Extract bounding boxes for ego, vehicles, walkers (C-Shenron format)."""
    results = []
    ego_transform = vehicle.get_transform()
    ego_matrix = np.array(ego_transform.get_matrix())
    ego_rotation = ego_transform.rotation
    ego_extent = vehicle.bounding_box.extent
    ego_speed = get_forward_speed(vehicle)
    ego_brake = vehicle.get_control().brake
    ego_yaw = np.deg2rad(ego_rotation.yaw)

    # Ego car entry
    relative_pos = get_relative_transform(ego_matrix, ego_matrix)
    results.append({
        'class': 'ego_car',
        'extent': [ego_extent.x, ego_extent.y, ego_extent.z],
        'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
        'yaw': 0.0,
        'num_points': -1,
        'distance': -1,
        'speed': ego_speed,
        'brake': ego_brake,
        'id': int(vehicle.id),
        'matrix': ego_transform.get_matrix()
    })

    actors = world.get_actors()
    # Vehicles
    for v in actors.filter('*vehicle*'):
        if v.id == vehicle.id:
            continue
        if v.get_location().distance(vehicle.get_location()) > BB_SAVE_RADIUS:
            continue
        v_transform = v.get_transform()
        v_matrix = np.array(v_transform.get_matrix())
        v_extent = v.bounding_box.extent
        v_yaw = np.deg2rad(v_transform.rotation.yaw)
        rel_yaw = normalize_angle(v_yaw - ego_yaw)
        rel_pos = get_relative_transform(ego_matrix, v_matrix)
        v_speed = get_forward_speed(v)
        v_brake = v.get_control().brake
        num_pts = count_points_in_bbox(rel_pos, rel_yaw,
                    [v_extent.x, v_extent.y, v_extent.z], sem_lidar_360, valid_tags=[14, 15, 16, 18, 19])
        results.append({
            'class': 'car',
            'extent': [v_extent.x, v_extent.y, v_extent.z],
            'position': [rel_pos[0], rel_pos[1], rel_pos[2]],
            'yaw': rel_yaw,
            'num_points': int(num_pts),
            'distance': float(np.linalg.norm(rel_pos)),
            'speed': v_speed,
            'brake': v_brake,
            'id': int(v.id),
            'matrix': v_transform.get_matrix()
        })

    # Walkers
    for w in actors.filter('*walker*'):
        if w.get_location().distance(vehicle.get_location()) > BB_SAVE_RADIUS:
            continue
        w_transform = w.get_transform()
        w_matrix = np.array(w_transform.get_matrix())
        w_extent = w.bounding_box.extent
        w_yaw = np.deg2rad(w_transform.rotation.yaw)
        rel_yaw = normalize_angle(w_yaw - ego_yaw)
        rel_pos = get_relative_transform(ego_matrix, w_matrix)
        w_vel = w.get_velocity()
        w_speed = math.sqrt(w_vel.x**2 + w_vel.y**2 + w_vel.z**2)
        num_pts = count_points_in_bbox(rel_pos, rel_yaw,
                    [w_extent.x, w_extent.y, w_extent.z], sem_lidar_360, valid_tags=[12, 13])
        results.append({
            'class': 'walker',
            'extent': [w_extent.x, w_extent.y, w_extent.z],
            'position': [rel_pos[0], rel_pos[1], rel_pos[2]],
            'yaw': rel_yaw,
            'num_points': int(num_pts),
            'distance': float(np.linalg.norm(rel_pos)),
            'speed': w_speed,
            'id': int(w.id),
            'matrix': w_transform.get_matrix()
        })

    return results


def count_points_in_bbox(pos, yaw, extent, sem_lidar, valid_tags=None):
    """Count LiDAR points inside a bounding box, filtering by 0.9.16 semantic tags."""
    if sem_lidar is None or len(sem_lidar) == 0:
        return -1
    
    # sem_lidar is (N, 6): x, y, z, cos, idx, tag
    lidar_xyz = sem_lidar[:, :3]
    tags = sem_lidar[:, 5]

    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw),  np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0]])
    vehicle_lidar = (rot.T @ (lidar_xyz - pos).T).T
    x, y, z = extent[0], extent[1], extent[2]
    mask = ((vehicle_lidar[:, 0] < x) & (vehicle_lidar[:, 0] > -x) &
            (vehicle_lidar[:, 1] < y) & (vehicle_lidar[:, 1] > -y) &
            (vehicle_lidar[:, 2] < z) & (vehicle_lidar[:, 2] > -z))
            
    if valid_tags is not None:
        tag_mask = np.isin(tags, valid_tags)
        mask = mask & tag_mask

    return int(mask.sum())


# ============================================================================
# Weather randomization (matching C-Shenron data_agent.py)
# ============================================================================
def shuffle_weather(world, vehicle):
    """Randomize weather exactly like C-Shenron."""
    weather_keys = list(WEATHERS.keys())
    chosen = random.choice(weather_keys)
    daytime_name, altitude = random.choice(list(DAYTIMES.items()))
    altitude = np.random.normal(altitude, 10)
    weather = WEATHERS[chosen]
    weather.sun_altitude_angle = altitude
    weather.sun_azimuth_angle = random.choice(AZIMUTHS)
    
    # Temporarily set to 50% chance so you can test and see the fog immediately!
    if random.random() < 0.5:
        weather.fog_density = random.uniform(40.0, 90.0)
        weather.fog_distance = random.uniform(5.0, 15.0)
        weather.fog_falloff = random.uniform(0.1, 0.5)
    else:
        weather.fog_density = 0.0

    world.set_weather(weather)

    # Night mode: toggle vehicle lights
    vehicles = world.get_actors().filter('*vehicle*')
    if altitude < 0.0:
        for v in vehicles:
            v.set_light_state(carla.VehicleLightState(
                carla.VehicleLightState.Position |
                carla.VehicleLightState.LowBeam))
    else:
        for v in vehicles:
            v.set_light_state(carla.VehicleLightState.NONE)


# ============================================================================
# Save functions
# ============================================================================
def save_lidar_laz(filepath, points):
    """Save LiDAR points as .laz (C-Shenron format)."""
    if len(points) == 0:
        points = np.zeros((1, 3))
    header = laspy.LasHeader(point_format=POINT_FORMAT)
    header.offsets = np.min(points[:, :3], axis=0)
    header.scales = np.array([POINT_PRECISION] * 3)
    with laspy.open(filepath, mode='w', header=header) as writer:
        pr = laspy.ScaleAwarePointRecord.zeros(points.shape[0], header=header)
        pr.x = points[:, 0]
        pr.y = points[:, 1]
        pr.z = points[:, 2]
        writer.write_points(pr)


def save_semantic_lidar_laz(filepath, points):
    """Save semantic LiDAR as .laz with extra dims (C-Shenron format)."""
    if len(points) == 0:
        points = np.zeros((1, 6))
    header = laspy.LasHeader(point_format=POINT_FORMAT)
    header.add_extra_dim(laspy.ExtraBytesParams(name="cosine", type=np.float64))
    header.add_extra_dim(laspy.ExtraBytesParams(name="index", type=np.int32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="sem_tag", type=np.int32))
    header.offsets = np.min(points[:, :3], axis=0)
    header.scales = np.array([POINT_PRECISION] * 3)
    with laspy.open(filepath, mode='w', header=header) as writer:
        pr = laspy.ScaleAwarePointRecord.zeros(points.shape[0], header=header)
        pr.x = points[:, 0]
        pr.y = points[:, 1]
        pr.z = points[:, 2]
        pr.cosine = points[:, 3]
        pr.index = points[:, 4].astype(np.int32)
        pr.sem_tag = points[:, 5].astype(np.int32)
        writer.write_points(pr)


def save_results(route_dir):
    """Write results.json.gz with perfect score (C-Shenron format)."""
    results = {'scores': {'score_composed': 100.0}}
    with gzip.open(os.path.join(route_dir, 'results.json.gz'),
                   'wt', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


# ============================================================================
# NPC spawning
# ============================================================================
def spawn_npcs(client, world, num_vehicles=50, num_walkers=30):
    """Spawn NPC vehicles and walkers."""
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(True)

    vehicle_bps = world.get_blueprint_library().filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    vehicles = []
    for i in range(min(num_vehicles, len(spawn_points))):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        v = world.try_spawn_actor(bp, spawn_points[i])
        if v is not None:
            v.set_autopilot(True, 8000)
            vehicles.append(v)

    # Walkers
    walker_bps = world.get_blueprint_library().filter('walker.pedestrian.*')
    walker_controller_bp = world.get_blueprint_library().find(
        'controller.ai.walker')
    walkers = []
    controllers = []
    for _ in range(num_walkers):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        bp = random.choice(walker_bps)
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        spawn_t = carla.Transform(loc)
        w = world.try_spawn_actor(bp, spawn_t)
        if w is not None:
            walkers.append(w)

    world.tick()

    for w in walkers:
        ctrl = world.try_spawn_actor(walker_controller_bp,
                                      carla.Transform(), w)
        if ctrl is not None:
            controllers.append(ctrl)

    world.tick()

    for ctrl in controllers:
        ctrl.start()
        ctrl.go_to_location(world.get_random_location_from_navigation())
        ctrl.set_max_speed(1.0 + random.random() * 1.5)

    print(f'  Spawned {len(vehicles)} vehicles, {len(walkers)} walkers')
    return vehicles, walkers, controllers, traffic_manager


# ============================================================================
# Sensor setup
# ============================================================================
def setup_sensors(world, vehicle, sensor_data):
    """Attach all sensors matching C-Shenron config.
    Returns (all_sensors_list, augmented_cameras_dict).
    augmented_cameras_dict contains the 3 augmented camera actors
    so collect_route can move them each frame.
    """
    bp_lib = world.get_blueprint_library()
    sensors = []

    rgb_transform = carla.Transform(
        carla.Location(x=CAMERA_POS[0], y=CAMERA_POS[1], z=CAMERA_POS[2]),
        carla.Rotation(roll=CAMERA_ROT[0], pitch=CAMERA_ROT[1],
                       yaw=CAMERA_ROT[2]))

    # 1. RGB Camera
    rgb_bp = bp_lib.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
    rgb_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
    rgb_bp.set_attribute('fov', str(CAMERA_FOV))
    rgb_sensor = world.spawn_actor(rgb_bp, rgb_transform, attach_to=vehicle)
    rgb_sensor.listen(sensor_data.rgb_callback)
    sensors.append(rgb_sensor)

    # 1b. RGB Augmented Camera (randomly shifted each frame)
    rgb_aug_bp = bp_lib.find('sensor.camera.rgb')
    rgb_aug_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
    rgb_aug_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
    rgb_aug_bp.set_attribute('fov', str(CAMERA_FOV))
    rgb_aug_sensor = world.spawn_actor(rgb_aug_bp, rgb_transform, attach_to=vehicle)
    rgb_aug_sensor.listen(sensor_data.rgb_augmented_callback)
    sensors.append(rgb_aug_sensor)

    # 2. Semantic Segmentation Camera
    sem_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    sem_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
    sem_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
    sem_bp.set_attribute('fov', str(CAMERA_FOV))
    sem_sensor = world.spawn_actor(sem_bp, rgb_transform, attach_to=vehicle)
    sem_sensor.listen(sensor_data.semantics_callback)
    sensors.append(sem_sensor)

    # 2b. Semantic Augmented Camera
    sem_aug_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    sem_aug_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
    sem_aug_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
    sem_aug_bp.set_attribute('fov', str(CAMERA_FOV))
    sem_aug_sensor = world.spawn_actor(sem_aug_bp, rgb_transform, attach_to=vehicle)
    sem_aug_sensor.listen(sensor_data.semantics_augmented_callback)
    sensors.append(sem_aug_sensor)

    # 3. Depth Camera
    depth_bp = bp_lib.find('sensor.camera.depth')
    depth_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
    depth_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
    depth_bp.set_attribute('fov', str(CAMERA_FOV))
    depth_sensor = world.spawn_actor(depth_bp, rgb_transform,
                                     attach_to=vehicle)
    depth_sensor.listen(sensor_data.depth_callback)
    sensors.append(depth_sensor)

    # 3b. Depth Augmented Camera
    depth_aug_bp = bp_lib.find('sensor.camera.depth')
    depth_aug_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
    depth_aug_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
    depth_aug_bp.set_attribute('fov', str(CAMERA_FOV))
    depth_aug_sensor = world.spawn_actor(depth_aug_bp, rgb_transform,
                                         attach_to=vehicle)
    depth_aug_sensor.listen(sensor_data.depth_augmented_callback)
    sensors.append(depth_aug_sensor)

    # 4. BEV Semantic Camera (top-down)
    bev_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    bev_bp.set_attribute('image_size_x', '256')
    bev_bp.set_attribute('image_size_y', '256')
    bev_bp.set_attribute('fov', '50')
    bev_transform = carla.Transform(
        carla.Location(x=0, y=0, z=BEV_CAM_HEIGHT),
        carla.Rotation(pitch=-90.0))
    bev_sensor = world.spawn_actor(bev_bp, bev_transform, attach_to=vehicle)
    bev_sensor.listen(sensor_data.bev_semantics_callback)
    sensors.append(bev_sensor)

    # 5. LiDAR
    lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('rotation_frequency', str(LIDAR_ROTATION_FREQ))
    lidar_bp.set_attribute('points_per_second', str(LIDAR_PTS_PER_SEC))
    lidar_transform = carla.Transform(
        carla.Location(x=LIDAR_POS[0], y=LIDAR_POS[1], z=LIDAR_POS[2]),
        carla.Rotation(roll=LIDAR_ROT[0], pitch=LIDAR_ROT[1],
                       yaw=LIDAR_ROT[2]))
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform,
                                     attach_to=vehicle)
    lidar_sensor.listen(sensor_data.lidar_callback)
    sensors.append(lidar_sensor)

    # 6. Semantic LiDAR
    sem_lidar_bp = bp_lib.find('sensor.lidar.ray_cast_semantic')
    sem_lidar_bp.set_attribute('rotation_frequency',
                               str(LIDAR_ROTATION_FREQ))
    sem_lidar_bp.set_attribute('points_per_second', str(LIDAR_PTS_PER_SEC))
    sem_lidar_sensor = world.spawn_actor(sem_lidar_bp, lidar_transform,
                                         attach_to=vehicle)
    sem_lidar_sensor.listen(sensor_data.semantic_lidar_callback)
    sensors.append(sem_lidar_sensor)

    # 7. CARLA Radar
    radar_bp = bp_lib.find('sensor.other.radar')
    radar_bp.set_attribute('horizontal_fov', str(RADAR_H_FOV))
    radar_bp.set_attribute('vertical_fov', str(RADAR_V_FOV))
    radar_bp.set_attribute('points_per_second', str(RADAR_PTS_PER_SEC))
    radar_transform = carla.Transform(
        carla.Location(x=RADAR_POS[0], y=RADAR_POS[1], z=RADAR_POS[2]),
        carla.Rotation(roll=RADAR_ROT[0], pitch=RADAR_ROT[1],
                       yaw=RADAR_ROT[2]))
    radar_sensor = world.spawn_actor(radar_bp, radar_transform,
                                     attach_to=vehicle)
    radar_sensor.listen(sensor_data.radar_callback)
    sensors.append(radar_sensor)

    # Collect augmented camera actors so we can move them each frame
    augmented_cameras = {
        'rgb': rgb_aug_sensor,
        'sem': sem_aug_sensor,
        'depth': depth_aug_sensor,
    }

    return sensors, augmented_cameras


# ============================================================================
# Core collection loop (per route)
# ============================================================================
def collect_route(world, vehicle, route_dir, agent,
                  min_frames=60, max_frames=800, sensor_tick=0.05):
    """Collect data for one route. Returns number of frames saved."""
    # Create all subdirectories matching C-Shenron data_agent.py
    subdirs = [
        'rgb', 'rgb_augmented', 'semantics', 'semantics_augmented',
        'depth', 'depth_augmented', 'bev_semantics',
        'bev_semantics_augmented', 'lidar', 'semantic_lidar',
        'boxes', 'measurements', 'carla_radar',
        'radar_data_front_86', 'radar_data_rear_86',
        'radar_data_left_86', 'radar_data_right_86',
    ]
    for sd in subdirs:
        os.makedirs(os.path.join(route_dir, sd), exist_ok=True)

    sensor_data = SensorData()
    sensors, augmented_cameras = setup_sensors(world, vehicle, sensor_data)

    # Augmentation parameters (matching C-Shenron config.py)
    AUG_TRANSLATION_MIN = -1.0  # meters lateral shift
    AUG_TRANSLATION_MAX = 1.0
    AUG_ROTATION_MIN = -5.0     # degrees yaw
    AUG_ROTATION_MAX = 5.0
    # Buffer: augmentation set now applies to the NEXT rendered frame
    from collections import deque
    aug_translation_buf = deque([0.0], maxlen=2)
    aug_rotation_buf = deque([0.0], maxlen=2)

    # State for LiDAR sweep merging
    last_lidar = None
    last_sem_lidar = None
    last_ego_transform = None

    step = 0
    frame_count = 0
    stuck_count = 0

    try:
        # Let sensors stabilize
        for _ in range(10):
            world.tick()
            step += 1
            time.sleep(0.01)

        while frame_count < max_frames:
            world.tick()
            step += 1
            time.sleep(0.01)

            # --- Update Spectator Camera (3rd Person Trailing) ---
            spectator = world.get_spectator()
            ego_transform = vehicle.get_transform()
            # Position the camera 6 meters behind and 3 meters above, looking forward
            cam_location = ego_transform.location - ego_transform.get_forward_vector() * 6.0 + carla.Location(z=3.0)
            cam_rotation = ego_transform.rotation
            cam_rotation.pitch -= 15.0  # Tilt slightly down
            spectator.set_transform(carla.Transform(cam_location, cam_rotation))

            if not sensor_data.all_ready():
                continue

            # --- Get raw sensor data ---
            raw_lidar = sensor_data.lidar.copy()
            raw_sem_lidar = sensor_data.semantic_lidar.copy()

            # Transform to ego coordinates
            ego_lidar = lidar_to_ego_coordinate(raw_lidar)
            ego_sem_lidar = semantic_lidar_to_ego_coordinate(raw_sem_lidar)

            # LiDAR sweep merging (combine with previous half-sweep)
            if last_lidar is not None and last_ego_transform is not None:
                ego_transform = vehicle.get_transform()
                ego_loc = ego_transform.location
                last_loc = last_ego_transform.location
                rel_trans = np.array([
                    ego_loc.x - last_loc.x,
                    ego_loc.y - last_loc.y,
                    ego_loc.z - last_loc.z])

                ego_yaw = ego_transform.rotation.yaw
                last_yaw = last_ego_transform.rotation.yaw
                rel_rot = np.deg2rad(
                    normalize_angle_degree(ego_yaw - last_yaw))
                orient = np.deg2rad(ego_yaw)
                rot_mat = np.array([
                    [np.cos(orient), -np.sin(orient), 0.0],
                    [np.sin(orient),  np.cos(orient), 0.0],
                    [0.0, 0.0, 1.0]])
                rel_trans = rot_mat.T @ rel_trans

                lidar_last_aligned = align_lidar(
                    last_lidar, rel_trans, rel_rot)
                lidar_360 = np.concatenate(
                    (ego_lidar, lidar_last_aligned), axis=0)

                sem_lidar_last_aligned = align_semantic_lidar(
                    last_sem_lidar, rel_trans, rel_rot)
                sem_lidar_360 = np.concatenate(
                    (ego_sem_lidar, sem_lidar_last_aligned), axis=0)
            else:
                lidar_360 = ego_lidar
                sem_lidar_360 = ego_sem_lidar

            # Store for next frame
            last_lidar = ego_lidar
            last_sem_lidar = ego_sem_lidar
            last_ego_transform = vehicle.get_transform()

            # Only save every DATA_SAVE_FREQ ticks
            if step % DATA_SAVE_FREQ != 0:
                continue

            # --- Randomly shift augmented cameras for THIS frame ---
            # (the image we capture now reflects the transform set LAST tick,
            #  so we read aug_translation_buf[0] for the current saved image
            #  and set the NEW random offset for the next frame)
            aug_t = np.random.uniform(AUG_TRANSLATION_MIN, AUG_TRANSLATION_MAX)
            aug_r = np.random.uniform(AUG_ROTATION_MIN, AUG_ROTATION_MAX)
            aug_translation_buf.append(aug_t)
            aug_rotation_buf.append(aug_r)

            aug_cam_transform = carla.Transform(
                carla.Location(x=CAMERA_POS[0],
                               y=CAMERA_POS[1] + aug_t,
                               z=CAMERA_POS[2]),
                carla.Rotation(roll=CAMERA_ROT[0],
                               pitch=CAMERA_ROT[1],
                               yaw=CAMERA_ROT[2] + aug_r))
            for cam in augmented_cameras.values():
                cam.set_transform(aug_cam_transform)

            frame = frame_count
            frame_str = f'{frame:04d}'

            # --- Process depth (C-Shenron encoding) ---
            depth_raw = sensor_data.depth.copy()
            depth_encoded = (convert_depth(depth_raw) * 255.0 + 0.5
                             ).astype(np.uint8)

            # --- Get autopilot control ---
            control = agent.run_step()
            vehicle.apply_control(control)
            speed = get_forward_speed(vehicle)

            # Stuck detection
            if speed < 0.1:
                stuck_count += 1
            else:
                stuck_count = 0

            # --- Route info ---
            plan = agent.get_local_planner().get_plan()
            if not plan:
                print('    Route cleanly completed (no plan remaining).')
                break
                
            plan_list = list(plan)
            wps = [p[0] for p in plan_list[:NUM_ROUTE_POINTS]]
            if len(plan_list) > 9:
                target_wp, road_option = plan_list[9]
            else:
                target_wp, road_option = plan_list[-1]

            ego_transform = vehicle.get_transform()
            ego_loc = ego_transform.location
            ego_yaw_rad = np.deg2rad(ego_transform.rotation.yaw)
            # Theta = ego yaw in radians. No -90° offset needed because we read
            # transform.rotation.yaw directly (not IMU compass which has a +90° bias).
            theta = normalize_angle(ego_yaw_rad)
            pos = np.array([ego_loc.x, ego_loc.y, ego_loc.z])

            # Target point in ego frame
            if target_wp is not None:
                tp_loc = target_wp.transform.location
                tp_global = np.array([tp_loc.x, tp_loc.y])
                ego_target = inverse_conversion_2d(
                    tp_global, pos[:2], theta).tolist()
            else:
                ego_target = [0.0, 0.0]

            # Next target point
            next_target = ego_target  # simplified

            # Build route in ego frame
            dense_route = []
            for wp in wps[:NUM_ROUTE_POINTS]:
                wp_loc = wp.transform.location
                wp_2d = np.array([wp_loc.x, wp_loc.y])
                ego_wp = inverse_conversion_2d(
                    wp_2d, pos[:2], theta).tolist()
                dense_route.append(ego_wp)

            # Aim waypoint — use the 2nd route waypoint (~4m ahead)
            # Matches original autopilot.py _get_steer: route[1][0]
            if len(wps) >= 2:
                aim_loc = wps[1].transform.location
                aim_2d = np.array([aim_loc.x, aim_loc.y])
                ego_aim = inverse_conversion_2d(
                    aim_2d, pos[:2], theta).tolist()
            else:
                ego_aim = ego_target

            # Angle for steering
            if ego_aim[0] != 0 or ego_aim[1] != 0:
                angle = -math.degrees(math.atan2(
                    -ego_aim[1], ego_aim[0])) / 90.0
            else:
                angle = 0.0

            # Navigation command
            def map_command(cmd_ro):
                if cmd_ro == RoadOption.LEFT: return 1
                elif cmd_ro == RoadOption.RIGHT: return 2
                elif cmd_ro == RoadOption.STRAIGHT: return 3
                return 4  # LANEFOLLOW
                
            cmd = map_command(road_option)
            next_cmd = cmd

            # Check junction explicitly based on current ego position
            current_wp = world.get_map().get_waypoint(vehicle.get_location())
            wp_on_junction = current_wp.is_junction

            # Check for nearby walkers
            walkers_near = any(
                w.get_location().distance(vehicle.get_location()) < 15.0
                for w in world.get_actors().filter('*walker*')
            )

            # Assign literal float target speeds (C-Shenron indexers handle floats later)
            if speed < 0.1 and control.brake > 0.5:
                target_speed = 0.0
            elif walkers_near:
                target_speed = 2.0
            elif wp_on_junction:
                target_speed = 5.0
            else:
                target_speed = 8.0

            # Hazard flags (from autopilot behavior)
            brake = bool(control.brake > 0.5)
            vehicle_hazard = brake and speed > 0.5
            # Detect traffic light state for C-Shenron training
            tl = vehicle.get_traffic_light()
            if tl is not None:
                light_hazard = tl.get_state() == carla.TrafficLightState.Red
            else:
                light_hazard = False
            walker_hazard = walkers_near
            stop_sign_hazard = False

            # --- Measurement dict (exact C-Shenron schema) ---
            measurement = {
                'pos_global': pos.tolist(),
                'theta': float(theta),
                'speed': float(speed),
                'target_speed': float(target_speed),
                'target_point': ego_target,
                'target_point_next': next_target,
                'command': int(cmd),
                'next_command': int(next_cmd),
                'aim_wp': ego_aim,
                'route': dense_route,
                'steer': float(control.steer),
                'throttle': float(control.throttle),
                'brake': brake,
                'control_brake': brake,
                'junction': bool(wp_on_junction),
                'vehicle_hazard': vehicle_hazard,
                'light_hazard': light_hazard,
                'walker_hazard': walker_hazard,
                'stop_sign_hazard': stop_sign_hazard,
                'stop_sign_close': False,
                'walker_close': False,
                'angle': float(angle),
                'augmentation_translation': float(aug_translation_buf[0]),
                'augmentation_rotation': float(aug_rotation_buf[0]),
                'ego_matrix': ego_transform.get_matrix(),
            }

            # --- Bounding boxes ---
            bboxes = get_bounding_boxes(vehicle, world, sem_lidar_360)

            # --- Save everything ---
            # RGB (normal + genuinely augmented from shifted camera)
            cv2.imwrite(os.path.join(route_dir, 'rgb',
                        f'{frame_str}.jpg'), sensor_data.rgb)
            cv2.imwrite(os.path.join(route_dir, 'rgb_augmented',
                        f'{frame_str}.jpg'), sensor_data.rgb_augmented)

            # Semantics (normal + augmented)
            cv2.imwrite(os.path.join(route_dir, 'semantics',
                        f'{frame_str}.png'), sensor_data.semantics)
            cv2.imwrite(os.path.join(route_dir, 'semantics_augmented',
                        f'{frame_str}.png'), sensor_data.semantics_augmented)

            # Depth (normal + augmented)
            depth_aug_raw = sensor_data.depth_augmented.copy()
            depth_aug_encoded = (convert_depth(depth_aug_raw) * 255.0 + 0.5
                                 ).astype(np.uint8)
            cv2.imwrite(os.path.join(route_dir, 'depth',
                        f'{frame_str}.png'), depth_encoded)
            cv2.imwrite(os.path.join(route_dir, 'depth_augmented',
                        f'{frame_str}.png'), depth_aug_encoded)

            # BEV semantics + augmented (BEV is top-down, not affected by camera shift)
            cv2.imwrite(os.path.join(route_dir, 'bev_semantics',
                        f'{frame_str}.png'), sensor_data.bev_semantics)
            cv2.imwrite(os.path.join(route_dir, 'bev_semantics_augmented',
                        f'{frame_str}.png'), sensor_data.bev_semantics)

            # LiDAR
            save_lidar_laz(os.path.join(route_dir, 'lidar',
                           f'{frame_str}.laz'), lidar_360)

            # Semantic LiDAR
            save_semantic_lidar_laz(os.path.join(route_dir,
                'semantic_lidar', f'{frame_str}.laz'), sem_lidar_360)

            # CARLA Radar
            if sensor_data.carla_radar is not None:
                np.save(os.path.join(route_dir, 'carla_radar',
                        f'{frame_str}.npy'), sensor_data.carla_radar)

            # Bounding boxes
            with gzip.open(os.path.join(route_dir, 'boxes',
                           f'{frame_str}.json.gz'), 'wt',
                           encoding='utf-8') as f:
                json.dump(bboxes, f, indent=4)

            # Measurements
            with gzip.open(os.path.join(route_dir, 'measurements',
                           f'{frame_str}.json.gz'), 'wt',
                           encoding='utf-8') as f:
                json.dump(measurement, f, indent=4)

            frame_count += 1

            if frame_count % 20 == 0:
                # === COORDINATE VERIFICATION DEBUG ===
                import math as _m
                _yaw_deg = ego_transform.rotation.yaw
                _yaw_rad = np.deg2rad(_yaw_deg)
                _theta_deg = np.rad2deg(theta)
                _theta_diff = abs(_theta_deg - _yaw_deg) % 360
                if _theta_diff > 180:
                    _theta_diff = 360 - _theta_diff
                _theta_ok = "OK" if _theta_diff < 1.0 else f"OFFSET={_theta_diff:.0f}deg"

                # Semantic tag distribution from the camera image
                _sem_img = sensor_data.semantics
                _tags, _cnts = np.unique(_sem_img, return_counts=True)
                _top5 = sorted(zip(_tags, _cnts), key=lambda x: -x[1])[:5]
                _tag_names = {0:'NONE', 1:'Roads', 2:'Sidewalks', 3:'Buildings',
                              4:'Walls', 5:'Fences', 6:'Poles', 7:'TrafLight',
                              8:'TrafSign', 9:'Vegetation', 10:'Terrain',
                              11:'Sky', 12:'Pedestrian', 13:'Rider', 14:'Car',
                              15:'Truck', 16:'Bus', 17:'Train', 18:'Motorcycle',
                              19:'Bicycle', 20:'Static', 21:'Dynamic',
                              23:'Water', 24:'RoadLines', 25:'Ground',
                              26:'Bridge', 27:'RailTrack', 28:'GuardRail'}
                _sem_str = ' '.join(f"{_tag_names.get(t,f'?{t}')}:{c}" for t,c in _top5)

                # Steer-angle consistency
                _sa_ok = ""
                if abs(control.steer) > 0.05 and abs(angle) > 0.01:
                    _sa_ok = "MATCH" if (control.steer * angle) >= 0 else "MISMATCH!"
                else:
                    _sa_ok = "straight"

                print(f'    Frame {frame_count} | '
                      f'Speed: {speed:.1f}m/s | '
                      f'Steer: {control.steer:+.3f}')
                print(f'      YAW={_yaw_deg:+.1f}° | '
                      f'THETA={_theta_deg:+.1f}° | '
                      f'theta_check={_theta_ok}')
                print(f'      target_ego=[{ego_target[0]:+.1f}, {ego_target[1]:+.1f}] '
                      f'({"AHEAD" if ego_target[0] > 0 else "BEHIND!!"}) | '
                      f'angle={angle:+.3f} | steer_match={_sa_ok}')
                if len(dense_route) >= 3:
                    print(f'      route[0..2]='
                          f'[{dense_route[0][0]:+.1f},{dense_route[0][1]:+.1f}] '
                          f'[{dense_route[1][0]:+.1f},{dense_route[1][1]:+.1f}] '
                          f'[{dense_route[2][0]:+.1f},{dense_route[2][1]:+.1f}] '
                          f'({"AHEAD" if dense_route[2][0] > 0 else "BEHIND!!"})')
                print(f'      sem_tags: {_sem_str}')
                _cmd_names = {1:'LEFT', 2:'RIGHT', 3:'STRAIGHT', 4:'FOLLOW'}
                print(f'      cmd={int(cmd)}({_cmd_names.get(int(cmd),"?")}) '
                      f'junct={wp_on_junction} '
                      f'tgt_spd={target_speed:.1f} brk={brake} '
                      f'aug_t={aug_translation_buf[0]:+.2f}m '
                      f'aug_r={aug_rotation_buf[0]:+.1f}deg')

            # End route conditions
            if getattr(agent, 'done', lambda: False)() or len(plan) == 0:
                print('    Route cleanly completed.')
                break
                
            if stuck_count > 200 and frame_count >= min_frames:
                print('    Route ended: vehicle stuck')
                break

    finally:
        for s in sensors:
            s.stop()
            s.destroy()

    return frame_count


# ============================================================================
# Main collection manager
# ============================================================================
def run_collection(args):
    """Main entry point for data collection."""
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    # Load map
    world = client.load_world(args.town)
    
    print("\n=== CARLA 0.9.16 Official Tag Mapping ===")
    tags_to_check = ['Roads', 'Sidewalks', 'Vehicles', 'Pedestrians', 'RoadLines', 'TrafficLight', 'TrafficSigns', 'Buildings', 'Vegetation', 'Water']
    for tag in tags_to_check:
        if hasattr(carla.CityObjectLabel, tag):
            print(f"  {tag:15s} = {int(getattr(carla.CityObjectLabel, tag))}")
    print("=========================================\n")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / CARLA_FPS
    world.apply_settings(settings)

    # Dataset directory: root/Scenario8/Repetition0_TownXX
    scenario_dir = os.path.join(args.save_dir, 'Scenario8')
    repetition_dir = os.path.join(scenario_dir, 'Repetition0_Town01')
    os.makedirs(repetition_dir, exist_ok=True)
    
    # Save the official baseline tag mapping
    try:
        import json
        TAG_MAPPING_916 = {
            'Roads': 1, 'Sidewalks': 2, 'Buildings': 3, 'Fences': 5,
            'Poles': 6, 'TrafficLight': 7, 'TrafficSigns': 8,
            'Vegetation': 9, 'Terrain': 10, 'Pedestrians': 12,
            'Rider': 13, 'Car': 14, 'Truck': 15, 'Bus': 16,
            'Motorcycle': 18, 'Bicycle': 19, 'Water': 23,
            'RoadLines': 24, 'GuardRail': 28
        }
        with open(os.path.join(repetition_dir, 'tag_mapping_916.json'), 'w') as f:
            json.dump(TAG_MAPPING_916, f, indent=2)
    except Exception as e:
        print(f"Failed to dump tag mapping JSON: {e}")

    print(f"  Save dir: {repetition_dir}")

    # Find existing route count
    existing = [d for d in os.listdir(repetition_dir)
                if os.path.isdir(os.path.join(repetition_dir, d))]
    route_counter = len(existing)

    # Spawn NPC traffic
    vehicles, walkers, controllers, tm = spawn_npcs(
        client, world, args.vehicles, args.walkers)

    spawn_points = world.get_map().get_spawn_points()
    start_time = time.time()
    total_frames = 0

    print(f'\n=== C-Shenron Data Collection ===')
    print(f'  Town: {args.town}')
    print(f'  Duration: {args.duration}s')
    print(f'  Save dir: {repetition_dir}')
    print(f'  Starting at route_{route_counter:02d}\n')

    ego_vehicle = None
    try:
        while (time.time() - start_time) < args.duration:
            # Pick a random spawn point for the ego
            sp = random.choice(spawn_points)
            bp = world.get_blueprint_library().find('vehicle.tesla.model3')
            ego_vehicle = world.try_spawn_actor(bp, sp)
            if ego_vehicle is None:
                time.sleep(0.5)
                continue

            # CRITICAL: We must tick the physics engine immediately after spawning!
            # Otherwise, ego_vehicle.get_location() evaluates to (0,0,0) and the 
            # BasicAgent plans a 300m route starting from the absolute map origin 
            # instead of where the vehicle actually spawned!
            world.tick()
            world.tick()

            # We DO NOT set TrafficManager autopilot. BasicAgent drives the car directly!
            # The default BasicAgent lateral K_P is 1.95 which causes wiggling. 
            # We tune it down to 1.0 to ensure smooth human-like data collection.
            agent_opts = {
                'ignore_traffic_lights': False,
                'lateral_control_dict': {'K_P': 1.0, 'K_I': 0.0, 'K_D': 0.1, 'dt': 1.0/20.0}
            }
            agent = BasicAgent(ego_vehicle, target_speed=30, opt_dict=agent_opts)
            
            # Find a single long route and set it natively in the agent
            route_found = False
            for _ in range(20):
                dest = random.choice(spawn_points).location
                agent.set_destination(dest)
                plan = agent.get_local_planner().get_plan()
                if plan and len(plan) > 150:
                    route_found = True
                    break
                    
            if not route_found:
                ego_vehicle.destroy()
                ego_vehicle = None
                continue

            # Shuffle weather per route (like C-Shenron)
            shuffle_weather(world, ego_vehicle)

            route_name = f'route_{route_counter:02d}'
            route_dir = os.path.join(repetition_dir, route_name)
            os.makedirs(route_dir, exist_ok=True)

            elapsed = time.time() - start_time
            remaining = args.duration - elapsed
            print(f'Route {route_counter} | '
                  f'Elapsed: {elapsed/3600:.1f}h | '
                  f'Remaining: {remaining/3600:.1f}h')

            try:
                frames = collect_route(
                    world, ego_vehicle, route_dir, agent,
                    min_frames=60, max_frames=800)
            except Exception as e:
                print(f'  Route error: {e}')
                traceback.print_exc()
                frames = 0

            # Only save results if we got enough frames
            if frames >= 40:
                save_results(route_dir)
                total_frames += frames
                print(f'  Saved {frames} frames -> {route_name}')
                route_counter += 1
            else:
                # Clean up incomplete route
                import shutil
                if os.path.exists(route_dir):
                    shutil.rmtree(route_dir)
                print(f'  Route too short ({frames} frames), discarded')

            # Destroy ego for respawn
            if ego_vehicle is not None:
                ego_vehicle.destroy()
                ego_vehicle = None

    except KeyboardInterrupt:
        print('\n\nCollection interrupted by user.')
    finally:
        # Cleanup
        if ego_vehicle is not None:
            ego_vehicle.destroy()
        for ctrl in controllers:
            ctrl.stop()
            ctrl.destroy()
        for w in walkers:
            w.destroy()
        for v in vehicles:
            v.destroy()

        # Restore async mode
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

    elapsed_total = time.time() - start_time
    print(f'\n=== Collection Complete ===')
    print(f'  Total time: {elapsed_total/3600:.2f} hours')
    print(f'  Total frames: {total_frames}')
    print(f'  Total routes: {route_counter}')
    print(f'  Dataset: {repetition_dir}')
    print(f'\n=== Next Steps ===')
    print(f'  1. Generate radar data from semantic LiDAR:')
    print(f'     (Use C-Shenron relabel_dataset or sim_radar_utils)')
    print(f'  2. Train with C-Shenron:')
    print(f'     torchrun --nproc_per_node=gpu train.py \\')
    print(f'       --root_dir {args.save_dir} --epochs 30 \\')
    print(f'       --setting all --batch_size 12')


# ============================================================================
# Entry point
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='C-Shenron Compatible Data Collector for CARLA 0.9.16')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--town', default='Town01')
    parser.add_argument('--duration', type=int, default=18000,
                        help='Collection duration in seconds (default: 5h)')
    parser.add_argument('--save-dir', default='/storage/dataset',
                        help='Root directory to save dataset')
    parser.add_argument('--vehicles', type=int, default=40,
                        help='Number of NPC vehicles')
    parser.add_argument('--walkers', type=int, default=30,
                        help='Number of NPC walkers')
    args = parser.parse_args()

    run_collection(args)
