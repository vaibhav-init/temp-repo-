#!/usr/bin/env python3
"""
Real-time Crash Probability Predictor for CARLA 0.9.16
======================================================

Runs the trained MLP model in real-time alongside an ego vehicle
driven by CARLA's Traffic Manager autopilot. Predicts crash probability
every frame and prints ALL information to the terminal.

NO intervention is performed — this is a prediction-only system.
The ego vehicle drives naturally (realistically) and the system
only observes and predicts.

Usage:
    python crash_predictor_live.py
    python crash_predictor_live.py --model models/crash_mlp.pth --scaler models/scaler.pkl
    python crash_predictor_live.py --duration 300 --npc-vehicles 50

Press Ctrl+C to stop.
"""

import carla
import numpy as np
import os
import math
import time
import random
import argparse
import pickle
import traceback

import torch
import torch.nn as nn


# ============================================================================
# Configuration
# ============================================================================
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
DEFAULT_TOWN = 'Town01'
FPS = 20

DEFAULT_MODEL_PATH = 'models/crash_mlp.pth'
DEFAULT_SCALER_PATH = 'models/scaler.pkl'

# Obstacle detection (same as data collector)
MAX_SEARCH_DISTANCE = 50.0
LANE_HALF_WIDTH = 4.0

# Feature columns (must match training)
FEATURE_COLUMNS = [
    'ego_speed', 'ego_acceleration', 'nearest_distance',
    'relative_velocity', 'ttc', 'obstacle_speed', 'obstacle_type'
]


# ============================================================================
# MLP Model (must match train_mlp.py architecture)
# ============================================================================
class CrashMLP(nn.Module):
    """MLP for crash probability prediction — architecture must match training."""

    def __init__(self, input_dim=7, hidden_dims=None, dropout=0.3):
        super(CrashMLP, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        layers = []
        prev_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            drop_rate = max(0.1, dropout - 0.1 * i)
            layers.append(nn.Dropout(drop_rate))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================================
# Obstacle Detection (same as data collector)
# ============================================================================
def find_nearest_obstacle(ego_vehicle, world):
    """
    Find the nearest obstacle (vehicle or pedestrian) in ego's forward cone.

    Returns dict with:
      distance, relative_velocity, obstacle_speed, obstacle_type, lateral_offset
    """
    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location
    ego_forward = ego_transform.get_forward_vector()
    ego_right = ego_transform.get_right_vector()
    ego_velocity = ego_vehicle.get_velocity()

    nearest = {
        'distance': MAX_SEARCH_DISTANCE,
        'relative_velocity': 0.0,
        'obstacle_speed': 0.0,
        'obstacle_type': 2,  # none
        'lateral_offset': 0.0,
    }

    vehicles = world.get_actors().filter('*vehicle*')
    pedestrians = world.get_actors().filter('*walker.pedestrian*')

    all_obstacles = []
    for v in vehicles:
        if v.id != ego_vehicle.id:
            all_obstacles.append((v, 0))  # vehicle
    for p in pedestrians:
        all_obstacles.append((p, 1))  # pedestrian

    for obstacle, obs_type in all_obstacles:
        obs_location = obstacle.get_location()
        world_dist = ego_location.distance(obs_location)
        if world_dist > MAX_SEARCH_DISTANCE:
            continue

        dx = obs_location.x - ego_location.x
        dy = obs_location.y - ego_location.y

        forward_dist = dx * ego_forward.x + dy * ego_forward.y
        lateral_dist = dx * ego_right.x + dy * ego_right.y

        if forward_dist < 1.0:
            continue
        if abs(lateral_dist) > LANE_HALF_WIDTH:
            continue

        if forward_dist < nearest['distance']:
            obs_velocity = obstacle.get_velocity()
            obs_speed = math.sqrt(obs_velocity.x**2 + obs_velocity.y**2 + obs_velocity.z**2)

            los_len = math.sqrt(dx**2 + dy**2)
            if los_len < 0.1:
                continue
            los_x = dx / los_len
            los_y = dy / los_len

            ego_approach = ego_velocity.x * los_x + ego_velocity.y * los_y
            obs_approach = obs_velocity.x * los_x + obs_velocity.y * los_y
            relative_vel = ego_approach - obs_approach

            nearest = {
                'distance': forward_dist,
                'relative_velocity': relative_vel,
                'obstacle_speed': obs_speed,
                'obstacle_type': obs_type,
                'lateral_offset': lateral_dist,
            }

    return nearest


# ============================================================================
# Collision Recorder
# ============================================================================
class CollisionRecorder:
    """Records collision events for prediction accuracy evaluation."""

    def __init__(self, vehicle, world):
        self.collisions = []
        self.frame_counter = [0]

        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        self.sensor.listen(self._on_collision)
        print("  ✅ Collision sensor attached")

    def _on_collision(self, event):
        frame = self.frame_counter[0]
        self.collisions.append({
            'frame': frame,
            'other_actor': event.other_actor.type_id,
            'impulse': event.normal_impulse.length(),
        })

    def cleanup(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


# ============================================================================
# Traffic Spawning (reused from data collector)
# ============================================================================
def spawn_npc_vehicles(world, client, traffic_manager, count=40):
    """Spawn NPC traffic vehicles with varied speeds."""
    bp_lib = world.get_blueprint_library()
    vehicle_bps = [bp for bp in bp_lib.filter('vehicle.*')
                   if int(bp.get_attribute('number_of_wheels')) >= 4]
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    batch = []
    for i in range(min(count, len(spawn_points))):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(
                bp.get_attribute('color').recommended_values))
        bp.set_attribute('role_name', 'autopilot')
        batch.append(carla.command.SpawnActor(bp, spawn_points[i]))

    vehicle_ids = []
    results = client.apply_batch_sync(batch, True)
    for result in results:
        if not result.error:
            vehicle_ids.append(result.actor_id)

    tm_port = traffic_manager.get_port()
    for vid in vehicle_ids:
        vehicle = world.get_actor(vid)
        if vehicle:
            vehicle.set_autopilot(True, tm_port)
            speed_diff = random.randint(-20, 50)
            traffic_manager.vehicle_percentage_speed_difference(vehicle, speed_diff)
            traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(3.0, 8.0))

    print(f"  🚗 Spawned {len(vehicle_ids)}/{count} NPC vehicles")
    return vehicle_ids


def spawn_pedestrians(world, client, count=30):
    """Spawn walking pedestrians with AI controllers."""
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    controller_bp = bp_lib.find('controller.ai.walker')

    walkers = []
    for _ in range(count):
        bp = random.choice(walker_bps)
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        w = world.try_spawn_actor(bp, carla.Transform(loc))
        if w is not None:
            walkers.append(w)

    controllers = []
    for w in walkers:
        ctrl = world.spawn_actor(controller_bp, carla.Transform(), attach_to=w)
        controllers.append(ctrl)

    world.tick()

    for ctrl in controllers:
        dest = world.get_random_location_from_navigation()
        if dest is not None:
            ctrl.start()
            ctrl.go_to_location(dest)
            ctrl.set_max_speed(1.0 + random.random() * 1.5)

    walker_ids = [w.id for w in walkers]
    ctrl_ids = [c.id for c in controllers]
    print(f"  🚶 Spawned {len(walkers)}/{count} pedestrians")
    return walker_ids, ctrl_ids


def spawn_challenging_scenario(world, ego_vehicle):
    """Spawn a random challenging scenario near ego."""
    scenario_type = random.choice([
        'sudden_brake', 'stopped_vehicle', 'jaywalker', 'tight_traffic'
    ])

    bp_lib = world.get_blueprint_library()
    carla_map = world.get_map()
    spawned_actors = []

    ego_loc = ego_vehicle.get_location()
    ego_wp = carla_map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

    if ego_wp is None:
        return spawned_actors, scenario_type

    print(f"\n  🎬 Scenario: {scenario_type.upper()}")

    if scenario_type == 'sudden_brake':
        wp = ego_wp
        for _ in range(7):
            nwps = wp.next(3.0)
            if not nwps:
                break
            wp = nwps[0]

        vehicle_bp = random.choice([bp for bp in bp_lib.filter('vehicle.*')
                                     if int(bp.get_attribute('number_of_wheels')) >= 4])
        spawn_tf = wp.transform
        spawn_tf.location.z += 0.5
        lead = world.try_spawn_actor(vehicle_bp, spawn_tf)
        if lead:
            lead.enable_constant_velocity(carla.Vector3D(5.0, 0, 0))
            spawned_actors.append({
                'actor': lead, 'type': 'sudden_braker',
                'brake_time': time.time() + random.uniform(4, 7), 'braked': False
            })
            print(f"     Lead vehicle will brake suddenly")

    elif scenario_type == 'stopped_vehicle':
        wp = ego_wp
        for _ in range(8):
            nwps = wp.next(3.0)
            if not nwps:
                break
            wp = nwps[0]

        vehicle_bp = random.choice([bp for bp in bp_lib.filter('vehicle.*')
                                     if int(bp.get_attribute('number_of_wheels')) >= 4])
        spawn_tf = wp.transform
        spawn_tf.location.z += 0.5
        stopped = world.try_spawn_actor(vehicle_bp, spawn_tf)
        if stopped:
            stopped.set_target_velocity(carla.Vector3D(0, 0, 0))
            stopped.apply_control(carla.VehicleControl(brake=1.0))
            spawned_actors.append({'actor': stopped, 'type': 'stopped'})
            print(f"     Stopped vehicle ahead")

    elif scenario_type == 'jaywalker':
        wp = ego_wp
        for _ in range(5):
            nwps = wp.next(3.0)
            if not nwps:
                break
            wp = nwps[0]

        walker_bp = random.choice(bp_lib.filter('walker.pedestrian.*'))
        spawn_tf = wp.transform
        spawn_tf.location += carla.Location(x=0, y=-3, z=0.5)
        walker = world.try_spawn_actor(walker_bp, spawn_tf)
        if walker:
            ctrl_bp = bp_lib.find('controller.ai.walker')
            ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
            world.tick()
            ctrl.start()
            cross_loc = spawn_tf.location + carla.Location(x=0, y=8, z=0)
            ctrl.go_to_location(cross_loc)
            ctrl.set_max_speed(random.uniform(1.5, 2.5))
            spawned_actors.append({'actor': walker, 'type': 'jaywalker', 'controller': ctrl})
            print(f"     Jaywalker crossing road")

    elif scenario_type == 'tight_traffic':
        spawn_points = carla_map.get_spawn_points()
        random.shuffle(spawn_points)
        count = 0
        for sp in spawn_points[:12]:
            if ego_loc.distance(sp.location) < 60:
                vbp = random.choice([bp for bp in bp_lib.filter('vehicle.*')
                                      if int(bp.get_attribute('number_of_wheels')) >= 4])
                npc = world.try_spawn_actor(vbp, sp)
                if npc:
                    spd = random.uniform(2.0, 6.0)
                    fwd = sp.get_forward_vector()
                    npc.enable_constant_velocity(carla.Vector3D(fwd.x * spd, fwd.y * spd, 0))
                    spawned_actors.append({'actor': npc, 'type': 'traffic'})
                    count += 1
        print(f"     Spawned {count} tight-traffic vehicles")

    return spawned_actors, scenario_type


def cleanup_all(world, client, scenario_actors, vehicle_ids, walker_ids, ctrl_ids):
    """Cleanup all spawned actors."""
    for actor_info in scenario_actors:
        try:
            if 'controller' in actor_info:
                actor_info['controller'].stop()
                actor_info['controller'].destroy()
            actor_info['actor'].destroy()
        except Exception:
            pass

    for cid in ctrl_ids:
        try:
            a = world.get_actor(cid)
            if a:
                a.stop()
        except Exception:
            pass

    destroy_ids = vehicle_ids + ctrl_ids + walker_ids
    if destroy_ids:
        client.apply_batch([carla.command.DestroyActor(x) for x in destroy_ids])


# ============================================================================
# Crash Probability Bar (terminal visualization)
# ============================================================================
def probability_bar(prob, width=30):
    """Generate a terminal-friendly probability bar."""
    filled = int(prob * width)
    empty = width - filled

    if prob < 0.3:
        color_start = '\033[92m'  # Green
        emoji = '🟢'
    elif prob < 0.6:
        color_start = '\033[93m'  # Yellow
        emoji = '🟡'
    else:
        color_start = '\033[91m'  # Red
        emoji = '🔴'
    color_end = '\033[0m'

    bar = color_start + '█' * filled + '░' * empty + color_end
    return f"{emoji} [{bar}] {prob:.3f}"


# ============================================================================
# Main Inference Loop
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Real-time crash probability predictor for CARLA 0.9.16')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, help='Path to trained model')
    parser.add_argument('--scaler', default=DEFAULT_SCALER_PATH, help='Path to scaler')
    parser.add_argument('--town', default=DEFAULT_TOWN, help='CARLA town')
    parser.add_argument('--duration', type=int, default=180, help='Duration in seconds')
    parser.add_argument('--npc-vehicles', type=int, default=40, help='Number of NPC vehicles')
    parser.add_argument('--npc-pedestrians', type=int, default=30, help='Number of pedestrians')
    parser.add_argument('--host', default=CARLA_HOST)
    parser.add_argument('--port', type=int, default=CARLA_PORT)
    parser.add_argument('--print-every', type=int, default=5,
                        help='Print prediction every N frames (1=every frame)')
    args = parser.parse_args()

    total_frames = args.duration * FPS

    print("=" * 70)
    print("REAL-TIME CRASH PROBABILITY PREDICTOR")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  Scaler:     {args.scaler}")
    print(f"  Town:       {args.town}")
    print(f"  Duration:   {args.duration}s ({total_frames} frames)")
    print(f"  NPC cars:   {args.npc_vehicles}")
    print(f"  Pedestrians:{args.npc_pedestrians}")
    print(f"  Print freq: every {args.print_every} frames")
    print("=" * 70)

    # ----------------------------------------------------------
    # 1. Load Model
    # ----------------------------------------------------------
    print(f"\n🧠 Loading model from {args.model}...")

    if not os.path.exists(args.model):
        print(f"  ❌ Model file not found: {args.model}")
        print(f"     Run train_mlp.py first to train the model.")
        return

    if not os.path.exists(args.scaler):
        print(f"  ❌ Scaler file not found: {args.scaler}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.model, map_location=device)
    input_dim = checkpoint.get('input_dim', 7)
    hidden_dims = checkpoint.get('hidden_dims', [64, 32, 16])
    dropout = checkpoint.get('dropout', 0.3)

    model = CrashMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  ✅ Model loaded (trained at epoch {checkpoint.get('epoch', '?')})")
    print(f"     Architecture: {input_dim} → {' → '.join(map(str, hidden_dims))} → 1")
    print(f"     Val AUC: {checkpoint.get('val_auc', '?')}")
    print(f"     Device: {device}")

    # Load scaler
    with open(args.scaler, 'rb') as f:
        scaler = pickle.load(f)
    print(f"  ✅ Scaler loaded")
    print(f"     Means: {[f'{m:.2f}' for m in scaler.mean_]}")
    print(f"     Stds:  {[f'{s:.2f}' for s in scaler.scale_]}")

    # ----------------------------------------------------------
    # 2. Connect to CARLA
    # ----------------------------------------------------------
    print(f"\n🔌 Connecting to CARLA at {args.host}:{args.port}...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    world = client.get_world()
    current_map = world.get_map().name
    if args.town not in current_map:
        print(f"  Loading {args.town}...")
        world = client.load_world(args.town)
    else:
        print(f"  Already on {current_map}")

    # Synchronous mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)
    print(f"  ⚙️  Synchronous mode: {FPS} FPS")

    # Traffic Manager
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    # Weather
    weather_preset = random.choice([
        ('Fog + Rain', carla.WeatherParameters(
            cloudiness=80, precipitation=50, fog_density=60, wetness=60,
            sun_altitude_angle=30, fog_distance=0, fog_falloff=0.2)),
        ('Heavy Rain', carla.WeatherParameters(
            cloudiness=90, precipitation=80, fog_density=30, wetness=80,
            sun_altitude_angle=20, fog_distance=0, fog_falloff=0.2)),
        ('Clear', carla.WeatherParameters(
            cloudiness=10, precipitation=0, fog_density=0, wetness=0,
            sun_altitude_angle=60)),
    ])
    world.set_weather(weather_preset[1])
    print(f"  🌤️  Weather: {weather_preset[0]}")

    ego_vehicle = None
    collision_recorder = None
    npc_vehicle_ids = []
    walker_ids = []
    ctrl_ids = []
    scenario_actors = []

    try:
        # ----------------------------------------------------------
        # 3. Spawn Ego Vehicle
        # ----------------------------------------------------------
        bp_lib = world.get_blueprint_library()
        ego_bp = bp_lib.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

        ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
        print(f"\n  🚗 Ego spawned: {ego_vehicle.type_id}")
        print(f"     Position: ({spawn_point.location.x:.0f}, {spawn_point.location.y:.0f})")

        # Aggressive TM autopilot
        tm_port = traffic_manager.get_port()
        ego_vehicle.set_autopilot(True, tm_port)
        traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, -20)  # 20% over limit
        traffic_manager.distance_to_leading_vehicle(ego_vehicle, 3.0)
        traffic_manager.auto_lane_change(ego_vehicle, True)
        try:
            traffic_manager.ignore_lights_percentage(ego_vehicle, 10)
        except AttributeError:
            pass

        print(f"  ✅ TM autopilot: speed_diff=-20%, follow_dist=3.0m, ignore_lights=10%")

        # Collision sensor
        collision_recorder = CollisionRecorder(ego_vehicle, world)

        # Let it settle
        for _ in range(20):
            world.tick()

        # ----------------------------------------------------------
        # 4. Spawn Traffic
        # ----------------------------------------------------------
        print(f"\n  Spawning traffic...")
        npc_vehicle_ids = spawn_npc_vehicles(
            world, client, traffic_manager, args.npc_vehicles)
        walker_ids, ctrl_ids = spawn_pedestrians(
            world, client, args.npc_pedestrians)

        for _ in range(10):
            world.tick()

        # Spawn challenging scenario
        scenario_actors, scenario_type = spawn_challenging_scenario(world, ego_vehicle)

        # ----------------------------------------------------------
        # 5. Real-time Prediction Loop
        # ----------------------------------------------------------
        print(f"\n{'=' * 70}")
        print(f"  🏁 LIVE PREDICTION STARTED — {args.duration}s ({total_frames} frames)")
        print(f"  Printing every {args.print_every} frames")
        print(f"  Press Ctrl+C to stop")
        print(f"{'=' * 70}")
        print()
        print(f"  {'Frame':>7} │ {'Speed':>7} │ {'Accel':>7} │ {'Dist':>7} │ {'RelV':>7} │ "
              f"{'TTC':>6} │ {'ObsSpd':>6} │ {'Obs':>4} │ {'CRASH PROBABILITY':>30}")
        print(f"  {'─' * 7}─┼─{'─' * 7}─┼─{'─' * 7}─┼─{'─' * 7}─┼─{'─' * 7}─┼─"
              f"{'─' * 6}─┼─{'─' * 6}─┼─{'─' * 4}─┼─{'─' * 30}")

        prev_speed = 0.0
        start_time = time.time()

        # Stats tracking
        all_predictions = []
        high_prob_count = 0
        frames_before_collision = []  # predictions just before each collision
        last_collision_count = 0

        for frame_idx in range(total_frames):
            collision_recorder.frame_counter[0] = frame_idx

            try:
                world.tick()
            except RuntimeError as e:
                print(f"\n  ⚠️  world.tick() failed: {e}")
                break

            # Get ego state
            ego_velocity = ego_vehicle.get_velocity()
            ego_speed = math.sqrt(
                ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)
            ego_acceleration = (ego_speed - prev_speed) * FPS
            prev_speed = ego_speed

            # Find nearest obstacle
            nearest = find_nearest_obstacle(ego_vehicle, world)

            # Compute TTC
            if nearest['relative_velocity'] > 0.1:
                ttc = nearest['distance'] / nearest['relative_velocity']
                ttc = min(ttc, 10.0)
            else:
                ttc = 10.0

            # Build feature vector (same order as FEATURE_COLUMNS)
            raw_features = np.array([[
                ego_speed,
                np.clip(ego_acceleration, -20, 20),
                nearest['distance'],
                np.clip(nearest['relative_velocity'], -20, 20),
                ttc,
                nearest['obstacle_speed'],
                nearest['obstacle_type']
            ]], dtype=np.float32)

            # Normalize with scaler
            scaled_features = scaler.transform(raw_features)

            # Predict crash probability
            with torch.no_grad():
                features_tensor = torch.FloatTensor(scaled_features).to(device)
                crash_prob = model(features_tensor).item()

            all_predictions.append(crash_prob)

            if crash_prob >= 0.6:
                high_prob_count += 1

            # Check if a new collision just happened
            current_collision_count = len(collision_recorder.collisions)
            if current_collision_count > last_collision_count:
                new_collision = collision_recorder.collisions[-1]
                # What was our prediction just before this collision?
                recent_preds = all_predictions[-min(10, len(all_predictions)):]
                avg_recent = np.mean(recent_preds) if recent_preds else 0.0
                max_recent = max(recent_preds) if recent_preds else 0.0

                print(f"\n  {'─' * 90}")
                print(f"  💥💥💥 COLLISION OCCURRED! 💥💥💥")
                print(f"  Hit:     {new_collision['other_actor']}")
                print(f"  Impulse: {new_collision['impulse']:.1f} N·s")
                print(f"  Crash probability at impact: {crash_prob:.3f}")
                print(f"  Avg prediction (last 10 frames): {avg_recent:.3f}")
                print(f"  Max prediction (last 10 frames): {max_recent:.3f}")

                if max_recent >= 0.5:
                    print(f"  ✅ SYSTEM DETECTED THIS COLLISION (max pred >= 0.5)")
                else:
                    print(f"  ❌ SYSTEM MISSED THIS COLLISION (max pred < 0.5)")

                frames_before_collision.append({
                    'frame': frame_idx,
                    'crash_prob_at_impact': crash_prob,
                    'avg_last_10': avg_recent,
                    'max_last_10': max_recent,
                    'detected': max_recent >= 0.5,
                    'hit': new_collision['other_actor'],
                })
                print(f"  {'─' * 90}\n")
                last_collision_count = current_collision_count

            # Handle sudden brake triggers
            current_time = time.time()
            for actor_info in scenario_actors:
                if actor_info.get('type') == 'sudden_braker' and not actor_info.get('braked', False):
                    if current_time >= actor_info.get('brake_time', float('inf')):
                        print(f"\n  🛑 LEAD VEHICLE BRAKING HARD!\n")
                        try:
                            actor_info['actor'].enable_constant_velocity(carla.Vector3D(0, 0, 0))
                            actor_info['actor'].apply_control(
                                carla.VehicleControl(throttle=0.0, brake=1.0))
                        except Exception:
                            pass
                        actor_info['braked'] = True

            # Update spectator
            spectator = world.get_spectator()
            tf = ego_vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                tf.location - tf.get_forward_vector() * 12 + carla.Location(z=6),
                carla.Rotation(pitch=-20, yaw=tf.rotation.yaw)
            ))

            # Print prediction
            if frame_idx % args.print_every == 0:
                obs_type_str = ['VEH', 'PED', '---'][nearest['obstacle_type']]
                prob_bar = probability_bar(crash_prob)

                elapsed = time.time() - start_time
                line = (f"  {frame_idx:7d} │ {ego_speed:5.1f}m/s │ {ego_acceleration:+5.1f}m/s² │ "
                        f"{nearest['distance']:5.1f}m │ {nearest['relative_velocity']:+5.1f}m/s │ "
                        f"{ttc:5.1f}s │ {nearest['obstacle_speed']:5.1f} │ {obs_type_str:>4} │ "
                        f"{prob_bar}")
                print(line)

            # Print periodic summary every 200 frames (10 seconds)
            if frame_idx > 0 and frame_idx % 200 == 0:
                elapsed = time.time() - start_time
                recent_100 = all_predictions[-100:]
                avg_prob = np.mean(recent_100)
                max_prob = max(recent_100)
                min_prob = min(recent_100)
                total_collisions = len(collision_recorder.collisions)

                print(f"\n  ╔══════════════════════════════════════════════════════════════")
                print(f"  ║ {elapsed:.0f}s SUMMARY — Frame {frame_idx}/{total_frames}")
                print(f"  ║ Avg crash prob (last 100): {avg_prob:.3f}")
                print(f"  ║ Min/Max prob (last 100):   {min_prob:.3f} / {max_prob:.3f}")
                print(f"  ║ High probability frames:   {high_prob_count} ({high_prob_count/max(1,frame_idx)*100:.1f}%)")
                print(f"  ║ Total collisions:          {total_collisions}")
                print(f"  ╚══════════════════════════════════════════════════════════════\n")

        # ----------------------------------------------------------
        # 6. Final Report
        # ----------------------------------------------------------
        total_time = time.time() - start_time
        total_collisions = len(collision_recorder.collisions)

        print(f"\n{'=' * 70}")
        print(f"PREDICTION SESSION COMPLETE")
        print(f"{'=' * 70}")
        print(f"\n  ⏱️  Duration: {total_time:.1f}s ({len(all_predictions)} frames)")
        print(f"  🚗 Total collisions: {total_collisions}")

        if len(all_predictions) > 0:
            preds_arr = np.array(all_predictions)
            print(f"\n  📊 Prediction Statistics:")
            print(f"     Mean crash probability:   {preds_arr.mean():.4f}")
            print(f"     Std crash probability:    {preds_arr.std():.4f}")
            print(f"     Min:                      {preds_arr.min():.4f}")
            print(f"     Max:                      {preds_arr.max():.4f}")
            print(f"     Median:                   {np.median(preds_arr):.4f}")

            print(f"\n  📊 Prediction Distribution:")
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                count = (preds_arr >= threshold).sum()
                print(f"     P >= {threshold:.1f}: {count:7d} frames ({count/len(preds_arr)*100:.1f}%)")

        if frames_before_collision:
            detected = sum(1 for f in frames_before_collision if f['detected'])
            total = len(frames_before_collision)
            print(f"\n  📊 Collision Detection Performance:")
            print(f"     Total collisions:      {total}")
            print(f"     Correctly warned:      {detected} ({detected/max(1,total)*100:.1f}%)")
            print(f"     Missed:                {total - detected}")

            print(f"\n  📊 Per-collision Details:")
            for i, fc in enumerate(frames_before_collision):
                status = "✅ DETECTED" if fc['detected'] else "❌ MISSED"
                print(f"     {i+1}. Frame {fc['frame']}: P={fc['crash_prob_at_impact']:.3f} "
                      f"(avg10={fc['avg_last_10']:.3f}, max10={fc['max_last_10']:.3f}) "
                      f"→ {fc['hit']} — {status}")

        print(f"\n{'=' * 70}")

    except KeyboardInterrupt:
        print(f"\n\n  ⚠️  Interrupted by user")

    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        traceback.print_exc()

    finally:
        print(f"\n🧹 Cleaning up...")

        if collision_recorder:
            try:
                collision_recorder.cleanup()
            except Exception:
                pass

        cleanup_all(world, client, scenario_actors, npc_vehicle_ids, walker_ids, ctrl_ids)

        if ego_vehicle:
            try:
                ego_vehicle.destroy()
            except Exception:
                pass

        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        print("  ✅ Cleanup complete")


if __name__ == '__main__':
    main()
