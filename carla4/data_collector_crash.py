#!/usr/bin/env python3
"""
Crash Probability Data Collector for CARLA 0.9.16
===================================================

Collects kinematic features and collision labels for MLP training.

Ego vehicle uses CARLA Traffic Manager autopilot. Crashes happen
NATURALLY from dense, chaotic traffic with aggressive NPC behavior —
no scripted sudden-brake scenarios.

Features collected per frame:
  - ego_speed           (m/s)
  - ego_acceleration    (m/s²)
  - nearest_distance    (m, to closest obstacle ahead in lane)
  - relative_velocity   (m/s, positive = closing in)
  - ttc                 (seconds, time-to-collision, capped at 10s)
  - obstacle_speed      (m/s)
  - obstacle_type       (0=vehicle, 1=pedestrian, 2=none)
  - lateral_offset      (m)
  - ego_steering        (current steering angle)

Label (applied retroactively):
  - collision_within_2s : 1 if a collision happens within next 2 seconds

Usage:
    python data_collector_crash.py
    python data_collector_crash.py --scenarios 80 --duration 60
    python data_collector_crash.py --scenarios 100 --duration 60 --town Town03

Press Ctrl+C to stop collection early.
"""

import carla
import numpy as np
import os
import csv
import math
import time
import random
import argparse
import traceback

# ============================================================================
# Configuration
# ============================================================================
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
DEFAULT_TOWN = 'Town01'

# Map — Town01 only (good intersection layout with traffic lights)
MAP_POOL = ['Town01']
MAP_CHANGE_INTERVAL = 999  # Never change map
FPS = 20  # Simulation tick rate
SAVE_DIR = 'dataset_crash'

# Obstacle detection parameters
MAX_SEARCH_DISTANCE = 50.0   # meters — max range to look for obstacles
LANE_HALF_WIDTH = 4.0        # meters — lateral search width

# Label parameters
LOOKAHEAD_SECONDS = 2.0
LOOKAHEAD_FRAMES = int(LOOKAHEAD_SECONDS * FPS)  # 40 frames at 20 FPS

# Stuck recovery
STUCK_SPEED_THRESHOLD = 0.5    # m/s — below this counts as stuck
STUCK_TELEPORT_SECONDS = 60    # After 60s stuck, teleport ego to new location

# Weather presets
WEATHER_PRESETS = [
    {'name': 'Rainy',        'cloudiness': 80,  'precipitation': 60,  'fog_density': 30,  'wetness': 60,  'sun_altitude': 35},
    {'name': 'Heavy Rain',   'cloudiness': 95,  'precipitation': 100, 'fog_density': 50,  'wetness': 100, 'sun_altitude': 15},
    {'name': 'Dense Fog',    'cloudiness': 70,  'precipitation': 0,   'fog_density': 100, 'wetness': 30,  'sun_altitude': 40},
    {'name': 'Fog + Rain',   'cloudiness': 90,  'precipitation': 70,  'fog_density': 80,  'wetness': 80,  'sun_altitude': 25},
    {'name': 'Night Fog',    'cloudiness': 50,  'precipitation': 0,   'fog_density': 90,  'wetness': 20,  'sun_altitude': -30},
    {'name': 'Night Storm',  'cloudiness': 100, 'precipitation': 100, 'fog_density': 70,  'wetness': 100, 'sun_altitude': -20},
    {'name': 'Whiteout',     'cloudiness': 100, 'precipitation': 100, 'fog_density': 100, 'wetness': 100, 'sun_altitude': 10},
]

# ── Ego driving profiles ────────────────────────────────────────────────
# CRASH STRATEGY:
# - High ignore_lights (50-80%) → ego frequently runs red lights
#   → gets T-boned by cross-traffic at intersections
# - Moderate ignore_vehicles (8-15%) → ego occasionally misjudges
#   and rear-ends slower vehicles (but not constantly)
# - Fast speeds with tight follow distances amplify crash likelihood
# - NPCs drive NORMALLY so crashes look natural
DRIVING_PROFILES = [
    {'name': 'Reckless',      'speed_diff': -40, 'follow_dist': 1.5, 'ignore_lights': 80, 'ignore_vehicles': 15, 'lane_change': True, 'weight': 25},
    {'name': 'Aggressive',    'speed_diff': -30, 'follow_dist': 2.0, 'ignore_lights': 60, 'ignore_vehicles': 12, 'lane_change': True, 'weight': 30},
    {'name': 'Fast',          'speed_diff': -20, 'follow_dist': 2.5, 'ignore_lights': 50, 'ignore_vehicles': 8,  'lane_change': True, 'weight': 25},
    {'name': 'Normal',        'speed_diff':   0, 'follow_dist': 3.0, 'ignore_lights': 30, 'ignore_vehicles': 5,  'lane_change': True, 'weight': 15},
    {'name': 'Cautious',      'speed_diff':  10, 'follow_dist': 5.0, 'ignore_lights': 10, 'ignore_vehicles': 0,  'lane_change': True, 'weight': 5},
]
_PROFILE_WEIGHTS = [p['weight'] for p in DRIVING_PROFILES]

# ── NPC driving profiles ────────────────────────────────────────────────
# NPCs drive NORMALLY — no ignore_vehicles, no ignore_lights.
# They provide the cross-traffic that T-bones the ego at red lights
# and the slower vehicles that the ego occasionally rear-ends.
NPC_PROFILES = [
    {'name': 'Fast NPC',       'speed_diff': -15, 'follow_dist': 2.5, 'ignore_lights': 0, 'ignore_vehicles': 0, 'weight': 25},
    {'name': 'Normal NPC',     'speed_diff':   0, 'follow_dist': 3.5, 'ignore_lights': 0, 'ignore_vehicles': 0, 'weight': 50},
    {'name': 'Slow NPC',       'speed_diff':  15, 'follow_dist': 5.0, 'ignore_lights': 0, 'ignore_vehicles': 0, 'weight': 25},
]
_NPC_PROFILE_WEIGHTS = [p['weight'] for p in NPC_PROFILES]


# ============================================================================
# Radar Sensor Recorder
# ============================================================================
class RadarRecorder:
    """
    Processes radar detections into obstacle features.

    CARLA radar returns a list of detections, each with:
      - depth: distance in meters
      - velocity: radial velocity (negative = approaching)
      - azimuth: horizontal angle in radians
      - altitude: vertical angle in radians

    We extract the nearest detection in forward cone and compute:
      distance, relative_velocity, obstacle_speed, lateral_offset
    """
    def __init__(self, vehicle, world, range_m=50.0, fov_h=30, fov_v=10, pps=1500):
        self.latest_data = {
            'distance': range_m,
            'relative_velocity': 0.0,
            'obstacle_speed': 0.0,
            'obstacle_type': 2,  # 2=none, radar can't distinguish type
            'lateral_offset': 0.0,
            'num_detections': 0,
        }
        self._ego_speed = 0.0  # Updated each frame from main loop

        # Spawn radar sensor
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(fov_h))
        bp.set_attribute('vertical_fov', str(fov_v))
        bp.set_attribute('range', str(range_m))
        bp.set_attribute('points_per_second', str(pps))

        # Mount radar on front bumper
        radar_tf = carla.Transform(
            carla.Location(x=2.5, z=0.7),  # front of car, bumper height
            carla.Rotation(pitch=0)
        )
        self.sensor = world.spawn_actor(bp, radar_tf, attach_to=vehicle)
        self.sensor.listen(self._on_radar)
        self._range = range_m
        print(f"  ✅ Radar sensor (range={range_m}m, fov={fov_h}°, {pps} pts/s)")

    def update_ego_speed(self, speed):
        """Call each frame so we can compute obstacle_speed."""
        self._ego_speed = speed

    def _on_radar(self, radar_data):
        """Process radar detections, find nearest obstacle in forward cone."""
        detections = radar_data  # iterable of carla.RadarDetection

        nearest_dist = self._range
        nearest_vel = 0.0
        nearest_azimuth = 0.0
        count = 0

        for det in detections:
            count += 1
            # det.depth = distance, det.velocity = radial velocity
            # det.azimuth = horizontal angle (rad), det.altitude = vertical angle (rad)

            # Filter: only consider detections within narrow forward cone
            if abs(det.azimuth) > 0.3:  # ~17 degrees each side
                continue
            if det.depth < 1.0:  # too close, likely self-detection
                continue

            if det.depth < nearest_dist:
                nearest_dist = det.depth
                nearest_vel = det.velocity  # negative = approaching
                nearest_azimuth = det.azimuth

        # Convert to our feature format
        # CARLA radar velocity: negative = approaching ego
        relative_velocity = -nearest_vel  # flip sign: positive = closing in

        # Approximate obstacle speed from relative velocity + ego speed
        obstacle_speed = max(0, self._ego_speed - relative_velocity)

        # Lateral offset from azimuth: lateral = distance * sin(azimuth)
        lateral_offset = nearest_dist * math.sin(nearest_azimuth) if nearest_dist < self._range else 0.0

        self.latest_data = {
            'distance': nearest_dist,
            'relative_velocity': relative_velocity,
            'obstacle_speed': obstacle_speed,
            'obstacle_type': 0 if nearest_dist < self._range else 2,  # 0=detected, 2=none
            'lateral_offset': lateral_offset,
            'num_detections': count,
        }

    def get_nearest(self):
        """Return latest processed radar data (same format as old find_nearest_obstacle)."""
        return self.latest_data.copy()

    def cleanup(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


# ============================================================================
# Collision Recorder
# ============================================================================
class CollisionRecorder:
    """
    Records collision events with frame indexing for retroactive labeling.

    Deduplication: ignores repeat collisions with the same actor within
    a cooldown window to prevent spam (CARLA fires collision events every
    frame while two actors remain in contact).
    """
    COOLDOWN_SECONDS = 5.0     # Min interval between collisions with same actor
    MIN_IMPULSE = 300.0        # Minimum impulse (N·s) to count as real crash

    def __init__(self, vehicle, world):
        self.collision_frame_indices = []  # Deduplicated frame indices
        self.collision_details = []        # Deduplicated details
        self.frame_counter = [0]           # Mutable ref updated by main loop
        self._last_collision_time = {}     # actor_id -> last collision timestamp
        self._raw_collision_count = 0      # Total raw (unfiltered) events
        self._skipped_low_impulse = 0      # Skipped due to low impulse

        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        self.sensor.listen(self._on_collision)
        print(f"  ✅ Collision sensor (cooldown={self.COOLDOWN_SECONDS}s, min_impulse={self.MIN_IMPULSE}N·s)")

    def _on_collision(self, event):
        self._raw_collision_count += 1
        now = time.time()
        actor_id = event.other_actor.id
        actor_type = event.other_actor.type_id
        impulse = event.normal_impulse.length()

        # FILTER 1: Only vehicles and pedestrians (skip fences, signs, walls)
        if not (actor_type.startswith('vehicle.') or actor_type.startswith('walker.')):
            return

        # FILTER 2: Minimum impulse — skip gentle bumps from stopped cars touching
        if impulse < self.MIN_IMPULSE:
            self._skipped_low_impulse += 1
            return

        # FILTER 3: Cooldown — skip if same actor within cooldown period
        if actor_id in self._last_collision_time:
            elapsed = now - self._last_collision_time[actor_id]
            if elapsed < self.COOLDOWN_SECONDS:
                return

        self._last_collision_time[actor_id] = now
        frame_idx = self.frame_counter[0]
        self.collision_frame_indices.append(frame_idx)

        detail = {
            'frame_idx': frame_idx,
            'other_actor': actor_type,
            'impulse': impulse,
        }
        self.collision_details.append(detail)

        print(f"\n  💥 COLLISION at frame {frame_idx}!")
        print(f"     Hit: {actor_type}")
        print(f"     Impulse: {impulse:.1f} N·s")
        print(f"     (raw: {self._raw_collision_count}, real: {len(self.collision_frame_indices)}, "
              f"skipped_weak: {self._skipped_low_impulse})")

    def cleanup(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


# ============================================================================
# Retroactive Label Application
# ============================================================================
def apply_collision_labels(scenario_data, collision_frame_indices, lookahead_frames):
    """
    Retroactively label frames before each collision as positive.

    For each collision at frame C, frames [C - lookahead_frames, C] get label=1.
    All other frames get label=0.
    """
    # Initialize all as 0
    for row in scenario_data:
        row['collision_within_2s'] = 0

    # Mark frames preceding each collision
    for cf in collision_frame_indices:
        start = max(0, cf - lookahead_frames)
        end = min(len(scenario_data), cf + 1)
        for i in range(start, end):
            scenario_data[i]['collision_within_2s'] = 1

    pos = sum(1 for r in scenario_data if r['collision_within_2s'] == 1)
    neg = len(scenario_data) - pos

    print(f"  📊 Labels applied: {pos} positive ({pos/max(1,len(scenario_data))*100:.1f}%), {neg} negative")

    return scenario_data


# ============================================================================
# Traffic Spawning — with aggressive NPC profiles
# ============================================================================
def spawn_npc_vehicles(world, client, traffic_manager, count=120):
    """
    Spawn NPC traffic vehicles with varied but realistic driving profiles.
    Tries ALL available spawn points to maximize vehicle count.
    """
    bp_lib = world.get_blueprint_library()
    vehicle_bps = [bp for bp in bp_lib.filter('vehicle.*')
                   if int(bp.get_attribute('number_of_wheels')) >= 4]
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    # Use ALL spawn points if count exceeds available
    num_to_spawn = min(count, len(spawn_points))

    batch = []
    for i in range(num_to_spawn):
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

    # Assign varied driving profiles to NPCs
    tm_port = traffic_manager.get_port()
    profile_counts = {}
    for vid in vehicle_ids:
        vehicle = world.get_actor(vid)
        if vehicle:
            vehicle.set_autopilot(True, tm_port)

            # Pick a random NPC profile (weighted)
            profile = random.choices(NPC_PROFILES, weights=_NPC_PROFILE_WEIGHTS, k=1)[0]
            pname = profile['name']
            profile_counts[pname] = profile_counts.get(pname, 0) + 1

            traffic_manager.vehicle_percentage_speed_difference(vehicle, profile['speed_diff'])
            traffic_manager.distance_to_leading_vehicle(vehicle, profile['follow_dist'])
            traffic_manager.auto_lane_change(vehicle, True)
            try:
                traffic_manager.ignore_lights_percentage(vehicle, profile['ignore_lights'])
            except AttributeError:
                pass
            try:
                traffic_manager.ignore_vehicles_percentage(vehicle, profile['ignore_vehicles'])
            except AttributeError:
                pass

    print(f"  🚗 Spawned {len(vehicle_ids)}/{count} NPC vehicles")
    for pname, cnt in sorted(profile_counts.items()):
        print(f"     {pname}: {cnt}")
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


# ============================================================================
# Spawn Point Selection — near intersections for more action
# ============================================================================
def pick_interesting_spawn(world):
    """
    Pick a spawn point that is near a junction/intersection.
    Junctions = where vehicles cross paths = more natural crashes.
    Falls back to random if no good junction spawn found.
    """
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    random.shuffle(spawn_points)

    # Prefer spawn points near junctions (50% of the time)
    if random.random() < 0.5:
        junction_spawns = []
        for sp in spawn_points:
            wp = carla_map.get_waypoint(sp.location, project_to_road=True,
                                         lane_type=carla.LaneType.Driving)
            if wp:
                # Check if within ~30m of a junction
                ahead = wp
                for _ in range(10):
                    nxt = ahead.next(3.0)
                    if not nxt:
                        break
                    ahead = nxt[0]
                    if ahead.is_junction:
                        junction_spawns.append(sp)
                        break

        if junction_spawns:
            sp = random.choice(junction_spawns)
            print(f"     (near junction)")
            return sp

    return random.choice(spawn_points)


# ============================================================================
# Cleanup Helpers
# ============================================================================
def cleanup_traffic(world, client, vehicle_ids, walker_ids, ctrl_ids):
    """Destroy all NPC traffic."""
    # Stop walker controllers first
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
# CSV Writer
# ============================================================================
CSV_COLUMNS = [
    'frame_id', 'scenario_id', 'timestamp',
    'ego_speed', 'ego_acceleration', 'nearest_distance',
    'relative_velocity', 'ttc', 'obstacle_speed', 'obstacle_type',
    'lateral_offset', 'ego_steering', 'collision_within_2s'
]


def write_scenario_to_csv(csv_path, scenario_data, write_header=False):
    """Append scenario data rows to CSV file."""
    mode = 'w' if write_header else 'a'
    with open(csv_path, mode=mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in scenario_data:
            writer.writerow(row)
    print(f"  💾 Wrote {len(scenario_data)} rows to {csv_path}")


# ============================================================================
# Main Collection Loop
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Crash probability data collector for CARLA 0.9.16')
    parser.add_argument('--town', default=DEFAULT_TOWN, help='CARLA town to use')
    parser.add_argument('--scenarios', type=int, default=80, help='Number of scenarios')
    parser.add_argument('--duration', type=int, default=90, help='Duration per scenario in seconds')
    parser.add_argument('--npc-vehicles', type=int, default=120, help='Number of NPC vehicles (dense)')
    parser.add_argument('--npc-pedestrians', type=int, default=60, help='Number of pedestrians')
    parser.add_argument('--host', default=CARLA_HOST)
    parser.add_argument('--port', type=int, default=CARLA_PORT)
    args = parser.parse_args()

    frames_per_scenario = args.duration * FPS

    print("=" * 70)
    print("CRASH PROBABILITY DATA COLLECTOR (Natural Crashes)")
    print("=" * 70)
    print(f"  CARLA Server:     {args.host}:{args.port}")
    print(f"  Town:             {args.town}")
    print(f"  Scenarios:        {args.scenarios}")
    print(f"  Duration/scenario:{args.duration}s ({frames_per_scenario} frames)")
    print(f"  Total frames:     ~{args.scenarios * frames_per_scenario:,}")
    print(f"  NPC vehicles:     {args.npc_vehicles}")
    print(f"  NPC pedestrians:  {args.npc_pedestrians}")
    print(f"  Lookahead:        {LOOKAHEAD_SECONDS}s ({LOOKAHEAD_FRAMES} frames)")
    print(f"  Save directory:   {SAVE_DIR}")
    print(f"  Strategy:         Aggressive NPCs + dense traffic + junctions")
    print("=" * 70)

    # Create output directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    csv_path = os.path.join(SAVE_DIR, 'data.csv')

    # Connect to CARLA
    print(f"\n🔌 Connecting to CARLA at {args.host}:{args.port}...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    # Load initial map
    world = client.get_world()
    current_map_name = world.get_map().name.split('/')[-1]
    initial_town = random.choice(MAP_POOL)
    if initial_town not in current_map_name:
        print(f"  Loading {initial_town}...")
        world = client.load_world(initial_town)
        current_map_name = initial_town
    else:
        print(f"  Already on {current_map_name}")

    # Synchronous mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)
    print(f"  ⚙️  Synchronous mode: {FPS} FPS (dt={1.0/FPS:.3f}s)")

    # Traffic Manager
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    # Global TM settings — make overall traffic more chaotic
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
    traffic_manager.set_random_device_seed(random.randint(0, 10000))
    print(f"  ⚙️  Traffic Manager on port 8000")

    # Global stats
    global_frame_count = 0
    global_collision_count = 0
    global_positive_labels = 0

    ego_vehicle = None
    collision_recorder = None
    radar_recorder = None
    npc_vehicle_ids = []
    walker_ids = []
    ctrl_ids = []
    scenario_idx = 0

    try:
        for scenario_idx in range(args.scenarios):
            print(f"\n{'=' * 70}")
            print(f"SCENARIO {scenario_idx + 1}/{args.scenarios}")
            print(f"{'=' * 70}")

            # ----------------------------------------------------------
            # Map rotation
            # ----------------------------------------------------------
            if scenario_idx > 0 and scenario_idx % MAP_CHANGE_INTERVAL == 0:
                new_town = random.choice(MAP_POOL)
                if new_town not in current_map_name:
                    print(f"\n  🗺️  Switching map: {current_map_name} → {new_town}")
                    settings = world.get_settings()
                    settings.synchronous_mode = False
                    world.apply_settings(settings)
                    world = client.load_world(new_town)
                    current_map_name = new_town
                    settings = world.get_settings()
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = 1.0 / FPS
                    world.apply_settings(settings)
                    traffic_manager = client.get_trafficmanager(8000)
                    traffic_manager.set_synchronous_mode(True)
                    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
                    print(f"  ✅ Loaded {new_town}")

            print(f"  🗺️  Map: {current_map_name}")

            # ----------------------------------------------------------
            # Weather & driving profile
            # ----------------------------------------------------------
            weather_preset = random.choice(WEATHER_PRESETS)
            driving_profile = random.choices(DRIVING_PROFILES, weights=_PROFILE_WEIGHTS, k=1)[0]

            weather = carla.WeatherParameters(
                cloudiness=weather_preset['cloudiness'],
                precipitation=weather_preset['precipitation'],
                precipitation_deposits=0.0,
                wind_intensity=random.uniform(0, 40),
                sun_altitude_angle=weather_preset['sun_altitude'],
                fog_density=weather_preset['fog_density'],
                fog_distance=0.0,
                fog_falloff=0.2,
                wetness=weather_preset['wetness'],
            )
            world.set_weather(weather)

            print(f"  🌤️  Weather: {weather_preset['name']}")
            print(f"     cloud={weather_preset['cloudiness']}  rain={weather_preset['precipitation']}  "
                  f"fog={weather_preset['fog_density']}  wet={weather_preset['wetness']}  "
                  f"sun_alt={weather_preset['sun_altitude']}")
            print(f"  🏎️  Ego profile: {driving_profile['name']}")
            print(f"     speed_diff={driving_profile['speed_diff']}%  "
                  f"follow_dist={driving_profile['follow_dist']}m  "
                  f"ignore_lights={driving_profile['ignore_lights']}%  "
                  f"ignore_vehicles={driving_profile['ignore_vehicles']}%")

            # ----------------------------------------------------------
            # Spawn Ego Vehicle (near junctions when possible)
            # ----------------------------------------------------------
            bp_lib = world.get_blueprint_library()
            ego_bp = bp_lib.find('vehicle.tesla.model3')
            spawn_point = pick_interesting_spawn(world)

            ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
            print(f"\n  🚗 Ego spawned: {ego_vehicle.type_id}")
            print(f"     Position: ({spawn_point.location.x:.0f}, {spawn_point.location.y:.0f}, {spawn_point.location.z:.0f})")

            # Apply TM autopilot with driving profile
            tm_port = traffic_manager.get_port()
            ego_vehicle.set_autopilot(True, tm_port)
            traffic_manager.vehicle_percentage_speed_difference(
                ego_vehicle, driving_profile['speed_diff'])
            traffic_manager.distance_to_leading_vehicle(
                ego_vehicle, driving_profile['follow_dist'])
            traffic_manager.auto_lane_change(
                ego_vehicle, driving_profile['lane_change'])

            try:
                traffic_manager.ignore_lights_percentage(
                    ego_vehicle, driving_profile['ignore_lights'])
            except AttributeError:
                pass
            try:
                traffic_manager.ignore_signs_percentage(
                    ego_vehicle, driving_profile.get('ignore_signs', 10))
            except AttributeError:
                pass
            try:
                traffic_manager.ignore_vehicles_percentage(
                    ego_vehicle, driving_profile.get('ignore_vehicles', 0))
            except AttributeError:
                pass

            print(f"  ✅ TM autopilot configured")

            # ----------------------------------------------------------
            # Sensors: Collision + Radar
            # ----------------------------------------------------------
            collision_recorder = CollisionRecorder(ego_vehicle, world)
            radar_recorder = RadarRecorder(ego_vehicle, world,
                                           range_m=MAX_SEARCH_DISTANCE,
                                           fov_h=30, fov_v=10, pps=1500)

            # Let vehicle settle
            for _ in range(20):
                world.tick()

            # ----------------------------------------------------------
            # Spawn Traffic (with aggressive NPC profiles)
            # ----------------------------------------------------------
            print(f"\n  Spawning traffic...")
            npc_vehicle_ids = spawn_npc_vehicles(
                world, client, traffic_manager, args.npc_vehicles)
            walker_ids, ctrl_ids = spawn_pedestrians(
                world, client, args.npc_pedestrians)

            # Let traffic settle and start moving
            for _ in range(30):
                world.tick()

            # ----------------------------------------------------------
            # Data Collection Loop
            # ----------------------------------------------------------
            print(f"\n  🏁 Starting collection (runs until crash)")
            print(f"  {'─' * 60}")

            scenario_data = []
            prev_speed = 0.0
            stuck_counter = 0
            start_time = time.time()
            post_collision_frames = -1  # -1 = no collision yet
            POST_COLLISION_COLLECT = FPS * 3  # collect 3s after collision then end
            frame_idx = 0

            while True:
                # Update collision recorder's frame counter
                collision_recorder.frame_counter[0] = frame_idx

                # If we already had a collision, count down post-crash frames
                if post_collision_frames >= 0:
                    post_collision_frames += 1
                    if post_collision_frames >= POST_COLLISION_COLLECT:
                        print(f"\n  🔄 Collision occurred — ending scenario after {POST_COLLISION_COLLECT//FPS}s post-crash data")
                        break

                # Check if a new collision just happened
                if len(collision_recorder.collision_frame_indices) > 0 and post_collision_frames < 0:
                    post_collision_frames = 0

                # Tick the world
                try:
                    world.tick()
                except RuntimeError as e:
                    print(f"  ⚠️  world.tick() failed: {e}")
                    break

                # Get ego state
                try:
                    ego_velocity = ego_vehicle.get_velocity()
                    ego_speed = math.sqrt(
                        ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)
                    ego_acceleration = (ego_speed - prev_speed) * FPS
                    prev_speed = ego_speed

                    ego_control = ego_vehicle.get_control()
                    ego_location = ego_vehicle.get_location()
                except RuntimeError:
                    print(f"  ⚠️  Ego vehicle lost, ending scenario")
                    break

                # --- Stuck recovery: teleport to new location ---
                if ego_speed < STUCK_SPEED_THRESHOLD:
                    stuck_counter += 1

                    if stuck_counter >= FPS * STUCK_TELEPORT_SECONDS:
                        print(f"\n  ⚠️  Ego stuck ({stuck_counter // FPS}s) → teleporting to new location")
                        # Pick a new random spawn point far from current position
                        spawn_points = world.get_map().get_spawn_points()
                        random.shuffle(spawn_points)
                        new_sp = None
                        for sp in spawn_points:
                            if ego_location.distance(sp.location) > 50:
                                new_sp = sp
                                break
                        if new_sp is None:
                            new_sp = random.choice(spawn_points)

                        ego_vehicle.set_transform(new_sp)
                        # Brief re-settle
                        for _ in range(5):
                            world.tick()
                        stuck_counter = 0
                        print(f"     Teleported to ({new_sp.location.x:.0f}, {new_sp.location.y:.0f})")
                else:
                    stuck_counter = 0

                # Find nearest obstacle via RADAR sensor
                radar_recorder.update_ego_speed(ego_speed)
                nearest = radar_recorder.get_nearest()

                # Compute TTC
                if nearest['relative_velocity'] > 0.1:
                    ttc = nearest['distance'] / nearest['relative_velocity']
                    ttc = min(ttc, 10.0)
                else:
                    ttc = 10.0

                # Build frame data
                frame_data = {
                    'frame_id': global_frame_count + frame_idx,
                    'scenario_id': scenario_idx,
                    'timestamp': round(frame_idx / FPS, 3),
                    'ego_speed': round(ego_speed, 3),
                    'ego_acceleration': round(ego_acceleration, 3),
                    'nearest_distance': round(nearest['distance'], 3),
                    'relative_velocity': round(nearest['relative_velocity'], 3),
                    'ttc': round(ttc, 3),
                    'obstacle_speed': round(nearest['obstacle_speed'], 3),
                    'obstacle_type': nearest['obstacle_type'],
                    'lateral_offset': round(nearest['lateral_offset'], 3),
                    'ego_steering': round(ego_control.steer, 4),
                    'collision_within_2s': 0,
                }
                scenario_data.append(frame_data)

                # Update spectator to follow ego
                spectator = world.get_spectator()
                tf = ego_vehicle.get_transform()
                spectator.set_transform(carla.Transform(
                    tf.location - tf.get_forward_vector() * 12 + carla.Location(z=6),
                    carla.Rotation(pitch=-20, yaw=tf.rotation.yaw)
                ))

                # Print status every 20 frames (1 second)
                if frame_idx % 20 == 0 and frame_idx > 0:
                    obs_type_str = ['VEH', 'PED', '---'][nearest['obstacle_type']]
                    elapsed = time.time() - start_time
                    collisions_so_far = len(collision_recorder.collision_frame_indices)

                    print(f"  [{frame_idx:5d}] "
                          f"SPD:{ego_speed:5.1f}m/s  "
                          f"ACC:{ego_acceleration:+6.1f}m/s²  "
                          f"DIST:{nearest['distance']:5.1f}m  "
                          f"REL_V:{nearest['relative_velocity']:+5.1f}m/s  "
                          f"TTC:{ttc:5.1f}s  "
                          f"OBS:{obs_type_str}  "
                          f"STEER:{ego_control.steer:+5.2f}  "
                          f"COL:{collisions_so_far}  "
                          f"[{elapsed:.0f}s]")

                frame_idx += 1
            # ----------------------------------------------------------
            # Post-process: Apply Collision Labels
            # ----------------------------------------------------------
            num_collisions = len(collision_recorder.collision_frame_indices)
            print(f"\n  📋 Scenario {scenario_idx + 1} Summary:")
            print(f"     Frames collected: {len(scenario_data)}")
            print(f"     Collisions: {num_collisions}")

            if num_collisions > 0:
                print(f"     Collision details:")
                for detail in collision_recorder.collision_details:
                    print(f"       Frame {detail['frame_idx']}: "
                          f"hit {detail['other_actor']} "
                          f"(impulse: {detail['impulse']:.1f})")

            # Apply retroactive labels
            scenario_data = apply_collision_labels(
                scenario_data,
                collision_recorder.collision_frame_indices,
                LOOKAHEAD_FRAMES
            )

            # Feature statistics
            if len(scenario_data) > 0:
                speeds = [r['ego_speed'] for r in scenario_data]
                dists = [r['nearest_distance'] for r in scenario_data]
                rels = [r['relative_velocity'] for r in scenario_data]
                ttcs = [r['ttc'] for r in scenario_data]

                print(f"     Feature ranges:")
                print(f"       ego_speed:     {min(speeds):6.1f} - {max(speeds):6.1f} m/s  (avg: {np.mean(speeds):.1f})")
                print(f"       distance:      {min(dists):6.1f} - {max(dists):6.1f} m    (avg: {np.mean(dists):.1f})")
                print(f"       rel_velocity:  {min(rels):+6.1f} - {max(rels):+6.1f} m/s  (avg: {np.mean(rels):+.1f})")
                print(f"       ttc:           {min(ttcs):6.1f} - {max(ttcs):6.1f} s    (avg: {np.mean(ttcs):.1f})")

            # Write to CSV
            write_header = not os.path.exists(csv_path)
            write_scenario_to_csv(csv_path, scenario_data, write_header=write_header)

            # Update global stats
            pos_this = sum(1 for r in scenario_data if r['collision_within_2s'] == 1)
            global_frame_count += len(scenario_data)
            global_collision_count += num_collisions
            global_positive_labels += pos_this

            print(f"\n  📊 Running totals:")
            print(f"     Total frames: {global_frame_count:,}")
            print(f"     Total collisions: {global_collision_count}")
            print(f"     Total positive labels: {global_positive_labels} "
                  f"({global_positive_labels/max(1,global_frame_count)*100:.1f}%)")

            # ----------------------------------------------------------
            # Cleanup
            # ----------------------------------------------------------
            print(f"\n  🧹 Cleaning up scenario {scenario_idx + 1}...")
            collision_recorder.cleanup()
            radar_recorder.cleanup()
            cleanup_traffic(world, client, npc_vehicle_ids, walker_ids, ctrl_ids)
            npc_vehicle_ids = []
            walker_ids = []
            ctrl_ids = []

            if ego_vehicle:
                try:
                    ego_vehicle.destroy()
                except Exception:
                    pass
                ego_vehicle = None

            print(f"  ✅ Cleanup complete")
            time.sleep(1)

    except KeyboardInterrupt:
        print(f"\n\n  ⚠️  Interrupted by user after scenario {scenario_idx + 1}")

    except Exception as e:
        print(f"\n  ❌ Fatal error: {e}")
        traceback.print_exc()

    finally:
        # Final cleanup
        print(f"\n{'=' * 70}")
        print("FINAL CLEANUP")
        print(f"{'=' * 70}")

        if collision_recorder:
            try:
                collision_recorder.cleanup()
            except Exception:
                pass

        if radar_recorder:
            try:
                radar_recorder.cleanup()
            except Exception:
                pass

        cleanup_traffic(world, client, npc_vehicle_ids, walker_ids, ctrl_ids)

        if ego_vehicle:
            try:
                ego_vehicle.destroy()
            except Exception:
                pass

        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        print(f"\n{'=' * 70}")
        print("DATA COLLECTION COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Total scenarios run: {scenario_idx + 1}")
        print(f"  Total frames:        {global_frame_count:,}")
        print(f"  Total collisions:    {global_collision_count}")
        print(f"  Positive labels:     {global_positive_labels} "
              f"({global_positive_labels/max(1,global_frame_count)*100:.1f}%)")
        print(f"  Data saved to:       {csv_path}")
        print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
