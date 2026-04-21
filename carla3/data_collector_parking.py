#!/usr/bin/env python3
"""
Parking & Bicycle Scenario Data Collector.
- Parks vehicles along road edges (both sides)
- Spawns bicycles on footpaths / sidewalks
- Ego drives slowly through these zones
- Same sensor setup + danger zone labels as main collector
"""

import carla
import numpy as np
import cv2
import os
import random
import time
import argparse
import csv
import math
import logging
import traceback

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR = "dataset_parking"
FPS = 10
SAVE_FREQ = 5  # save every 5 ticks ≈ 2 frames/sec

# Sensor specs (same as main collector)
CAM_RES_X, CAM_RES_Y, CAM_FOV = 1024, 256, 110
RADAR_H_FOV, RADAR_V_FOV, RADAR_PTS = 30.0, 20.0, 1500

# Danger zone (ego-local, metres)
DANGER_LENGTH = 25.0
DANGER_WIDTH  = 2.0   # ego lane only

# Scenario counts
NUM_PARKED_VEHICLES = 15   # parked along road edges
NUM_BICYCLES        = 8    # on footpaths / sidewalks

# Ego speed
EGO_SPEED_REDUCTION = 60   # 60% below limit ≈ 12 km/h

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ParkingCollector")


# ── Sensor Wrapper ────────────────────────────────────────────────────────────
class SensorData:
    def __init__(self):
        self.rgb = None
        self.radar = None

    def rgb_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        self.rgb = array.reshape((image.height, image.width, 4))[:, :, :3]

    def radar_callback(self, data):
        points = np.frombuffer(data.raw_data, dtype=np.float32)
        self.radar = points.reshape(-1, 4).copy()

    def all_ready(self):
        return self.rgb is not None and self.radar is not None


# ── Helpers ───────────────────────────────────────────────────────────────────
def setup_sensors(world, vehicle, sensor_data):
    bp_lib = world.get_blueprint_library()
    sensors = []

    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(CAM_RES_X))
    cam_bp.set_attribute('image_size_y', str(CAM_RES_Y))
    cam_bp.set_attribute('fov', str(CAM_FOV))
    cam_tf = carla.Transform(carla.Location(x=-1.5, y=0.0, z=2.0))
    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
    cam.listen(sensor_data.rgb_callback)
    sensors.append(cam)

    radar_bp = bp_lib.find('sensor.other.radar')
    radar_bp.set_attribute('horizontal_fov', str(RADAR_H_FOV))
    radar_bp.set_attribute('vertical_fov', str(RADAR_V_FOV))
    radar_bp.set_attribute('points_per_second', str(RADAR_PTS))
    radar_tf = carla.Transform(carla.Location(x=2.0, z=1.0))
    radar = world.spawn_actor(radar_bp, radar_tf, attach_to=vehicle)
    radar.listen(sensor_data.radar_callback)
    sensors.append(radar)

    return sensors


def check_danger_zone(ego_vehicle, all_actors):
    """Same danger zone check as main collector."""
    ego_t = ego_vehicle.get_transform()
    ego_loc = ego_t.location
    ego_fwd = ego_t.get_forward_vector()
    ego_right = ego_t.get_right_vector()
    label = 0
    detail = "safe"

    for actor in all_actors:
        if actor.id == ego_vehicle.id:
            continue
        actor_loc = actor.get_location()
        if ego_loc.distance(actor_loc) > 40.0:
            continue

        dx = actor_loc.x - ego_loc.x
        dy = actor_loc.y - ego_loc.y
        forward_dist = dx * ego_fwd.x + dy * ego_fwd.y
        lateral_dist = dx * ego_right.x + dy * ego_right.y

        if hasattr(actor, 'bounding_box'):
            half_length = actor.bounding_box.extent.x
            half_width  = actor.bounding_box.extent.y
        else:
            half_length = 1.0
            half_width  = 0.5

        in_front = ((forward_dist - half_length) > 0.5 and
                    (forward_dist - half_length) < DANGER_LENGTH)
        in_path = (abs(lateral_dist) - half_width) < DANGER_WIDTH

        if in_front and in_path:
            actor_type = actor.type_id.split('.')[-1]
            label = 1
            detail = f"threat:{actor_type}@{forward_dist:.1f}m"
            break

    return label, detail


def render_radar_bev(radar_data, label, speed_mps, bev_size=400, max_range=30.0):
    """Bird's Eye View of radar detections."""
    bev = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    cx, cy = bev_size // 2, bev_size

    for ring_m in [5, 10, 15, 20, 25, 30]:
        r_px = int(ring_m / max_range * bev_size)
        cv2.circle(bev, (cx, cy), r_px, (40, 40, 40), 1)

    # Danger zone
    dz_left  = int(cx - (DANGER_WIDTH / max_range * bev_size))
    dz_right = int(cx + (DANGER_WIDTH / max_range * bev_size))
    dz_top   = int(cy - (DANGER_LENGTH / max_range * bev_size))
    dz_bot   = int(cy - (0.5 / max_range * bev_size))
    cv2.rectangle(bev, (dz_left, dz_top), (dz_right, dz_bot),
                  (0, 100, 0) if label == 0 else (0, 0, 180), 2)

    # Ego marker
    cv2.circle(bev, (cx, cy), 5, (0, 255, 0), -1)

    if radar_data is not None and len(radar_data) > 0:
        for det in radar_data:
            depth, azimuth, altitude, velocity = det
            x = depth * math.cos(azimuth)
            y = depth * math.sin(azimuth)
            px = int(cx - (y / max_range * bev_size))
            py = int(cy - (x / max_range * bev_size))
            if 0 <= px < bev_size and 0 <= py < bev_size:
                v_norm = min(abs(velocity) / 15.0, 1.0)
                brightness = int(100 + 155 * v_norm)
                color = (0, 0, brightness) if velocity < 0 else (brightness, 0, 0)
                cv2.circle(bev, (px, py), 3, color, -1)

    # HUD
    lbl_str = "THREAT" if label else "SAFE"
    lbl_clr = (0, 0, 255) if label else (0, 255, 0)
    cv2.putText(bev, f"Label: {lbl_str}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, lbl_clr, 2)
    cv2.putText(bev, f"Speed: {speed_mps:.1f} m/s", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(bev, "PARKING SCENARIO", (10, bev_size - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    return bev


# ── Scenario Spawners ─────────────────────────────────────────────────────────
def spawn_parked_vehicles(world, ego_vehicle, count=15):
    """
    Park vehicles along road edges near the ego's route.
    Uses shoulder/parking lane types, or road edge waypoints.
    """
    bp_lib = world.get_blueprint_library()
    car_bps = [bp for bp in bp_lib.filter('vehicle.*')
               if int(bp.get_attribute('number_of_wheels')) >= 4]
    carla_map = world.get_map()

    ego_wp = carla_map.get_waypoint(ego_vehicle.get_location(),
                                    project_to_road=True)
    if ego_wp is None:
        log.warning("Could not get ego waypoint for parking")
        return []

    # Walk along the ego's road and place parked cars on the side
    parked = []
    wp = ego_wp
    for i in range(count * 3):  # try more spots than needed
        if len(parked) >= count:
            break

        # Step ahead on the road
        next_wps = wp.next(8.0 + random.uniform(-2, 2))
        if not next_wps:
            break
        wp = next_wps[0]

        # Try to get the right shoulder or lane edge
        # Offset the waypoint to the right side of the road
        right_wp = wp.get_right_lane()
        if right_wp and right_wp.lane_type == carla.LaneType.Parking:
            park_tf = right_wp.transform
        else:
            # Offset manually to road edge (~3.5m right of lane center)
            fwd = wp.transform.get_forward_vector()
            right = wp.transform.get_right_vector()
            offset = random.choice([3.2, 3.5, -3.2, -3.5])  # left or right side
            park_loc = wp.transform.location + carla.Location(
                x=right.x * offset,
                y=right.y * offset,
                z=0.3)
            # Align with road direction
            park_tf = carla.Transform(park_loc, wp.transform.rotation)

        bp = random.choice(car_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(
                bp.get_attribute('color').recommended_values))

        v = world.try_spawn_actor(bp, park_tf)
        if v is not None:
            # Kill the engine — make it truly parked (no movement)
            v.set_target_velocity(carla.Vector3D(0, 0, 0))
            v.apply_control(carla.VehicleControl(
                throttle=0, brake=1.0, hand_brake=True))
            parked.append(v)

    log.info(f"🅿️  Parked {len(parked)}/{count} vehicles along road edges")
    return parked


def spawn_bicycles_on_sidewalk(world, ego_vehicle, count=8):
    """
    Spawn bicycles on sidewalks / footpaths near the ego's route.
    """
    bp_lib = world.get_blueprint_library()
    bike_bps = bp_lib.filter('vehicle.bh.crossbike')
    if len(bike_bps) == 0:
        bike_bps = bp_lib.filter('vehicle.*bicycle*')
    if len(bike_bps) == 0:
        # Fallback: use any 2-wheeled vehicle
        bike_bps = [bp for bp in bp_lib.filter('vehicle.*')
                    if int(bp.get_attribute('number_of_wheels')) == 2]
    if len(bike_bps) == 0:
        log.warning("No bicycle blueprints found!")
        return []

    carla_map = world.get_map()
    ego_wp = carla_map.get_waypoint(ego_vehicle.get_location(),
                                    project_to_road=True)
    if ego_wp is None:
        return []

    bikes = []
    wp = ego_wp
    for i in range(count * 4):
        if len(bikes) >= count:
            break

        next_wps = wp.next(10.0 + random.uniform(-3, 5))
        if not next_wps:
            break
        wp = next_wps[0]

        # Try to get a sidewalk waypoint
        right = wp.transform.get_right_vector()
        # Sidewalk is typically 5-7m from lane center
        offset = random.choice([5.5, 6.0, 6.5, -5.5, -6.0, -6.5])
        sidewalk_loc = wp.transform.location + carla.Location(
            x=right.x * offset,
            y=right.y * offset,
            z=0.3)

        # Check if this location is on a sidewalk
        sw_wp = carla_map.get_waypoint(sidewalk_loc,
                                       project_to_road=False,
                                       lane_type=carla.LaneType.Sidewalk)
        if sw_wp is None:
            # Just try the offset location anyway
            bike_tf = carla.Transform(sidewalk_loc, wp.transform.rotation)
        else:
            bike_tf = sw_wp.transform
            bike_tf.location.z += 0.3

        bp = random.choice(bike_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(
                bp.get_attribute('color').recommended_values))

        bike = world.try_spawn_actor(bp, bike_tf)
        if bike is not None:
            # Parked bike — no movement
            bike.set_target_velocity(carla.Vector3D(0, 0, 0))
            bike.apply_control(carla.VehicleControl(
                throttle=0, brake=1.0, hand_brake=True))
            bikes.append(bike)

    log.info(f"🚲  Spawned {len(bikes)}/{count} bicycles on sidewalks")
    return bikes


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Parking & Bicycle scenario data collector")
    parser.add_argument('--town', default='Town01')
    parser.add_argument('--frames', type=int, default=5000)
    parser.add_argument('--no-visualize', action='store_true',
                        help='Disable live radar + camera windows')
    parser.add_argument('--parked', type=int, default=NUM_PARKED_VEHICLES)
    parser.add_argument('--bikes', type=int, default=NUM_BICYCLES)
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    images_dir = os.path.join(SAVE_DIR, "images")
    radar_dir  = os.path.join(SAVE_DIR, "radar")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(radar_dir, exist_ok=True)
    csv_path = os.path.join(SAVE_DIR, "labels.csv")

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(20.0)

    log.info(f"Loading {args.town} …")
    world = client.load_world(args.town)

    # Clear weather (good visibility to see parked cars + bikes)
    weather = carla.WeatherParameters(
        cloudiness=20, precipitation=0, precipitation_deposits=0,
        wind_intensity=5, sun_altitude_angle=60,
        fog_density=0, fog_distance=100, wetness=0)
    world.set_weather(weather)
    log.info("☀️  Weather: clear day")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)

    sensors = []
    parked_vehicles = []
    bikes = []

    try:
        # ── 1. Spawn Ego ─────────────────────────────────────────────────
        bp_lib = world.get_blueprint_library()
        ego_bp = random.choice(bp_lib.filter('vehicle.tesla.model3'))
        ego_bp.set_attribute('role_name', 'hero')
        spawn_points = world.get_map().get_spawn_points()
        ego_vehicle = world.spawn_actor(ego_bp, random.choice(spawn_points))
        log.info(f"🚗  Ego spawned")

        # ── 2. Traffic Manager ───────────────────────────────────────────
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        ego_vehicle.set_autopilot(True, 8000)
        tm.vehicle_percentage_speed_difference(ego_vehicle, EGO_SPEED_REDUCTION)
        tm.auto_lane_change(ego_vehicle, False)
        log.info(f"🐌  Ego driving slowly")

        # ── 3. Sensors ───────────────────────────────────────────────────
        sensor_data = SensorData()
        sensors = setup_sensors(world, ego_vehicle, sensor_data)

        # Let world settle
        for _ in range(30):
            world.tick()

        # ── 4. Spawn parking scenario ────────────────────────────────────
        parked_vehicles = spawn_parked_vehicles(world, ego_vehicle, args.parked)

        # ── 5. Spawn bicycles on sidewalks ───────────────────────────────
        bikes = spawn_bicycles_on_sidewalk(world, ego_vehicle, args.bikes)

        # Settle
        for _ in range(10):
            world.tick()

        # ── 6. Collection Loop ───────────────────────────────────────────
        log.info("=" * 50)
        log.info("  PARKING SCENARIO COLLECTION STARTED")
        log.info(f"  Target: {args.frames} frames")
        log.info(f"  Parked: {len(parked_vehicles)}  Bikes: {len(bikes)}")
        log.info("=" * 50)

        frame_count = 0
        saved_count = 0
        threat_count = 0

        write_header = (not os.path.exists(csv_path)
                        or os.path.getsize(csv_path) == 0)
        with open(csv_path, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["frame", "label_0_1", "detail",
                                 "ego_speed_mps", "scenario"])

            while saved_count < args.frames:
                try:
                    world.tick()
                except RuntimeError as e:
                    log.warning(f"world.tick() failed: {e}")
                    break

                frame_count += 1

                # Spectator follow-cam
                spectator = world.get_spectator()
                tf = ego_vehicle.get_transform()
                cam_loc = (tf.location
                           - tf.get_forward_vector() * 6.0
                           + carla.Location(z=3.0))
                cam_rot = carla.Rotation(
                    pitch=tf.rotation.pitch - 15.0,
                    yaw=tf.rotation.yaw,
                    roll=tf.rotation.roll)
                spectator.set_transform(carla.Transform(cam_loc, cam_rot))

                if not sensor_data.all_ready():
                    continue
                if frame_count % SAVE_FREQ != 0:
                    continue

                # Label
                vehicles    = world.get_actors().filter('*vehicle*')
                pedestrians = world.get_actors().filter('*walker*')
                label, detail = check_danger_zone(
                    ego_vehicle, list(vehicles) + list(pedestrians))

                ego_vel = ego_vehicle.get_velocity()
                speed_mps = math.sqrt(
                    ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)

                # Save
                rgb_bgr = cv2.cvtColor(sensor_data.rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(images_dir, f"{saved_count:05d}.png"),
                    rgb_bgr)
                np.save(
                    os.path.join(radar_dir, f"{saved_count:05d}.npy"),
                    sensor_data.radar)

                writer.writerow([f"{saved_count:05d}", label, detail,
                                 f"{speed_mps:.2f}", "parking"])
                csvfile.flush()

                if label == 1:
                    threat_count += 1

                # Visualization
                if not args.no_visualize:
                    bev = render_radar_bev(sensor_data.radar, label, speed_mps)
                    cv2.imshow("Radar BEV", bev)

                    rgb_disp = cv2.resize(rgb_bgr, (800, 200))
                    lbl_clr = (0, 0, 255) if label else (0, 255, 0)
                    cv2.putText(rgb_disp,
                                f"{'THREAT' if label else 'SAFE'}  {detail}",
                                (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, lbl_clr, 2)
                    cv2.putText(rgb_disp, "PARKING SCENARIO", (600, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                    cv2.imshow("RGB Camera", rgb_disp)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        log.info("User pressed 'q'")
                        break

                if saved_count % 50 == 0:
                    pct = (threat_count / max(1, saved_count + 1)) * 100
                    log.info(
                        f"[{saved_count}/{args.frames}]  "
                        f"{'THREAT' if label else '  SAFE'}  "
                        f"speed={speed_mps:.1f} m/s  "
                        f"threats={threat_count} ({pct:.1f}%)")

                saved_count += 1

        log.info("=" * 50)
        log.info(f"  DONE – {saved_count} frames saved")
        log.info(f"  Threats: {threat_count} "
                 f"({threat_count / max(1, saved_count) * 100:.1f}%)")
        log.info("=" * 50)

    except Exception as e:
        log.error(f"Fatal error: {e}")
        traceback.print_exc()

    finally:
        log.info("Cleaning up …")
        for s in sensors:
            try: s.destroy()
            except: pass
        for v in parked_vehicles:
            try: v.destroy()
            except: pass
        for b in bikes:
            try: b.destroy()
            except: pass
        if 'ego_vehicle' in locals():
            try: ego_vehicle.destroy()
            except: pass
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except: pass
        cv2.destroyAllWindows()
        log.info("Done ✅")


if __name__ == '__main__':
    main()
