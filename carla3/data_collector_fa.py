#!/usr/bin/env python3
"""
Simple & Stable Data Collector for False Alarm Detection.
Ego vehicle drives slowly on autopilot.
One lead vehicle is spawned ahead of it to create threat scenarios.
Dense fog weather. Minimal actors = no CARLA crashes.
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
SAVE_DIR = "dataset_fa"
FPS = 10
SAVE_FREQ = 5  # save every 5 ticks ≈ 2 frames/sec saved (no rush)

# Sensor specs
CAM_RES_X, CAM_RES_Y, CAM_FOV = 1024, 256, 110
RADAR_H_FOV, RADAR_V_FOV, RADAR_PTS = 30.0, 20.0, 1500

# Danger-zone geometry (ego-local, metres)
# Length = how far ahead to check
# Width = half-width from ego centerline (2m = your lane only, NOT opposite lane)
DANGER_LENGTH = 25.0
DANGER_WIDTH  = 2.0    # ego lane only (~3.5m lane, car is ~1.8m wide)

# Ego target speed (km/h) – Traffic Manager uses percentage difference
# from the road speed limit.  Town01 limit is ~30 km/h.
# 60% slower → ~12 km/h ≈ 3.3 m/s  (nice and chill)
EGO_SPEED_REDUCTION = 60   # percent slower than speed limit

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("DataCollector")


# ── Sensor Wrapper ────────────────────────────────────────────────────────────
class SensorData:
    def __init__(self):
        self.rgb = None
        self.radar = None
        self.frame = -1

    def rgb_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        self.rgb = array.reshape((image.height, image.width, 4))[:, :, :3]
        self.frame = image.frame

    def radar_callback(self, data):
        points = np.frombuffer(data.raw_data, dtype=np.float32)
        self.radar = points.reshape(-1, 4).copy()

    def all_ready(self):
        return self.rgb is not None and self.radar is not None


# ── Helpers ───────────────────────────────────────────────────────────────────
def setup_sensors(world, vehicle, sensor_data):
    bp_lib = world.get_blueprint_library()
    sensors = []

    # RGB camera
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(CAM_RES_X))
    cam_bp.set_attribute('image_size_y', str(CAM_RES_Y))
    cam_bp.set_attribute('fov', str(CAM_FOV))
    cam_tf = carla.Transform(carla.Location(x=-1.5, y=0.0, z=2.0),
                             carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0))
    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
    cam.listen(sensor_data.rgb_callback)
    sensors.append(cam)

    # Radar
    radar_bp = bp_lib.find('sensor.other.radar')
    radar_bp.set_attribute('horizontal_fov', str(RADAR_H_FOV))
    radar_bp.set_attribute('vertical_fov', str(RADAR_V_FOV))
    radar_bp.set_attribute('points_per_second', str(RADAR_PTS))
    radar_tf = carla.Transform(carla.Location(x=2.0, z=1.0))
    radar = world.spawn_actor(radar_bp, radar_tf, attach_to=vehicle)
    radar.listen(sensor_data.radar_callback)
    sensors.append(radar)

    return sensors


def check_danger_zone(ego_vehicle, all_actors, debug=False):
    """
    Returns (label, detail_string).
    label = 1 if any actor is in the forward danger zone, else 0.

    Uses simple position transform (not full matrix multiply) for accuracy.
    """
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
        world_dist = ego_loc.distance(actor_loc)
        if world_dist > 40.0:
            continue

        # Vector from ego to actor in world space
        dx = actor_loc.x - ego_loc.x
        dy = actor_loc.y - ego_loc.y

        # Project onto ego's forward and right axes (dot product)
        forward_dist = dx * ego_fwd.x + dy * ego_fwd.y   # +ve = ahead
        lateral_dist = dx * ego_right.x + dy * ego_right.y  # +ve = right

        # Use separate extents for length (x) and width (y)
        if hasattr(actor, 'bounding_box'):
            half_length = actor.bounding_box.extent.x  # front-to-back
            half_width  = actor.bounding_box.extent.y   # side-to-side
        else:
            half_length = 1.0
            half_width  = 0.5

        # Check if actor's nearest edge is within the danger zone
        # Forward: actor's rear edge (forward_dist - half_length) must be > 0.5m
        #          and its rear edge must be < DANGER_LENGTH
        in_front = ((forward_dist - half_length) > 0.5 and
                    (forward_dist - half_length) < DANGER_LENGTH)

        # Lateral: actor's nearest edge must be within DANGER_WIDTH of centerline
        in_path = (abs(lateral_dist) - half_width) < DANGER_WIDTH

        if debug and world_dist < 30.0:
            actor_type = actor.type_id.split('.')[-1]
            log.debug(f"  actor={actor_type:>15s}  fwd={forward_dist:+6.1f}m  "
                      f"lat={lateral_dist:+6.1f}m  "
                      f"half_l={half_length:.1f}  half_w={half_width:.1f}  "
                      f"in_front={in_front}  in_path={in_path}")

        if in_front and in_path:
            actor_type = actor.type_id.split('.')[-1]
            label = 1
            detail = f"threat:{actor_type}@{forward_dist:.1f}m"
            break

    return label, detail


def render_radar_bev(radar_data, label, speed_mps, bev_size=400, max_range=30.0):
    """
    Draw a Bird's Eye View of the radar detections.
    radar_data: Nx4 array [depth, azimuth, altitude, velocity]
    Returns a BGR image ready for cv2.imshow().

    Color code:
      RED    = approaching (negative velocity)
      BLUE   = receding  (positive velocity)
      brightness = |velocity|
    """
    bev = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)

    # Draw grid circles (range rings)
    cx, cy = bev_size // 2, bev_size  # ego is at bottom-center
    for ring_m in [5, 10, 15, 20, 25, 30]:
        r_px = int(ring_m / max_range * bev_size)
        cv2.circle(bev, (cx, cy), r_px, (40, 40, 40), 1)
        cv2.putText(bev, f"{ring_m}m", (cx + r_px + 2, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)

    # Draw danger zone rectangle
    dz_left  = int(cx - (DANGER_WIDTH / max_range * bev_size))
    dz_right = int(cx + (DANGER_WIDTH / max_range * bev_size))
    dz_top   = int(cy - (DANGER_LENGTH / max_range * bev_size))
    dz_bot   = int(cy - (0.5 / max_range * bev_size))
    cv2.rectangle(bev, (dz_left, dz_top), (dz_right, dz_bot),
                  (0, 100, 0) if label == 0 else (0, 0, 180), 2)

    # Draw ego vehicle marker
    cv2.circle(bev, (cx, cy), 5, (0, 255, 0), -1)
    cv2.putText(bev, "EGO", (cx - 15, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if radar_data is not None and len(radar_data) > 0:
        for det in radar_data:
            depth, azimuth, altitude, velocity = det

            # Polar to Cartesian (x = forward, y = left/right)
            x = depth * math.cos(azimuth)   # forward distance
            y = depth * math.sin(azimuth)   # lateral offset

            # Map to pixel coords (x is forward → up, y is lateral → left/right)
            px = int(cx - (y / max_range * bev_size))
            py = int(cy - (x / max_range * bev_size))

            if 0 <= px < bev_size and 0 <= py < bev_size:
                # Color by velocity: red = approaching, blue = receding
                v_norm = min(abs(velocity) / 15.0, 1.0)  # normalize to 0-1
                brightness = int(100 + 155 * v_norm)

                if velocity < 0:  # approaching
                    color = (0, 0, brightness)
                else:  # receding
                    color = (brightness, 0, 0)

                cv2.circle(bev, (px, py), 3, color, -1)

    # HUD text
    label_str = "THREAT" if label else "SAFE"
    label_color = (0, 0, 255) if label else (0, 255, 0)
    cv2.putText(bev, f"Label: {label_str}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
    cv2.putText(bev, f"Speed: {speed_mps:.1f} m/s", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(bev, f"Points: {len(radar_data) if radar_data is not None else 0}",
                (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Legend
    cv2.putText(bev, "RED=approaching  BLUE=receding",
                (10, bev_size - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    return bev


def spawn_lead_vehicle(world, ego_vehicle, traffic_manager):
    """
    Spawn a single NPC vehicle ahead of ego on the SAME LANE.
    Steps along the road in small increments to stay on the correct road.
    """
    bp_lib = world.get_blueprint_library()
    vehicle_bps = [bp for bp in bp_lib.filter('vehicle.*')
                   if int(bp.get_attribute('number_of_wheels')) >= 4]
    carla_map = world.get_map()

    ego_loc = ego_vehicle.get_location()
    log.info(f"   Ego actual position: ({ego_loc.x:.1f}, {ego_loc.y:.1f}, {ego_loc.z:.1f})")

    ego_wp = carla_map.get_waypoint(ego_loc,
                                    project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        log.error("❌  Could not get ego waypoint!")
        return None

    # Walk along the lane in small steps to build a chain of waypoints
    # This prevents jumping to a different road
    wp = ego_wp
    waypoint_chain = []
    for i in range(10):  # 10 steps x 3m = 30m ahead
        next_wps = wp.next(3.0)
        if not next_wps:
            break
        wp = next_wps[0]
        dist_from_ego = ego_loc.distance(wp.transform.location)
        waypoint_chain.append((wp, dist_from_ego))

    # Try to spawn at waypoints that are 10-25m from ego
    lead = None
    for wp, dist in waypoint_chain:
        if dist < 10.0 or dist > 25.0:
            continue  # too close or too far

        spawn_tf = wp.transform
        spawn_tf.location.z += 0.5

        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(
                bp.get_attribute('color').recommended_values))

        lead = world.try_spawn_actor(bp, spawn_tf)
        if lead is not None:
            # Use spawn_tf for distance (lead.get_location() returns 0,0,0 before tick)
            actual_dist = ego_loc.distance(spawn_tf.location)
            log.info(f"🚙  Lead vehicle spawned: {lead.type_id} "
                     f"at ({spawn_tf.location.x:.0f}, {spawn_tf.location.y:.0f}) "
                     f"  {actual_dist:.0f}m from ego")
            break

    if lead is None:
        log.error("❌  Could not spawn lead vehicle! Trying closest waypoint...")
        # Last resort: just try every waypoint in the chain
        for wp, dist in waypoint_chain:
            spawn_tf = wp.transform
            spawn_tf.location.z += 0.5
            bp = random.choice(vehicle_bps)
            lead = world.try_spawn_actor(bp, spawn_tf)
            if lead is not None:
                actual_dist = ego_loc.distance(spawn_tf.location)
                log.info(f"🚙  Lead vehicle spawned (fallback): "
                         f"{actual_dist:.0f}m from ego")
                break

    if lead is None:
        log.error("❌  Could not spawn lead vehicle at all!")
        return None

    lead.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.vehicle_percentage_speed_difference(lead, 80)
    traffic_manager.distance_to_leading_vehicle(lead, 5.0)

    return lead


def spawn_npc_vehicles(world, client, traffic_manager, count=20):
    """Spawn NPC traffic vehicles with autopilot."""
    bp_lib = world.get_blueprint_library()
    vehicle_bps = [bp for bp in bp_lib.filter('vehicle.*')
                   if int(bp.get_attribute('number_of_wheels')) >= 4]
    spawns = world.get_map().get_spawn_points()
    random.shuffle(spawns)

    batch = []
    for i in range(min(count, len(spawns))):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(
                bp.get_attribute('color').recommended_values))
        bp.set_attribute('role_name', 'autopilot')
        batch.append(carla.command.SpawnActor(bp, spawns[i])
                     .then(carla.command.SetAutopilot(
                         carla.command.FutureActor, True,
                         traffic_manager.get_port())))

    ids = []
    results = client.apply_batch_sync(batch, True)
    for r in results:
        if not r.error:
            ids.append(r.actor_id)

    log.info(f"🚗  Spawned {len(ids)}/{count} NPC vehicles")
    return ids


def spawn_pedestrians(world, client, count=30):
    """Spawn walking pedestrians with AI controllers."""
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    controller_bp = bp_lib.find('controller.ai.walker')

    # Spawn walkers
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

    # Attach AI controllers
    controllers = []
    for w in walkers:
        ctrl = world.spawn_actor(controller_bp,
                                 carla.Transform(), attach_to=w)
        controllers.append(ctrl)

    world.tick()  # let navigation mesh settle

    # Start walking
    for ctrl in controllers:
        dest = world.get_random_location_from_navigation()
        if dest is not None:
            ctrl.start()
            ctrl.go_to_location(dest)
            ctrl.set_max_speed(1.0 + random.random() * 1.5)

    walker_ids = [w.id for w in walkers]
    ctrl_ids = [c.id for c in controllers]
    log.info(f"🚶  Spawned {len(walkers)}/{count} pedestrians")
    return walker_ids, ctrl_ids


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Simple CARLA data collector – ego + 1 lead vehicle")
    parser.add_argument('--town', default='Town01',
                        help='CARLA Town to load')
    parser.add_argument('--frames', type=int, default=5000,
                        help='Total frames to collect')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Disable live radar BEV + RGB camera windows')
    args = parser.parse_args()

    # Output dirs
    os.makedirs(SAVE_DIR, exist_ok=True)
    images_dir = os.path.join(SAVE_DIR, "images")
    radar_dir  = os.path.join(SAVE_DIR, "radar")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(radar_dir, exist_ok=True)
    csv_path = os.path.join(SAVE_DIR, "labels.csv")

    # Connect
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(20.0)

    log.info(f"Loading {args.town} …")
    world = client.load_world(args.town)

    # MAXIMUM fog (as intense as possible)
    weather = carla.WeatherParameters(
        cloudiness=100, precipitation=0, precipitation_deposits=0,
        wind_intensity=0, sun_altitude_angle=45,
        fog_density=100, fog_distance=0, fog_falloff=0.1, wetness=30)
    world.set_weather(weather)
    log.info("⛅  Weather: MAXIMUM fog")

    # Synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)

    sensors = []
    lead_vehicle = None

    try:
        # ── 1. Spawn Ego Vehicle ─────────────────────────────────────────
        bp_lib = world.get_blueprint_library()
        ego_bp = random.choice(bp_lib.filter('vehicle.tesla.model3'))
        ego_bp.set_attribute('role_name', 'hero')
        spawn_points = world.get_map().get_spawn_points()
        ego_vehicle = world.spawn_actor(ego_bp, random.choice(spawn_points))
        log.info(f"🚗  Ego spawned at "
                 f"{ego_vehicle.get_location().x:.0f}, "
                 f"{ego_vehicle.get_location().y:.0f}")

        # ── 2. Traffic Manager (AFTER ego spawn) ─────────────────────────
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        # Enable autopilot on ego – SLOW
        ego_vehicle.set_autopilot(True, 8000)
        traffic_manager.vehicle_percentage_speed_difference(
            ego_vehicle, EGO_SPEED_REDUCTION)
        traffic_manager.distance_to_leading_vehicle(ego_vehicle, 4.0)
        traffic_manager.auto_lane_change(ego_vehicle, False)  # stay in lane
        log.info(f"🐌  Ego speed: {EGO_SPEED_REDUCTION}% below limit "
                 f"(~{30 * (100 - EGO_SPEED_REDUCTION) / 100:.0f} km/h)")

        # ── 3. Sensors ───────────────────────────────────────────────────
        sensor_data = SensorData()
        sensors = setup_sensors(world, ego_vehicle, sensor_data)

        # Let world settle
        for _ in range(20):
            world.tick()

        # ── 4. Spawn lead vehicle ahead of ego ────────────────────────
        lead_vehicle = spawn_lead_vehicle(world, ego_vehicle, traffic_manager)

        # ── 5. Spawn NPC traffic + pedestrians ───────────────────────────
        npc_vehicle_ids = spawn_npc_vehicles(world, client, traffic_manager, 20)
        ped_ids, ped_ctrl_ids = spawn_pedestrians(world, client, 30)

        # Let it settle
        for _ in range(10):
            world.tick()

        # ── 6. Collection Loop ───────────────────────────────────────────
        log.info("=" * 50)
        log.info("  COLLECTION STARTED")
        log.info(f"  Target: {args.frames} frames")
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
                                 "ego_speed_mps"])

            while saved_count < args.frames:
                try:
                    world.tick()
                except RuntimeError as e:
                    log.warning(f"world.tick() failed: {e}")
                    log.warning(f"Saved {saved_count} frames before crash.")
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

                # Ground-truth label (debug every 25 saved frames)
                vehicles    = world.get_actors().filter('*vehicle*')
                pedestrians = world.get_actors().filter('*walker*')
                do_debug = (saved_count % 25 == 0)
                label, detail = check_danger_zone(
                    ego_vehicle, list(vehicles) + list(pedestrians),
                    debug=do_debug)

                ego_vel = ego_vehicle.get_velocity()
                speed_mps = math.sqrt(
                    ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)

                # Save
                img_path = os.path.join(images_dir, f"{saved_count:05d}.png")
                rgb_bgr = cv2.cvtColor(sensor_data.rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path, rgb_bgr)
                np.save(os.path.join(radar_dir, f"{saved_count:05d}.npy"),
                        sensor_data.radar)

                writer.writerow([f"{saved_count:05d}", label, detail,
                                 f"{speed_mps:.2f}"])
                csvfile.flush()

                if label == 1:
                    threat_count += 1

                # ── Live visualization ───────────────────────────
                if not args.no_visualize:
                    # Radar BEV
                    bev = render_radar_bev(sensor_data.radar, label,
                                          speed_mps)
                    cv2.imshow("Radar BEV", bev)

                    # RGB camera feed (resized for display)
                    rgb_disp = cv2.resize(rgb_bgr, (800, 200))
                    # Draw danger zone label on camera too
                    lbl_clr = (0, 0, 255) if label else (0, 255, 0)
                    cv2.putText(rgb_disp,
                                f"{'THREAT' if label else 'SAFE'}  "
                                f"{detail}",
                                (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                lbl_clr, 2)
                    cv2.imshow("RGB Camera", rgb_disp)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        log.info("User pressed 'q' – stopping.")
                        break

                if saved_count % 50 == 0:
                    pct = (threat_count / max(1, saved_count + 1)) * 100
                    # Show distance to lead vehicle for debugging
                    lead_dist = "N/A"
                    if lead_vehicle is not None:
                        lead_dist = f"{ego_vehicle.get_location().distance(lead_vehicle.get_location()):.1f}m"
                    log.info(
                        f"[{saved_count}/{args.frames}]  "
                        f"{'THREAT' if label else '  SAFE'}  "
                        f"speed={speed_mps:.1f} m/s  "
                        f"lead_dist={lead_dist}  "
                        f"threats={threat_count} ({pct:.1f}%)")

                saved_count += 1

        log.info("=" * 50)
        log.info(f"  DONE – {saved_count} frames saved")
        log.info(f"  Threats: {threat_count}  "
                 f"({threat_count / max(1, saved_count) * 100:.1f}%)")
        log.info("=" * 50)

    except Exception as e:
        log.error(f"Fatal error: {e}")
        traceback.print_exc()

    finally:
        log.info("Cleaning up …")

        # Stop pedestrian controllers
        if 'ped_ctrl_ids' in locals():
            for cid in ped_ctrl_ids:
                try:
                    a = world.get_actor(cid)
                    if a: a.stop()
                except Exception:
                    pass

        # Batch destroy NPCs + pedestrians
        destroy_ids = []
        if 'npc_vehicle_ids' in locals():
            destroy_ids.extend(npc_vehicle_ids)
        if 'ped_ctrl_ids' in locals():
            destroy_ids.extend(ped_ctrl_ids)
        if 'ped_ids' in locals():
            destroy_ids.extend(ped_ids)
        if destroy_ids:
            client.apply_batch([carla.command.DestroyActor(x) for x in destroy_ids])

        # Sensors
        for s in sensors:
            try:
                s.destroy()
            except Exception:
                pass
        # Lead vehicle
        if lead_vehicle is not None:
            try:
                lead_vehicle.destroy()
            except Exception:
                pass
        # Ego
        if 'ego_vehicle' in locals():
            try:
                ego_vehicle.destroy()
            except Exception:
                pass
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except Exception:
            pass
        cv2.destroyAllWindows()
        log.info("Done ✅")


if __name__ == '__main__':
    main()
