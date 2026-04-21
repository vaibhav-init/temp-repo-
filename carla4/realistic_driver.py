#!/usr/bin/env python3
"""
Simple Forward Driver for CARLA - Accident Prediction Data Collection
======================================================================

This script uses a simple lane-following controller in challenging conditions.

Features:
  - Simple forward driving controller (follows lane waypoints)
  - Target speed: 30 km/h (safe city speed)
  - Basic obstacle avoidance (brakes if vehicle ahead)
  - 60 NPC traffic vehicles with autopilot
  - 50 pedestrians walking around
  - Fog weather (density=80) for reduced visibility
  - Rain (intensity=60) with wet roads (wetness=70)
  - Challenging scenarios (sudden brakes, stopped vehicles, jaywalkers)
  - Slower simulation speed (20 FPS) for better stability

Perfect for generating realistic driving scenarios for accident prediction.

Usage:
    python realistic_driver.py

Press Ctrl+C to stop
Scenarios auto-restart every 90 seconds
"""

import carla
import numpy as np
import time
import random
import math
from collections import deque

# No need to import BasicAgent - using simple custom controller instead

# ============================================================================
# Configuration
# ============================================================================
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
TOWN = 'Town01'  # Town01 is more stable

# Note: These parameters are NOT used anymore - Using simple forward controller
# Kept for reference only (not applied to current controller)
REACTION_DELAY = 0.5        # (Not used)
SPEED_EXCESS = 30           # (Not used - controller targets 30 km/h fixed speed)
FOLLOWING_DISTANCE = 8.0    # (Not used - simple obstacle detection at 10m)
BRAKE_INTENSITY = 0.5       # (Not used)
RISK_TAKING = 0.85          # (Not used)

# Traffic and environment
NUM_TRAFFIC_VEHICLES = 60   # Number of NPC vehicles (increased for more chaos)
NUM_PEDESTRIANS = 50        # Number of pedestrians (more jaywalking risk)

# Weather settings
USE_FOG = True              # Enable dense fog
FOG_DENSITY = 80            # 0-100 (80 = dense fog)
USE_RAIN = True             # Enable rain
RAIN_INTENSITY = 60         # 0-100 (60 = moderate rain)
WETNESS = 70                # Road wetness 0-100

# How long to run each scenario
SCENARIO_DURATION = 90  # seconds (longer to capture more crash scenarios)

# ============================================================================
# Simple Forward Driver (Follows Lane)
# ============================================================================
class SemiProDriver:
    """
    Simple driver that just drives forward following the lane.

    Uses waypoints to follow the road and simple PID control for steering.
    Target speed: 30 km/h (safe city speed)
    """

    def __init__(self, vehicle, world_map):
        self.vehicle = vehicle
        self.world_map = world_map
        self.target_speed = 30.0  # km/h - safe city speed

        print(f"  \U0001f697 Simple Forward Driver:")
        print(f"     Target speed: {self.target_speed} km/h")
        print(f"     Following lane waypoints")
        print(f"     Simple obstacle avoidance")

    def run_step(self):
        """
        Execute one step - drive forward following the lane.
        Returns: VehicleControl
        """
        # Get current vehicle state
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_velocity = self.vehicle.get_velocity()
        current_speed = 3.6 * math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)  # km/h

        # Get waypoint ahead
        current_waypoint = self.world_map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)

        if current_waypoint is None:
            # No waypoint found, just drive straight
            control = carla.VehicleControl()
            control.throttle = 0.3
            control.steer = 0.0
            control.brake = 0.0
            return control

        # Get waypoint 10 meters ahead
        next_waypoints = current_waypoint.next(10.0)
        if not next_waypoints:
            # Can't find next waypoint, drive straight
            control = carla.VehicleControl()
            control.throttle = 0.3
            control.steer = 0.0
            control.brake = 0.0
            return control

        target_waypoint = next_waypoints[0]
        target_location = target_waypoint.transform.location

        # Calculate steering (simple proportional control)
        # Get vector from vehicle to target
        target_vector = target_location - vehicle_location

        # Get vehicle's forward vector
        forward_vector = vehicle_transform.get_forward_vector()

        # Calculate angle between vehicle heading and target direction
        # Using cross product for angle
        cross_product = forward_vector.x * target_vector.y - forward_vector.y * target_vector.x
        dot_product = forward_vector.x * target_vector.x + forward_vector.y * target_vector.y

        angle = math.atan2(cross_product, dot_product)

        # Simple proportional steering (clamp between -1 and 1)
        steer = max(-1.0, min(1.0, angle * 1.5))

        # Speed control (simple)
        if current_speed < self.target_speed - 5:
            throttle = 0.6
            brake = 0.0
        elif current_speed > self.target_speed + 5:
            throttle = 0.0
            brake = 0.3
        else:
            throttle = 0.4
            brake = 0.0

        # Basic obstacle detection - check for vehicles ahead
        # (Simple version - just brake if going too fast near others)
        world = self.vehicle.get_world()
        vehicles = world.get_actors().filter('*vehicle*')

        for other_vehicle in vehicles:
            if other_vehicle.id == self.vehicle.id:
                continue

            other_location = other_vehicle.get_location()
            distance = vehicle_location.distance(other_location)

            # If vehicle is very close ahead, brake
            if distance < 10.0:
                # Check if it's ahead
                to_other = other_location - vehicle_location
                dot = forward_vector.x * to_other.x + forward_vector.y * to_other.y

                if dot > 0:  # Ahead of us
                    brake = 0.5
                    throttle = 0.0
                    break

        # Create control
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        control.brake = brake
        control.hand_brake = False
        control.manual_gear_shift = False

        return control


# ============================================================================
# Challenging Scenario Generator
# ============================================================================
def spawn_challenging_scenario(world, ego_vehicle, scenario_type='random'):
    """
    Spawn challenging traffic scenarios.

    Types:
      - sudden_brake: Lead vehicle brakes hard
      - cut_in: Vehicle cuts into ego's lane
      - stopped_vehicle: Stationary obstacle
      - jaywalker: Pedestrian crosses road
      - tight_traffic: Dense traffic
    """

    if scenario_type == 'random':
        scenario_type = random.choice([
            'sudden_brake', 'stopped_vehicle', 'tight_traffic', 'jaywalker'
        ])

    bp_lib = world.get_blueprint_library()
    carla_map = world.get_map()
    spawned_actors = []

    ego_loc = ego_vehicle.get_location()
    ego_wp = carla_map.get_waypoint(ego_loc, project_to_road=True)

    print(f"\n  \U0001f3ac Scenario: {scenario_type.upper()}")

    # ========================================================================
    # Scenario 1: Sudden Brake
    # ========================================================================
    if scenario_type == 'sudden_brake':
        # Spawn vehicle ahead that will brake suddenly (CLOSER for accident prediction)
        ahead_distance = random.uniform(15, 25)
        ahead_wp = ego_wp

        for _ in range(int(ahead_distance / 3)):
            next_wps = ahead_wp.next(3.0)
            if not next_wps:
                break
            ahead_wp = next_wps[0]

        vehicle_bp = random.choice(bp_lib.filter('vehicle.*'))
        spawn_tf = ahead_wp.transform
        spawn_tf.location.z += 0.5

        lead_vehicle = world.try_spawn_actor(vehicle_bp, spawn_tf)
        if lead_vehicle:
            # Drive at moderate speed, then brake hard
            lead_vehicle.enable_constant_velocity(carla.Vector3D(5.0, 0, 0))  # 5 m/s forward

            spawned_actors.append({
                'actor': lead_vehicle,
                'type': 'sudden_braker',
                'brake_time': time.time() + random.uniform(5, 8)  # Brake after 5-8 seconds (less warning)
            })

            print(f"     Lead vehicle spawned at {ahead_distance:.0f}m (will brake suddenly)")

    # ========================================================================
    # Scenario 2: Stopped Vehicle
    # ========================================================================
    elif scenario_type == 'stopped_vehicle':
        ahead_distance = random.uniform(18, 30)  # CLOSER for more challenging scenario
        ahead_wp = ego_wp

        for _ in range(int(ahead_distance / 3)):
            next_wps = ahead_wp.next(3.0)
            if not next_wps:
                break
            ahead_wp = next_wps[0]

        vehicle_bp = random.choice(bp_lib.filter('vehicle.*'))
        spawn_tf = ahead_wp.transform
        spawn_tf.location.z += 0.5

        stopped_vehicle = world.try_spawn_actor(vehicle_bp, spawn_tf)
        if stopped_vehicle:
            stopped_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            stopped_vehicle.apply_control(carla.VehicleControl(brake=1.0))

            spawned_actors.append({
                'actor': stopped_vehicle,
                'type': 'stopped',
            })

            print(f"     Stopped vehicle at {ahead_distance:.0f}m")

    # ========================================================================
    # Scenario 3: Tight Traffic
    # ========================================================================
    elif scenario_type == 'tight_traffic':
        # Spawn multiple vehicles around ego
        num_vehicles = random.randint(8, 12)

        spawn_points = carla_map.get_spawn_points()
        random.shuffle(spawn_points)

        for i, spawn_point in enumerate(spawn_points[:num_vehicles]):
            # Only spawn if reasonably close to ego
            if ego_loc.distance(spawn_point.location) < 80:
                vehicle_bp = random.choice(bp_lib.filter('vehicle.*'))
                npc = world.try_spawn_actor(vehicle_bp, spawn_point)

                if npc:
                    # Mix of speeds - some moving, some slow
                    speed = random.uniform(2.0, 8.0)  # 2-8 m/s
                    forward = spawn_point.get_forward_vector()
                    npc.enable_constant_velocity(carla.Vector3D(
                        forward.x * speed,
                        forward.y * speed,
                        0
                    ))

                    spawned_actors.append({
                        'actor': npc,
                        'type': 'traffic',
                    })

        print(f"     Spawned {len(spawned_actors)} traffic vehicles")

    # ========================================================================
    # Scenario 4: Jaywalker
    # ========================================================================
    elif scenario_type == 'jaywalker':
        # Spawn pedestrian that will cross the road (CLOSER and faster)
        ahead_distance = random.uniform(10, 18)
        ahead_wp = ego_wp

        for _ in range(int(ahead_distance / 3)):
            next_wps = ahead_wp.next(3.0)
            if not next_wps:
                break
            ahead_wp = next_wps[0]

        # Spawn pedestrian to the side of the road
        walker_bp = random.choice(bp_lib.filter('walker.pedestrian.*'))

        # Position to the side
        spawn_tf = ahead_wp.transform
        spawn_tf.location += carla.Location(x=0, y=-3, z=0.5)  # 3m to the left

        walker = world.try_spawn_actor(walker_bp, spawn_tf)
        if walker:
            # Add AI controller
            controller_bp = bp_lib.find('controller.ai.walker')
            controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)

            world.tick()

            controller.start()
            # Walk across the road (perpendicular to traffic)
            cross_location = spawn_tf.location + carla.Location(x=0, y=8, z=0)
            controller.go_to_location(cross_location)
            controller.set_max_speed(2.2)  # Faster walking/running (more dangerous)

            spawned_actors.append({
                'actor': walker,
                'type': 'pedestrian',
                'controller': controller
            })

            print(f"     Pedestrian will cross at {ahead_distance:.0f}m")

    return spawned_actors


# ============================================================================
# Traffic Spawning Functions
# ============================================================================
def spawn_traffic_vehicles(world, client, num_vehicles=60):
    """
    Spawn NPC traffic vehicles with Basic Agent autopilot.
    Returns list of vehicle actor IDs.
    """
    bp_lib = world.get_blueprint_library()
    vehicle_bps = [bp for bp in bp_lib.filter('vehicle.*')
                   if int(bp.get_attribute('number_of_wheels')) >= 4]
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    batch = []
    for i in range(min(num_vehicles, len(spawn_points))):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(
                bp.get_attribute('color').recommended_values))
        bp.set_attribute('role_name', 'autopilot')

        # Create spawn command
        batch.append(carla.command.SpawnActor(bp, spawn_points[i]))

    # Spawn all vehicles
    vehicles_list = []
    results = client.apply_batch_sync(batch, True)

    for i, result in enumerate(results):
        if not result.error:
            vehicles_list.append(result.actor_id)

    # Now set autopilot for spawned vehicles using Basic Agent-like behavior
    # We'll use Traffic Manager for simplicity with NPCs
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    for vehicle_id in vehicles_list:
        vehicle = world.get_actor(vehicle_id)
        if vehicle:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            # Randomize speed a bit
            speed_diff = random.randint(-10, 20)
            traffic_manager.vehicle_percentage_speed_difference(vehicle, speed_diff)

    print(f"  \U0001f697 Spawned {len(vehicles_list)}/{num_vehicles} traffic vehicles")
    return vehicles_list


def spawn_pedestrians(world, client, num_pedestrians=50):
    """
    Spawn walking pedestrians with AI controllers.
    Returns (walker_ids, controller_ids).
    """
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    controller_bp = bp_lib.find('controller.ai.walker')

    # Spawn walkers
    walkers = []
    batch = []
    for _ in range(num_pedestrians):
        bp = random.choice(walker_bps)
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')

        loc = world.get_random_location_from_navigation()
        if loc is not None:
            batch.append(carla.command.SpawnActor(bp, carla.Transform(loc)))

    results = client.apply_batch_sync(batch, True)
    walker_ids = []
    for result in results:
        if not result.error:
            walker_ids.append(result.actor_id)

    # Spawn controllers
    batch = []
    for walker_id in walker_ids:
        batch.append(carla.command.SpawnActor(
            controller_bp, carla.Transform(), walker_id))

    results = client.apply_batch_sync(batch, True)
    controller_ids = []
    for result in results:
        if not result.error:
            controller_ids.append(result.actor_id)

    # Wait for navigation mesh to settle
    world.tick()

    # Start walking
    for controller_id in controller_ids:
        controller = world.get_actor(controller_id)
        if controller:
            controller.start()
            dest = world.get_random_location_from_navigation()
            if dest:
                controller.go_to_location(dest)
                controller.set_max_speed(1.0 + random.random() * 1.5)

    print(f"  \U0001f6b6 Spawned {len(walker_ids)}/{num_pedestrians} pedestrians")
    return walker_ids, controller_ids


# ============================================================================
# Collision Detector
# ============================================================================
class CollisionMonitor:
    """Monitor and report collisions."""

    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.collisions = []

        # Attach collision sensor
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        self.sensor.listen(self._on_collision)

    def _on_collision(self, event):
        collision_info = {
            'time': time.time(),
            'other_actor': event.other_actor.type_id,
            'impulse': event.normal_impulse.length(),
        }
        self.collisions.append(collision_info)

        print(f"\n  \U0001f4a5 COLLISION! Hit {event.other_actor.type_id}")
        print(f"     Impulse: {event.normal_impulse.length():.1f} N�s")

    def cleanup(self):
        if self.sensor:
            self.sensor.destroy()


# ============================================================================
# Main Loop
# ============================================================================
def main():
    print("="*70)
    print("CARLA Simple Forward Driver - Accident Prediction Simulator")
    print("="*70)
    print("Using simple lane-following controller (30 km/h target speed)")
    print(f"Traffic: {NUM_TRAFFIC_VEHICLES} vehicles, {NUM_PEDESTRIANS} pedestrians")
    print(f"Weather: Fog={USE_FOG} (density={FOG_DENSITY}), Rain={USE_RAIN} (intensity={RAIN_INTENSITY})")
    print("Challenging scenarios with realistic obstacle behavior")
    print("="*70)

    # Connect to CARLA
    print(f"\nConnecting to CARLA at {CARLA_HOST}:{CARLA_PORT}...")
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(20.0)  # Longer timeout for stability

    # Get current world first (avoid reloading if already on Town01)
    world = client.get_world()
    current_map = world.get_map().name

    if TOWN not in current_map:
        print(f"Current map: {current_map}")
        print(f"Loading {TOWN}...")
        world = client.load_world(TOWN)
    else:
        print(f"Already on {current_map}, using existing world")

    # Synchronous mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS (slower, more stable)
    world.apply_settings(settings)

    # Weather: Fog and Rain
    weather = carla.WeatherParameters(
        cloudiness=90.0,
        precipitation=RAIN_INTENSITY if USE_RAIN else 0.0,
        precipitation_deposits=0.0,
        wind_intensity=20.0,
        sun_altitude_angle=45.0,
        fog_density=FOG_DENSITY if USE_FOG else 0.0,
        fog_distance=0.0,
        fog_falloff=0.2,
        wetness=WETNESS if USE_RAIN else 0.0
    )
    world.set_weather(weather)

    weather_desc = []
    if USE_FOG:
        weather_desc.append(f"Fog (density={FOG_DENSITY})")
    if USE_RAIN:
        weather_desc.append(f"Rain (intensity={RAIN_INTENSITY})")

    print(f"\n\u26c5 Weather: {', '.join(weather_desc) if weather_desc else 'Clear'}")

    # Get map for Basic Agent
    carla_map = world.get_map()

    ego_vehicle = None
    driver = None
    traffic_vehicle_ids = []
    pedestrian_ids = []
    pedestrian_controller_ids = []
    collision_monitor = None
    scenario_actors = []

    try:
        while True:
            # ================================================================
            # Spawn Ego Vehicle
            # ================================================================
            print("\n" + "="*70)
            print("Spawning ego vehicle...")

            bp_lib = world.get_blueprint_library()
            ego_bp = bp_lib.find('vehicle.tesla.model3')

            spawn_points = world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)

            ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
            print(f"  \u2705 Spawned at ({spawn_point.location.x:.0f}, {spawn_point.location.y:.0f})")

            # Setup simple forward driver
            driver = SemiProDriver(ego_vehicle, carla_map)

            # Collision monitor
            collision_monitor = CollisionMonitor(ego_vehicle, world)

            # Let it settle
            for _ in range(20):
                world.tick()

            # ================================================================
            # Spawn Background Traffic
            # ================================================================
            print("\n" + "="*70)
            print("Spawning background traffic...")
            print("="*70)

            traffic_vehicle_ids = spawn_traffic_vehicles(world, client, NUM_TRAFFIC_VEHICLES)
            pedestrian_ids, pedestrian_controller_ids = spawn_pedestrians(world, client, NUM_PEDESTRIANS)

            # Let traffic settle
            for _ in range(10):
                world.tick()

            # ================================================================
            # Spawn Challenging Scenario
            # ================================================================
            scenario_actors = spawn_challenging_scenario(
                world, ego_vehicle
            )

            # ================================================================
            # Main Driving Loop
            # ================================================================
            print(f"\n  \U0001f3c1 Starting scenario (duration: {SCENARIO_DURATION}s)")
            print("     Press Ctrl+C to stop\n")
            print("  \U0001f697 Simple driver is now following the lane...")

            start_time = time.time()
            scenario_triggered = False

            frame_counter = 0
            last_control_print = 0
            while time.time() - start_time < SCENARIO_DURATION:
                # Run agent control - IMPORTANT: This makes the agent drive!
                try:
                    control = driver.run_step()
                    ego_vehicle.apply_control(control)

                    # Debug: Print control values once every 100 frames to verify agent is working
                    if frame_counter - last_control_print > 100:
                        print(f"  [DEBUG] Control: throttle={control.throttle:.2f}, brake={control.brake:.2f}, steer={control.steer:.2f}")
                        last_control_print = frame_counter

                except Exception as e:
                    print(f"Warning: Agent control failed: {e}")
                    import traceback
                    traceback.print_exc()

                # Tick the world
                world.tick()
                frame_counter += 1

                # Trigger sudden brake scenario
                elapsed = time.time()
                for actor_info in scenario_actors:
                    if actor_info.get('type') == 'sudden_braker':
                        if elapsed >= actor_info.get('brake_time', 999) and not scenario_triggered:
                            print("\n  \U0001f6d1 LEAD VEHICLE BRAKING HARD!")
                            actor_info['actor'].enable_constant_velocity(carla.Vector3D(0, 0, 0))
                            actor_info['actor'].apply_control(carla.VehicleControl(
                                throttle=0.0,
                                brake=1.0
                            ))
                            scenario_triggered = True

                # Update spectator to follow ego
                spectator = world.get_spectator()
                tf = ego_vehicle.get_transform()
                spectator.set_transform(carla.Transform(
                    tf.location - tf.get_forward_vector() * 12 + carla.Location(z=6),
                    carla.Rotation(pitch=-20, yaw=tf.rotation.yaw)
                ))

                # Print status every 2 seconds
                if frame_counter % 40 == 0:  # 20 FPS * 2 seconds
                    vel = ego_vehicle.get_velocity()
                    speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                    elapsed_time = time.time() - start_time
                    loc = ego_vehicle.get_location()

                    print(f"  [{elapsed_time:.0f}s] DRIVING | Speed: {speed_kmh:.1f} km/h | Pos: ({loc.x:.0f}, {loc.y:.0f})")

                # Small delay to slow down execution and reduce CPU load
                time.sleep(0.02)  # 20ms delay makes simulation smoother

            # ================================================================
            # Scenario Complete
            # ================================================================
            print("\n" + "="*70)
            print("SCENARIO COMPLETE")
            print("="*70)

            num_collisions = len(collision_monitor.collisions)

            if num_collisions == 0:
                print("  \u2705 No collisions! Basic Agent handled the scenario.")
            elif num_collisions == 1:
                print("  \U0001f4a5 1 collision occurred.")
            else:
                print(f"  \U0001f4a5 {num_collisions} collisions occurred.")

            for i, collision in enumerate(collision_monitor.collisions):
                print(f"     {i+1}. Hit {collision['other_actor']} (impulse: {collision['impulse']:.1f})")

            print("\n  Waiting 3 seconds before restarting...")
            time.sleep(3)

            # Cleanup before restart
            collision_monitor.cleanup()

            # Cleanup scenario actors
            for actor_info in scenario_actors:
                try:
                    actor_info['actor'].destroy()
                    if 'controller' in actor_info:
                        actor_info['controller'].destroy()
                except:
                    pass
            scenario_actors = []

            # Cleanup traffic
            print("\nCleaning up traffic...")

            # Stop pedestrian controllers before destroying
            for ctrl_id in pedestrian_controller_ids:
                try:
                    ctrl = world.get_actor(ctrl_id)
                    if ctrl:
                        ctrl.stop()
                except:
                    pass

            client.apply_batch([carla.command.DestroyActor(x) for x in traffic_vehicle_ids])
            client.apply_batch([carla.command.DestroyActor(x) for x in pedestrian_controller_ids])
            client.apply_batch([carla.command.DestroyActor(x) for x in pedestrian_ids])
            traffic_vehicle_ids = []
            pedestrian_ids = []
            pedestrian_controller_ids = []

            if ego_vehicle:
                ego_vehicle.destroy()
                ego_vehicle = None

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n  \u26a0\ufe0f  Interrupted by user")

    finally:
        print("\nCleaning up...")

        # Cleanup collision monitor
        if collision_monitor:
            collision_monitor.cleanup()

        # Cleanup scenario actors
        for actor_info in scenario_actors:
            try:
                actor_info['actor'].destroy()
                if 'controller' in actor_info:
                    actor_info['controller'].destroy()
            except:
                pass

        # Cleanup traffic and pedestrians
        try:
            # Stop pedestrian controllers first
            if pedestrian_controller_ids:
                print("  Stopping pedestrian controllers...")
                for ctrl_id in pedestrian_controller_ids:
                    try:
                        ctrl = world.get_actor(ctrl_id)
                        if ctrl:
                            ctrl.stop()
                    except:
                        pass

            if traffic_vehicle_ids:
                print("  Destroying traffic vehicles...")
                client.apply_batch([carla.command.DestroyActor(x) for x in traffic_vehicle_ids])
            if pedestrian_controller_ids:
                print("  Destroying pedestrian controllers...")
                client.apply_batch([carla.command.DestroyActor(x) for x in pedestrian_controller_ids])
            if pedestrian_ids:
                print("  Destroying pedestrians...")
                client.apply_batch([carla.command.DestroyActor(x) for x in pedestrian_ids])
        except:
            pass

        # Cleanup ego vehicle
        if ego_vehicle:
            try:
                ego_vehicle.destroy()
            except:
                pass

        # Restore settings
        try:
            world.apply_settings(original_settings)
        except:
            pass

        print("  \u2705 Cleanup complete")
        print("\n" + "="*70)


if __name__ == '__main__':
    main()
