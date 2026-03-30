#!/usr/bin/env python3
"""
Verification script for data_collector_v2.py coordinate system correctness.

Run this AFTER collecting a small dataset (2-3 minutes, ~200 frames).
It checks that:
  1. Target points are ahead of the ego vehicle (positive X in ego frame)
  2. Route waypoints form a sensible forward-facing path
  3. Steering angle sign matches autopilot steer direction
  4. Theta value is consistent with ego_matrix yaw (no spurious -90° offset)
  5. Generates visual plots to manually inspect

Usage:
  python verify_coordinate_system.py /path/to/dataset/route_00000
"""

import os
import sys
import gzip
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def load_measurement(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def yaw_from_ego_matrix(ego_matrix):
    """Extract yaw (radians) from the 4x4 ego_matrix saved in measurements."""
    # ego_matrix[0][0] = cos(yaw), ego_matrix[1][0] = sin(yaw) in CARLA's convention
    return math.atan2(ego_matrix[1][0], ego_matrix[0][0])


def normalize_angle(x):
    x = x % (2 * math.pi)
    if x > math.pi:
        x -= 2 * math.pi
    return x


def inverse_conversion_2d(point, translation, yaw):
    """Reference implementation matching transfuser_utils.py"""
    rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw),  np.cos(yaw)]])
    return rot.T @ (np.array(point) - np.array(translation))


def check_single_frame(m, frame_idx):
    """Run all checks on a single measurement. Returns dict of check results."""
    results = {}

    # --- Check 1: theta vs ego_matrix yaw ---
    ego_matrix = np.array(m['ego_matrix'])
    yaw_from_matrix = yaw_from_ego_matrix(ego_matrix)
    saved_theta = m['theta']

    # What theta SHOULD be: just the normalized yaw
    correct_theta = normalize_angle(yaw_from_matrix)
    # What theta would be with the -90° bug:
    buggy_theta = normalize_angle(yaw_from_matrix - np.deg2rad(90.0))

    theta_diff_correct = abs(normalize_angle(saved_theta - correct_theta))
    theta_diff_buggy = abs(normalize_angle(saved_theta - buggy_theta))

    results['theta_matches_yaw'] = theta_diff_correct < 0.01  # Within ~0.5°
    results['theta_has_90_bug'] = theta_diff_buggy < 0.01
    results['theta_diff_deg'] = math.degrees(theta_diff_correct)
    results['saved_theta_deg'] = math.degrees(saved_theta)
    results['correct_theta_deg'] = math.degrees(correct_theta)

    # --- Check 2: Target point is ahead (positive X in ego frame) ---
    tp = m['target_point']
    results['target_forward'] = tp[0]  # Should be positive (= ahead)
    results['target_lateral'] = tp[1]  # Left/right offset
    results['target_is_ahead'] = tp[0] > 0

    # --- Check 3: Verify target_point by recomputing from global data ---
    # We can cross-check by taking pos_global, theta, and reverse-engineering
    # what the global target point was, then re-doing inverse_conversion_2d
    pos = np.array(m['pos_global'][:2])

    # --- Check 4: Route waypoints should generally be ahead ---
    route = np.array(m.get('route', []))
    if len(route) > 0:
        route_forward_pct = np.mean(route[:, 0] > 0) * 100  # % of points with positive X
        results['route_forward_pct'] = route_forward_pct
        results['route_centroid'] = [float(np.mean(route[:, 0])), float(np.mean(route[:, 1]))]
    else:
        results['route_forward_pct'] = -1
        results['route_centroid'] = [0, 0]

    # --- Check 5: Angle-steer consistency ---
    # When autopilot steers left (steer < 0), target should be to the left
    # angle is normalized to [-1, 1] where positive = right turn
    results['angle'] = m['angle']
    results['steer'] = m['steer']
    # They should have the same sign (both positive = right, both negative = left)
    if abs(m['steer']) > 0.05 and abs(m['angle']) > 0.01:
        results['steer_angle_consistent'] = (m['steer'] * m['angle']) >= 0
    else:
        results['steer_angle_consistent'] = None  # Going straight, can't check

    # --- Check 6: Speed sanity ---
    results['speed'] = m['speed']
    results['target_speed'] = m['target_speed']

    return results


def plot_ego_frame(measurements, output_path, title_suffix=""):
    """Plot several frames showing ego vehicle + target + route in ego-local frame."""
    # Pick frames evenly distributed through the dataset
    n = len(measurements)
    indices = np.linspace(0, n - 1, min(9, n), dtype=int)

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(f'Ego-Local Frame Visualization{title_suffix}\n'
                 '+X = Forward, +Y = Left\n'
                 'Green = route, Red star = target point, Blue arrow = ego forward',
                 fontsize=14)

    for ax_idx, frame_idx in enumerate(indices):
        ax = axes[ax_idx // 3][ax_idx % 3]
        m = measurements[frame_idx]

        tp = m['target_point']
        route = np.array(m.get('route', []))
        aim = m.get('aim_wp', [0, 0])

        # Plot ego vehicle as blue arrow at origin pointing in +X direction
        ax.annotate('', xy=(2, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        ax.plot(0, 0, 'bs', markersize=10, label='Ego')

        # Plot route
        if len(route) > 0:
            ax.plot(route[:, 0], route[:, 1], 'g.-', alpha=0.7, label='Route')
            for i, r in enumerate(route):
                ax.annotate(str(i), (r[0], r[1]), fontsize=6, alpha=0.5)

        # Plot target point
        ax.plot(tp[0], tp[1], 'r*', markersize=15, label='Target Point')

        # Plot aim waypoint
        ax.plot(aim[0], aim[1], 'm^', markersize=10, label='Aim WP')

        # Info text
        info = (f"Frame {frame_idx}\n"
                f"θ={m['theta']:.2f}rad ({math.degrees(m['theta']):.1f}°)\n"
                f"speed={m['speed']:.1f} m/s\n"
                f"steer={m['steer']:.3f}\n"
                f"angle={m['angle']:.3f}\n"
                f"cmd={m.get('command', '?')}")
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlim(-30, 50)
        ax.set_ylim(-30, 30)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('X (Forward →)')
        ax.set_ylabel('Y (← Left)')
        if ax_idx == 0:
            ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {output_path}")


def plot_global_trajectory(measurements, output_path):
    """Plot the global trajectory with heading arrows to sanity-check theta."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    positions = np.array([m['pos_global'][:2] for m in measurements])
    thetas = np.array([m['theta'] for m in measurements])

    ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.5, label='Trajectory')
    ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=12, label='Start')
    ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=12, label='End')

    # Draw heading arrows every N frames
    N = max(1, len(measurements) // 30)
    for i in range(0, len(measurements), N):
        dx = 3 * np.cos(thetas[i])
        dy = 3 * np.sin(thetas[i])
        ax.annotate('', xy=(positions[i, 0] + dx, positions[i, 1] + dy),
                    xytext=(positions[i, 0], positions[i, 1]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # Also draw what the CORRECT theta arrows would look like (from ego_matrix)
    for i in range(0, len(measurements), N):
        correct_yaw = yaw_from_ego_matrix(np.array(measurements[i]['ego_matrix']))
        dx = 3 * np.cos(correct_yaw)
        dy = 3 * np.sin(correct_yaw)
        ax.annotate('', xy=(positions[i, 0] + dx, positions[i, 1] + dy),
                    xytext=(positions[i, 0], positions[i, 1]),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5, alpha=0.6))

    red_patch = mpatches.Patch(color='red', label='Saved theta (heading arrows)')
    green_patch = mpatches.Patch(color='green', label='Correct theta from ego_matrix')
    ax.legend(handles=[red_patch, green_patch], fontsize=10)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Global X')
    ax.set_ylabel('Global Y')
    ax.set_title('Global Trajectory + Heading Arrows\n'
                 'Red = saved theta, Green = ego_matrix yaw\n'
                 'If they match → theta is correct. If 90° off → bug is present.')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {output_path}")


def plot_angle_vs_steer(measurements, output_path):
    """Scatter plot of angle vs steer to check directional consistency."""
    angles = [m['angle'] for m in measurements]
    steers = [m['steer'] for m in measurements]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(angles, steers, alpha=0.3, s=5)
    ax.set_xlabel('Saved Angle (from target)')
    ax.set_ylabel('Autopilot Steer')
    ax.set_title('Angle vs Steer Correlation\n'
                 'Should show a positive correlation (same sign)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    # Highlight quadrants
    ax.fill_between([-1, 0], -1, 0, alpha=0.05, color='green')  # Both negative = left turn OK
    ax.fill_between([0, 1], 0, 1, alpha=0.05, color='green')    # Both positive = right turn OK
    ax.fill_between([-1, 0], 0, 1, alpha=0.05, color='red')     # Opposite signs = BAD
    ax.fill_between([0, 1], -1, 0, alpha=0.05, color='red')     # Opposite signs = BAD

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_coordinate_system.py /path/to/dataset/route_00000")
        print("       python verify_coordinate_system.py /path/to/dataset  (checks all routes)")
        sys.exit(1)

    dataset_path = Path(sys.argv[1])

    # Find route directories
    if (dataset_path / 'measurements').is_dir():
        route_dirs = [dataset_path]
    else:
        route_dirs = sorted([d for d in dataset_path.iterdir()
                            if d.is_dir() and (d / 'measurements').is_dir()])

    if not route_dirs:
        print(f"ERROR: No route directories with 'measurements/' found in {dataset_path}")
        sys.exit(1)

    for route_dir in route_dirs:
        print(f"\n{'='*70}")
        print(f"Checking: {route_dir.name}")
        print(f"{'='*70}")

        meas_dir = route_dir / 'measurements'
        meas_files = sorted(meas_dir.glob('*.json.gz'))

        if not meas_files:
            print(f"  WARNING: No measurement files found in {meas_dir}")
            continue

        print(f"  Found {len(meas_files)} measurement files")

        # Load all measurements
        measurements = []
        for mf in meas_files:
            try:
                measurements.append(load_measurement(mf))
            except Exception as e:
                print(f"  WARNING: Failed to load {mf.name}: {e}")

        if not measurements:
            continue

        # Run checks
        theta_correct_count = 0
        theta_buggy_count = 0
        target_ahead_count = 0
        steer_consistent_count = 0
        steer_check_total = 0

        all_results = []
        for i, m in enumerate(measurements):
            r = check_single_frame(m, i)
            all_results.append(r)

            if r['theta_matches_yaw']:
                theta_correct_count += 1
            if r['theta_has_90_bug']:
                theta_buggy_count += 1
            if r['target_is_ahead']:
                target_ahead_count += 1
            if r['steer_angle_consistent'] is not None:
                steer_check_total += 1
                if r['steer_angle_consistent']:
                    steer_consistent_count += 1

        n = len(measurements)

        # Print results
        print(f"\n  --- THETA CHECK ---")
        print(f"  Theta matches raw yaw (CORRECT):  {theta_correct_count}/{n} "
              f"({100*theta_correct_count/n:.1f}%)")
        print(f"  Theta matches yaw-90° (BUGGY):    {theta_buggy_count}/{n} "
              f"({100*theta_buggy_count/n:.1f}%)")

        if theta_correct_count > theta_buggy_count:
            print(f"  ✅ THETA IS CORRECT (no -90° offset)")
        elif theta_buggy_count > theta_correct_count:
            print(f"  ❌ THETA HAS THE -90° BUG — FIX data_collector_v2.py line 824!")
        else:
            print(f"  ⚠️  INCONCLUSIVE — check the plots")

        print(f"\n  --- TARGET POINT CHECK ---")
        print(f"  Target ahead (X > 0):  {target_ahead_count}/{n} "
              f"({100*target_ahead_count/n:.1f}%)")
        if target_ahead_count / n > 0.9:
            print(f"  ✅ TARGET POINTS ARE AHEAD OF EGO (correct)")
        else:
            print(f"  ❌ MANY TARGET POINTS ARE BEHIND EGO — coordinate transform is wrong!")

        avg_forward = np.mean([r['target_forward'] for r in all_results])
        avg_lateral = np.mean([r['target_lateral'] for r in all_results])
        print(f"  Average target: forward={avg_forward:.1f}m, lateral={avg_lateral:.1f}m")

        print(f"\n  --- ROUTE CHECK ---")
        route_fwd_pcts = [r['route_forward_pct'] for r in all_results if r['route_forward_pct'] >= 0]
        if route_fwd_pcts:
            avg_route_fwd = np.mean(route_fwd_pcts)
            print(f"  Route points ahead of ego:  {avg_route_fwd:.1f}% average")
            if avg_route_fwd > 80:
                print(f"  ✅ ROUTE WAYPOINTS ARE MOSTLY FORWARD (correct)")
            else:
                print(f"  ❌ ROUTE WAYPOINTS ARE NOT FORWARD — coordinate bug!")

        print(f"\n  --- STEER/ANGLE CONSISTENCY ---")
        if steer_check_total > 0:
            pct = 100 * steer_consistent_count / steer_check_total
            print(f"  Steer-angle same sign:  {steer_consistent_count}/{steer_check_total} "
                  f"({pct:.1f}%)")
            if pct > 70:
                print(f"  ✅ STEERING AND ANGLE ARE CONSISTENT")
            else:
                print(f"  ❌ STEERING AND ANGLE DISAGREE — angle calculation is wrong!")
        else:
            print(f"  ⚠️  Car went mostly straight, insufficient turning data to check")

        print(f"\n  --- AUGMENTATION CHECK ---")
        aug_translations = [m['augmentation_translation'] for m in measurements]
        aug_rotations = [m['augmentation_rotation'] for m in measurements]
        if all(a == 0.0 for a in aug_translations) and all(a == 0.0 for a in aug_rotations):
            print(f"  ⚠️  ALL augmentation values are 0.0 — augmented images are identical to normal!")
            print(f"      Training with config.augment=1 will NOT help with lane recovery.")
        else:
            print(f"  ✅ Augmentation values are non-zero")
            print(f"     Translation range: [{min(aug_translations):.2f}, {max(aug_translations):.2f}]m")
            print(f"     Rotation range: [{min(aug_rotations):.2f}, {max(aug_rotations):.2f}]°")

        # Generate plots
        output_dir = route_dir / 'verification_plots'
        output_dir.mkdir(exist_ok=True)

        print(f"\n  Generating verification plots...")
        plot_ego_frame(measurements, output_dir / 'ego_frame_samples.png')
        plot_global_trajectory(measurements, output_dir / 'global_trajectory.png')
        plot_angle_vs_steer(measurements, output_dir / 'angle_vs_steer.png')

        print(f"\n  📊 Check the plots in: {output_dir}/")
        print(f"     - ego_frame_samples.png:  Target + route in ego frame (should be AHEAD)")
        print(f"     - global_trajectory.png:  Red vs green arrows (should OVERLAP)")
        print(f"     - angle_vs_steer.png:     Points in GREEN quadrants = correct")

    print(f"\n{'='*70}")
    print(f"VERIFICATION COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
