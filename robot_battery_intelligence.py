
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# ROBOT MOVEMENT & BATTERY INTELLIGENCE DASHBOARD
# Technologies: Python, NumPy, Matplotlib, OpenCV
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

START = np.array([5.0, 10.0])
GOAL = np.array([95.0, 90.0])

INITIAL_BATTERY = 100.0

# These are simulation assumptions, not real robot measurements.
ENERGY_PER_DISTANCE = 0.22
TURN_ENERGY = 0.65


def create_routes():
    """Create three simulated routes between the same endpoints."""

    direct = np.array([
        [5, 10],
        [25, 28],
        [45, 46],
        [65, 64],
        [80, 78],
        [95, 90]
    ], dtype=float)

    zigzag = np.array([
        [5, 10],
        [20, 35],
        [35, 15],
        [50, 48],
        [65, 30],
        [80, 65],
        [95, 90]
    ], dtype=float)

    detour = np.array([
        [5, 10],
        [15, 10],
        [15, 35],
        [40, 35],
        [40, 60],
        [65, 60],
        [65, 80],
        [80, 80],
        [95, 90]
    ], dtype=float)

    return {
        "Direct Route": direct,
        "Zigzag Route": zigzag,
        "Detour Route": detour
    }


def analyze_route(points):
    """Calculate distance, turns, battery use and position history."""

    # NumPy subtracts consecutive positions to calculate movement.
    movements = np.diff(points, axis=0)

    # Distance for every movement segment.
    segment_distances = np.linalg.norm(movements, axis=1)

    total_distance = float(np.sum(segment_distances))

    # Calculate changes in movement direction.
    angles = np.arctan2(movements[:, 1], movements[:, 0])
    angle_changes = np.diff(angles)

    # Handle angle wraparound, such as +179 to -179 degrees.
    angle_changes = np.arctan2(
        np.sin(angle_changes),
        np.cos(angle_changes)
    )

    turn_count = int(np.sum(np.abs(angle_changes) > math.radians(10)))

    # Illustrative energy model:
    # distance costs energy; direction changes cost extra energy.
    energy_used = (
        total_distance * ENERGY_PER_DISTANCE
        + turn_count * TURN_ENERGY
    )

    # One battery percentage point represents one energy unit
    # in this simplified simulation.
    battery_remaining = max(
        0.0, INITIAL_BATTERY - energy_used
    )

    # Calculate cumulative distance and battery history.
    cumulative_distance = np.concatenate([
        [0.0], np.cumsum(segment_distances)
    ])

    segment_energy = (
        segment_distances * ENERGY_PER_DISTANCE
    )

    # Charge a turn cost at each detected direction change.
    turn_costs = np.zeros(len(segment_distances))

    for i in range(1, len(segment_distances)):
        if abs(angle_changes[i - 1]) > math.radians(10):
            turn_costs[i] = TURN_ENERGY

    energy_per_segment = segment_energy + turn_costs

    cumulative_energy = np.concatenate([
        [0.0], np.cumsum(energy_per_segment)
    ])

    battery_history = np.maximum(
        0.0, INITIAL_BATTERY - cumulative_energy
    )

    return {
        "points": points,
        "distance": total_distance,
        "turns": turn_count,
        "energy": float(energy_used),
        "battery": battery_remaining,
        "cumulative_distance": cumulative_distance,
        "battery_history": battery_history
    }


def save_csv(results):
    """Export all route statistics and position history."""

    filename = OUTPUT_DIR / "robot_data.csv"

    with filename.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Route", "Step", "X", "Y",
            "Cumulative_Distance",
            "Battery_Percent"
        ])

        for name, result in results.items():
            for i, point in enumerate(result["points"]):
                writer.writerow([
                    name,
                    i,
                    round(point[0], 2),
                    round(point[1], 2),
                    round(float(result["cumulative_distance"][i]), 2),
                    round(float(result["battery_history"][i]), 2)
                ])

    print(f"\nCSV data saved: {filename}")


def save_opencv_map(results, best_route):
    """Draw the selected robot route using OpenCV."""

    width, height = 900, 650
    margin = 55

    # OpenCV images use BGR colors.
    image = np.full((height, width, 3), 248, dtype=np.uint8)

    def to_pixel(point):
        x = margin + point[0] / 100 * (width - 2 * margin)
        y = height - margin - point[1] / 100 * (height - 2 * margin)
        return [int(x), int(y)]

    # Draw grid lines.
    for value in range(0, 101, 10):
        x = int(margin + value / 100 * (width - 2 * margin))
        y = int(height - margin - value / 100 * (height - 2 * margin))

        cv2.line(image, (x, margin), (x, height - margin),
                 (225, 225, 225), 1)

        cv2.line(image, (margin, y), (width - margin, y),
                 (225, 225, 225), 1)

    # Draw the selected route.
    points = np.array(
        [to_pixel(p) for p in results[best_route]["points"]],
        dtype=np.int32
    ).reshape((-1, 1, 2))

    cv2.polylines(image, [points], False, (210, 105, 30), 4)

    start_pixel = tuple(to_pixel(START))
    goal_pixel = tuple(to_pixel(GOAL))

    cv2.circle(image, start_pixel, 10, (40, 160, 40), -1)
    cv2.circle(image, goal_pixel, 10, (40, 40, 220), -1)

    cv2.putText(image, "START", (start_pixel[0] + 12, start_pixel[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 120, 40), 2)

    cv2.putText(image, "GOAL", (goal_pixel[0] - 65, goal_pixel[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 200), 2)

    cv2.putText(image, "ROBOT ROUTE MAP",
                (margin, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (45, 45, 45), 2)

    filename = OUTPUT_DIR / "robot_route_map.png"

    if not cv2.imwrite(str(filename), image):
        raise RuntimeError("OpenCV could not save the route map.")

    print(f"OpenCV map saved: {filename}")


def create_dashboard(results, best_route):
    """Create a four-panel Matplotlib dashboard."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "ROBOT MOVEMENT & BATTERY INTELLIGENCE",
        fontsize=16,
        fontweight="bold"
    )

    # Panel 1: Compare routes.
    ax = axes[0, 0]

    for name, result in results.items():
        points = result["points"]
        ax.plot(points[:, 0], points[:, 1], marker="o",
                markersize=4, linewidth=2, label=name)

    ax.scatter(*START, marker="s", s=90, label="Start")
    ax.scatter(*GOAL, marker="*", s=150, label="Goal")
    ax.set_title("Robot Routes")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 2: Battery history.
    ax = axes[0, 1]

    for name, result in results.items():
        ax.plot(
            result["cumulative_distance"],
            result["battery_history"],
            marker="o",
            label=name
        )

    ax.axhline(20, linestyle="--", label="20% reference")
    ax.set_title("Battery Level During Movement")
    ax.set_xlabel("Cumulative distance (map units)")
    ax.set_ylabel("Remaining battery (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 3: Total distance comparison.
    ax = axes[1, 0]
    names = list(results.keys())
    distances = [results[n]["distance"] for n in names]

    bars = ax.bar(names, distances)
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_title("Total Distance Comparison")
    ax.set_ylabel("Distance (map units)")
    ax.tick_params(axis="x", rotation=10)
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: Energy consumption.
    ax = axes[1, 1]
    energy = [results[n]["energy"] for n in names]

    bars = ax.bar(names, energy)
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_title("Estimated Energy Consumption")
    ax.set_ylabel("Energy units")
    ax.tick_params(axis="x", rotation=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    filename = OUTPUT_DIR / "robot_dashboard.png"
    fig.savefig(filename, dpi=160, bbox_inches="tight")

    print(f"Dashboard saved: {filename}")

    plt.show()
    plt.close(fig)


def main():
    print("=" * 58)
    print(" ROBOT MOVEMENT & BATTERY INTELLIGENCE DASHBOARD")
    print("=" * 58)

    routes = create_routes()
    results = {
        name: analyze_route(points)
        for name, points in routes.items()
    }

    print("\nROUTE ANALYSIS")
    print("-" * 58)

    for name, result in results.items():
        print(f"\n{name}")
        print(f"  Total distance:   {result['distance']:.2f} units")
        print(f"  Direction changes: {result['turns']}")
        print(f"  Energy used:      {result['energy']:.2f} units")
        print(f"  Battery left:     {result['battery']:.2f}%")

    # Lowest estimated energy consumption wins.
    best_route = min(results, key=lambda n: results[n]["energy"])
    best = results[best_route]

    print("\n" + "=" * 58)
    print("SIMULATION RECOMMENDATION")
    print("=" * 58)
    print(f"Lowest-energy route: {best_route}")
    print(f"Estimated energy:    {best['energy']:.2f} units")
    print(f"Estimated battery:   {best['battery']:.2f}%")

    print("\nNOTE: These are simulated values, not real robot measurements.")
    print("The shortest route may not always be the safest or best route.")

    save_csv(results)
    save_opencv_map(results, best_route)
    create_dashboard(results, best_route)

    print("\nAll project outputs are in the 'output' folder.")
    print("Open robot_dashboard.png and robot_route_map.png to view them.")


if __name__ == "__main__":
    main()
