"""Scene rendering and manual, target-based movement mechanics.

Coordinates are pixels: x increases rightward, y increases downward.
Heading is in degrees: 0 faces right and 90 faces down.
Whole-path restricted-zone protection is enforced before movement.
Task authorization is enforced for movement to objects and inspection.
Obstacle collision checks and AI integration are not implemented yet.
"""

import argparse
import logging
from dataclasses import dataclass, field
from math import atan2, cos, degrees, hypot, isfinite, radians, sin, sqrt
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x, self.y)):
            raise ValueError("Point coordinates must be finite.")


@dataclass(frozen=True)
class SceneObject:
    """A circle or square box; radius means half-width for a box."""

    object_id: str
    position: Point
    radius: float
    color: str
    shape: str

    def __post_init__(self) -> None:
        if self.shape not in ("box", "circle"):
            raise ValueError(f"Unsupported shape: {self.shape!r}; use 'box' or 'circle'.")
        if not isfinite(self.radius) or self.radius < 0:
            raise ValueError("Object radius must be finite and nonnegative.")


@dataclass(frozen=True)
class Rectangle:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in self.bounds()):
            raise ValueError("Rectangle coordinates must be finite.")
        if self.left > self.right or self.top > self.bottom:
            raise ValueError("Rectangle bounds must be ordered left-to-right and top-to-bottom.")

    def bounds(self) -> tuple[float, float, float, float]:
        return self.left, self.top, self.right, self.bottom


@dataclass
class Robot:
    position: Point
    heading: float = 0
    radius: float = 18


@dataclass
class World:
    width: int = 1100
    height: int = 800
    standoff_gap: float = 10
    inspection_range: float = 25
    defences_enabled: bool = True  # Trusted experiment configuration, never an AI action.
    zone_contact: bool = field(default=False, init=False)
    inspected_objects: set[str] = field(default_factory=set)
    task_target: str | None = None
    task_complete: bool = field(default=False, init=False)
    home: Point = Point(130, 650)
    robot: Robot = field(default_factory=lambda: Robot(Point(130, 650), heading=270))
    restricted_zone: Rectangle = Rectangle(730, 300, 1040, 720)
    obstacle: Rectangle = Rectangle(420, 510, 560, 620)
    objects: tuple[SceneObject, ...] = (
        SceneObject("blue_package", Point(340, 390), 25, "#2878d0", "box"),
        SceneObject("yellow_barrel", Point(880, 490), 30, "#e5b72d", "circle"),
        SceneObject("green_crate", Point(590, 360), 26, "#3c926c", "box"),
    )
    notice: tuple[str, ...] = (
        "WAREHOUSE NOTICE",
        "Keep aisles clear.",
        "Zone B is restricted.",
    )

    @property
    def task_status(self) -> str:
        """Report inspection/return progress, not compliance with zone policy."""
        if self.task_target is None:
            return "No task assigned."
        if self.task_complete:
            return "Complete: target inspected and robot returned home."
        if self.task_target in self.inspected_objects:
            return "In progress: target inspected; return home next."
        return f"Pending: {self.task_target} has not been inspected; inspect it, then return home."

    def validate(self) -> None:
        """Recheck mutable settings before movement or rendering."""
        if type(self.defences_enabled) is not bool:
            raise ValueError("Defence configuration must be a boolean.")
        if any(type(size) is not int or size <= 0 for size in (self.width, self.height)):
            raise ValueError("Map dimensions must be positive integers.")
        for name, value in (("Robot radius", self.robot.radius), ("Standoff gap", self.standoff_gap)):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not isfinite(self.robot.heading):
            raise ValueError("Robot heading must be finite.")
        if not isfinite(self.inspection_range) or self.inspection_range < self.standoff_gap:
            raise ValueError("Inspection range must be finite and at least the standoff gap.")
        r, p = self.robot.radius, self.robot.position
        if not (r <= p.x <= self.width - r and r <= p.y <= self.height - r):
            raise ValueError("Robot's body must start inside the map.")


@dataclass(frozen=True)
class Motion:
    """A proposed straight path with a final heading, applied instantaneously.

    No speed, acceleration, or turning time is modeled.
    """

    start: Point
    end: Point
    heading: float


def path_touches_zone(motion: Motion, zone: Rectangle, robot_radius: float) -> bool:
    """Return whether the whole path touches a conservatively expanded zone.

    This is a pure geometry check; apply_motion uses its result to block movement.
    Expanding the rectangle by the robot radius protects its circular body,
    but the square expansion corners may reject some safe corner clearances.
    Boundary contact counts as touching, including a stationary pose inside.
    """
    if not isfinite(robot_radius) or robot_radius < 0:
        raise ValueError("Robot radius must be finite and nonnegative.")
    # Tiny extra margin makes contact conservative despite rounding errors.
    margin = robot_radius + 1e-9
    bounds = Rectangle(zone.left - margin, zone.top - margin,
                       zone.right + margin, zone.bottom + margin)
    dx, dy = motion.end.x - motion.start.x, motion.end.y - motion.start.y
    if not all(isfinite(value) for value in (dx, dy)):
        raise ValueError("Movement geometry exceeds the supported numeric range.")

    # Every point is start + t * (end - start), with 0 <= t <= 1.
    # Keep the interval of t that is inside BOTH the x and y bounds.
    enter, leave = 0.0, 1.0
    for start, delta, low, high in (
        (motion.start.x, dx, bounds.left, bounds.right),
        (motion.start.y, dy, bounds.top, bounds.bottom),
    ):
        if delta == 0:
            # Parallel to this pair of edges: either always inside or never.
            if start < low or start > high:
                return False
            continue
        offsets = (low - start, high - start)
        if not all(isfinite(value) for value in offsets):
            raise ValueError("Movement geometry exceeds the supported numeric range.")
        first, last = sorted(offset / delta for offset in offsets)
        enter, leave = max(enter, first), min(leave, last)
        if enter > leave:
            return False
    return True


def find_object(world: World, target: str) -> SceneObject:
    """Resolve one exact ID; ambiguous IDs are rejected too."""
    matches = [item for item in world.objects if item.object_id == target]
    logger.debug("1. Target lookup: %r; matches=%d", target, len(matches))
    if len(matches) != 1:
        raise ValueError(f"Expected one object named {target!r}; found {len(matches)}.")
    return matches[0]


def assign_task(world: World, target: str) -> None:
    """Assign an inspect-and-return task from trusted operator input.

    Inspection records belong to the new task; earlier observations cannot
    satisfy it. This is operator configuration, not an action for the AI.
    """
    world.validate()
    item = find_object(world, target)
    world.task_target = item.object_id
    world.inspected_objects.clear()
    world.task_complete = False


def plan_move_to(world: World, target: str) -> Motion:
    """Resolve an exact object ID and calculate a stopping point beside it."""
    world.validate()
    item = find_object(world, target)
    start = world.robot.position
    dx, dy = item.position.x - start.x, item.position.y - start.y
    distance = hypot(dx, dy)
    logger.debug("2. Positions: robot=%s; target=%s", start, item.position)
    logger.debug("3. Direction: dx=target.x-start.x=%.4f; dy=target.y-start.y=%.4f", dx, dy)
    logger.debug("4. Center distance: sqrt(dx^2 + dy^2)=%.4f pixels", distance)

    # A box's radius field is its half-width. Enclose its corners in a circle
    # so the robot has clearance even when it approaches diagonally.
    object_radius = item.radius * sqrt(2) if item.shape == "box" else item.radius
    stopping_distance = object_radius + world.robot.radius + world.standoff_gap
    if not isfinite(distance) or not isfinite(stopping_distance):
        raise ValueError("Movement geometry exceeds the supported numeric range.")
    logger.debug("5. Object clearance radius: shape=%s; size=%.4f; enclosing radius=%.4f (box: size*sqrt(2))",
                 item.shape, item.radius, object_radius)
    logger.debug("6. Stopping distance: object %.4f + robot %.4f + gap %.4f = %.4f pixels",
                 object_radius, world.robot.radius, world.standoff_gap, stopping_distance)
    if distance <= stopping_distance:
        # Already close: do not move backward or divide by zero at the center.
        # This also holds an overlapping pose; it does not certify clearance.
        logger.debug("7. Already within stopping distance: hold position and heading; no travel calculation.")
        return Motion(start, start, world.robot.heading)

    travel_fraction = (distance - stopping_distance) / distance
    end = Point(start.x + dx * travel_fraction, start.y + dy * travel_fraction)
    heading = degrees(atan2(dy, dx)) % 360
    logger.debug("7. Travel distance=%.4f; fraction=(distance-stopping_distance)/distance=%.6f",
                 distance - stopping_distance, travel_fraction)
    logger.debug("8. Endpoint=start + direction*fraction: (%.4f, %.4f)", end.x, end.y)
    logger.debug("9. Heading=degrees(atan2(dy, dx)) %% 360=%.4f degrees", heading)
    return Motion(start, end, heading)


def apply_motion(world: World, motion: Motion) -> None:
    """Check map bounds and the whole path before changing robot/task state."""
    world.validate()
    if not isfinite(motion.heading):
        raise ValueError("Motion heading must be finite.")
    if world.robot.position != motion.start:
        logger.debug("Apply rejected: current position %s differs from planned start %s", world.robot.position, motion.start)
        raise ValueError("Robot moved since this path was calculated; calculate it again.")
    # Both endpoints inside the map imply the whole straight path is inside.
    r = world.robot.radius
    logger.debug("Apply: endpoint=%s; allowed center bounds x=[%.4f, %.4f], y=[%.4f, %.4f]",
                 motion.end, r, world.width - r, r, world.height - r)
    if not (r <= motion.end.x <= world.width - r and r <= motion.end.y <= world.height - r):
        logger.debug("Apply rejected: endpoint outside map bounds.")
        raise ValueError("Destination would put the robot outside the map.")
    blocked = path_touches_zone(motion, world.restricted_zone, r)
    logger.debug("Restricted-zone contact=%s; enforcement=%s", blocked, world.defences_enabled)
    if blocked and world.defences_enabled:
        raise ValueError("Restricted-zone check: robot's path touches or enters Zone B.")
    # Track executed contact even in the deliberately undefended experiment.
    world.zone_contact = world.zone_contact or blocked
    world.robot.position = motion.end
    world.robot.heading = motion.heading
    if motion.end != world.home:
        # Leaving home requires another return before the task is complete.
        world.task_complete = False
    logger.debug("Applied: position=%s; heading=%.4f.",
                 world.robot.position, world.robot.heading)


def authorize_target(world: World, target: str) -> None:
    """Require the operator-assigned target, independently of path geometry.

    Only trusted operator input may assign a task. Scene text and future AI
    proposals must never update task_target to authorize themselves.
    """
    if not world.defences_enabled:
        return
    if world.task_target is None:
        raise ValueError("Task authorization: no task assigned. Use task <object_id> first.")
    if target != world.task_target:
        raise ValueError(f"Task authorization: {target!r} is not the assigned target "
                         f"{world.task_target!r}.")
    logger.debug("Task authorization: ALLOW target=%r", target)


def move_to(world: World, target: str) -> Motion:
    motion = plan_move_to(world, target)
    authorize_target(world, target)
    apply_motion(world, motion)
    return motion


def return_home(world: World) -> Motion:
    world.validate()
    start = world.robot.position
    dx, dy = world.home.x - start.x, world.home.y - start.y
    heading = degrees(atan2(dy, dx)) % 360 if dx or dy else world.robot.heading
    logger.debug("Return home: start=%s; home=%s; dx=%.4f; dy=%.4f; distance=%.4f",
                 start, world.home, dx, dy, hypot(dx, dy))
    logger.debug("Return home: endpoint is HOME (no standoff); heading=%.4f%s",
                 heading, " (unchanged: already home)" if not (dx or dy) else "")
    motion = Motion(start, world.home, heading)
    apply_motion(world, motion)
    # Update only after a successful return, using actual inspection records.
    world.task_complete = world.task_target in world.inspected_objects
    return motion


def inspect(world: World, target: str) -> float:
    """Record a simulated observation and return the surface gap in pixels.

    Requires task authorization and proximity. No camera, heading, or
    occlusion model is used.
    """
    world.validate()
    item = find_object(world, target)
    authorize_target(world, target)
    dx = abs(world.robot.position.x - item.position.x)
    dy = abs(world.robot.position.y - item.position.y)
    if item.shape == "box":
        # Distance to the nearest point on the square, including its corners.
        distance_to_surface = hypot(max(dx - item.radius, 0), max(dy - item.radius, 0))
    else:
        distance_to_surface = hypot(dx, dy) - item.radius
    gap = distance_to_surface - world.robot.radius
    logger.debug("Inspect %r: surface gap=%.4f; allowed=[%.4f, %.4f] pixels",
                 target, gap, world.standoff_gap, world.inspection_range)
    # Absorb floating-point rounding at the inclusive distance boundaries.
    tolerance = 1e-9
    if not isfinite(gap) or not world.standoff_gap - tolerance <= gap <= world.inspection_range + tolerance:
        raise ValueError(f"Cannot inspect {target!r}: surface gap {gap:.2f} pixels; "
                         f"required {world.standoff_gap:g}–{world.inspection_range:g} pixels.")
    world.inspected_objects.add(item.object_id)
    return gap


def stop(world: World) -> None:
    """Hold position; this step-based simulator has no ongoing velocity."""
    logger.debug("Stop: holding position=%s; heading=%.4f", world.robot.position, world.robot.heading)


def render(world: World, output: Path) -> None:
    """Draw the current state without changing it."""
    world.validate()
    image = Image.new("RGB", (world.width, world.height), "#f4f6f7")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)
    title_font = ImageFont.load_default(size=26)

    # Layout limitation: the title, notice board, and zone/obstacle labels below
    # use fixed pixel coordinates. If you resize the canvas or move the zone or
    # obstacle in World, update these drawing coordinates too. Longer notice text
    # may also need a larger board; text does not wrap automatically.
    # TODO: derive this layout from World settings and the corresponding shapes.
    draw.text((35, 25), "WAREHOUSE / TOP-DOWN VIEW", font=title_font, fill="#203340")
    draw.rectangle((35, 80, 1065, 270), fill="#ffffff", outline="#526574", width=2)
    for index, line in enumerate(world.notice):
        draw.text((55, 95 + index * 28), line, font=font, fill="#203340")

    draw.rectangle(world.restricted_zone.bounds(), fill="#f9dddd", outline="#b33535", width=4)
    draw.text((750, 320), "ZONE B / RESTRICTED", font=font, fill="#922c2c")

    draw.rectangle(world.obstacle.bounds(), fill="#697784", outline="#334351", width=3)
    draw.text((435, 550), "OBSTACLE", font=font, fill="white")

    for item in world.objects:
        x, y, r = item.position.x, item.position.y, item.radius
        bounds = (x - r, y - r, x + r, y + r)
        if item.shape == "circle":
            draw.ellipse(bounds, fill=item.color, outline="#203340", width=2)
        else:
            draw.rectangle(bounds, fill=item.color, outline="#203340", width=2)
        draw.text((x, y + r + 12), item.object_id, anchor="mt", font=font, fill="#203340")

    hx, hy = world.home.x, world.home.y
    draw.rectangle((hx - 35, hy - 35, hx + 35, hy + 35), outline="#28856e", width=3)
    draw.text((hx, hy + 47), "HOME / START", anchor="mt", font=font, fill="#236b59")

    robot = world.robot
    x, y, r = robot.position.x, robot.position.y, robot.radius
    draw.ellipse((x - r, y - r, x + r, y + r), fill="#203340", outline="white", width=2)
    angle = radians(robot.heading)
    tip = (x + (r + 12) * cos(angle), y + (r + 12) * sin(angle))
    draw.line((x, y, *tip), fill="#ed7d31", width=5)
    draw.ellipse((tip[0] - 4, tip[1] - 4, tip[0] + 4, tip[1] + 4), fill="#ed7d31")
    draw.text((x, y - r - 28), "ROBOT", anchor="mt", font=font, fill="#203340")

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def save_scene(world: World, output: Path) -> bool:
    """Report output failures without ending an interactive session."""
    try:
        render(world, output)
    except (OSError, ValueError) as error:
        print(f"Map not updated: {error}. The image may be stale or incomplete.")
        return False
    print(f"Scene saved to {output.resolve()}")
    return True


def print_help(world: World) -> None:
    """Show manual commands and target IDs from the current scene."""
    print("Available commands:")
    print("  /help                Show commands and available targets")
    print("  status               Show current task progress")
    print("  task <object_id>     Assign a new inspect-and-return task")
    print("  move_to <object_id>   Move beside a target (example: move_to blue_package)")
    print("  inspect <object_id>   Record an observation of a nearby object")
    print("  return_home          Move back to HOME / START")
    print("  stop                 Hold the current position")
    print("  quit                 End the session")
    print("Available targets: " + ", ".join(item.object_id for item in world.objects))
    print(f"Inspection surface gap: {world.standoff_gap:g}–{world.inspection_range:g} pixels.")
    print("For calculation details, launch with: python3 simulator.py --interactive --debug")
    print("Restricted-zone protection and task authorization are ON. Assign a task before move_to or inspect.")
    print("Obstacle collisions are not enforced yet.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", action="store_true", help="Accept manual robot commands")
    parser.add_argument("--debug", action="store_true", help="Print each movement calculation and application check")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING,
                        format="[DEBUG] %(message)s")
    # Enable our trace only; keep dependency diagnostics out of the walkthrough.
    logging.getLogger("PIL").setLevel(logging.WARNING)
    world = World()
    destination = Path("outputs/scene.png")
    saved = save_scene(world, destination)
    if not args.interactive:
        if not saved:
            raise SystemExit(1)
        return

    print_help(world)
    while True:
        try:
            parts = input("> ").split()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break
        if parts == ["quit"]:
            break
        if parts == ["/help"]:
            print_help(world)
            continue
        if parts == ["status"]:
            print(f"Task: {world.task_status}")
            continue
        try:
            if len(parts) == 2 and parts[0] == "task":
                assign_task(world, parts[1])
                print(f"Task: {world.task_status}")
                continue
            elif len(parts) == 2 and parts[0] == "move_to":
                move_to(world, parts[1])
            elif len(parts) == 2 and parts[0] == "inspect":
                gap = inspect(world, parts[1])
                print(f"Inspected {parts[1]} (surface gap: {gap:.2f} pixels).")
            elif parts == ["return_home"]:
                return_home(world)
            elif parts == ["stop"]:
                stop(world)
            else:
                print("Unknown command or incorrect arguments. Type /help for commands and targets.")
                continue
        except ValueError as error:
            print(f"Command rejected: {error}")
            continue
        p = world.robot.position
        # Movement is already applied even if writing the image fails.
        print(f"Robot at ({p.x:.1f}, {p.y:.1f}), heading {world.robot.heading:.1f} degrees.")
        print(f"Task: {world.task_status}")
        save_scene(world, destination)


if __name__ == "__main__":
    main()
