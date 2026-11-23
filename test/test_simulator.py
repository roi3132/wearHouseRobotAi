"""Local movement, inspection, task progress and restricted-zone checks."""

import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from math import hypot, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from simulator import (
    Motion, Point, Rectangle, SceneObject, World, apply_motion, main,
    assign_task, inspect, move_to, path_touches_zone, plan_move_to, render, return_home, stop,
)


class PathZoneTests(unittest.TestCase):
    def test_crossing_entry_and_clear_paths_in_both_directions(self):
        zone = Rectangle(100, 100, 200, 200)
        cases = (
            (Point(50, 150), Point(250, 150), True),
            (Point(150, 50), Point(150, 250), True),
            (Point(50, 50), Point(250, 250), True),
            (Point(50, 150), Point(150, 150), True),
            (Point(50, 50), Point(250, 50), False),
            (Point(50, 50), Point(50, 250), False),
            (Point(50, 120), Point(120, 50), False),
        )
        for start, end, expected in cases:
            for a, b in ((start, end), (end, start)):
                with self.subTest(start=a, end=b):
                    self.assertEqual(path_touches_zone(Motion(a, b, 0), zone, 10), expected)

    def test_body_contact_corner_and_just_clear(self):
        zone = Rectangle(100, 100, 200, 200)
        for x, expected in ((90, True), (95, True), (89.99, False)):
            with self.subTest(x=x):
                self.assertEqual(path_touches_zone(
                    Motion(Point(x, 50), Point(x, 250), 0), zone, 10), expected)
        # Touches the square-expanded corner (90, 90): conservatively blocked.
        self.assertTrue(path_touches_zone(Motion(Point(80, 100), Point(100, 80), 0), zone, 10))

    def test_stationary_and_zero_radius(self):
        zone = Rectangle(100, 100, 200, 200)
        for point, expected in ((Point(150, 150), True), (Point(100, 150), True),
                                (Point(50, 50), False)):
            with self.subTest(point=point):
                self.assertEqual(path_touches_zone(Motion(point, point, 0), zone, 0), expected)

    def test_invalid_radius_and_overflow_rejected(self):
        motion = Motion(Point(0, 0), Point(10, 10), 0)
        zone = Rectangle(100, 100, 200, 200)
        for radius in (-1, float('nan'), float('inf')):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                path_touches_zone(motion, zone, radius)
        with self.assertRaises(ValueError):
            path_touches_zone(Motion(Point(-1e308, 0), Point(1e308, 0), 0), zone, 0)

    def test_check_does_not_change_world_or_execute_motion(self):
        world = World()
        motion = plan_move_to(world, "yellow_barrel")
        before = (world.robot.position, world.robot.heading, world.task_complete,
                  set(world.inspected_objects))
        self.assertTrue(path_touches_zone(motion, world.restricted_zone, world.robot.radius))
        self.assertEqual(before, (world.robot.position, world.robot.heading, world.task_complete,
                                  world.inspected_objects))


class MovementTests(unittest.TestCase):
    def test_planning_does_not_move_robot(self):
        world = World(task_target="blue_package")
        before = (world.robot.position, world.robot.heading)
        plan_move_to(world, "blue_package")
        self.assertEqual(before, (world.robot.position, world.robot.heading))

    def test_box_clearance_and_repeated_move(self):
        world = World(task_target="blue_package")
        move_to(world, "blue_package")
        p = world.robot.position
        self.assertAlmostEqual(hypot(p.x - 340, p.y - 390), 25 * sqrt(2) + 18 + 10)
        self.assertLess(p.x, 340)
        self.assertGreater(p.y, 390)
        self.assertTrue(270 < world.robot.heading < 360)
        move_to(world, "blue_package")
        self.assertAlmostEqual(p.x, world.robot.position.x)
        self.assertAlmostEqual(p.y, world.robot.position.y)

    def test_unknown_target_leaves_pose_unchanged(self):
        world = World(task_target="blue_package")
        before = (world.robot.position, world.robot.heading)
        with self.assertRaises(ValueError):
            move_to(world, "missing_package")
        self.assertEqual(before, (world.robot.position, world.robot.heading))

    def test_stop_and_return_home(self):
        world = World(task_target="blue_package")
        move_to(world, "blue_package")
        before = (world.robot.position, world.robot.heading)
        stop(world)
        self.assertEqual(before, (world.robot.position, world.robot.heading))
        return_home(world)
        self.assertEqual(world.robot.position, world.home)

    def test_target_at_robot_center_does_not_divide_by_zero(self):
        world = World(task_target="blue_package")
        world.robot.position = Point(340, 390)
        move_to(world, "blue_package")
        self.assertEqual(world.robot.position, Point(340, 390))

    def test_invalid_motion_heading_leaves_pose_unchanged(self):
        for heading in (float('inf'), float('-inf'), float('nan')):
            with self.subTest(heading=heading):
                world = World(task_target="blue_package")
                before = (world.robot.position, world.robot.heading)
                with self.assertRaises(ValueError):
                    apply_motion(world, Motion(world.robot.position, Point(200, 200), heading))
                self.assertEqual(before, (world.robot.position, world.robot.heading))

    def test_invalid_mutable_settings_rejected_before_movement(self):
        cases = (("standoff_gap", -100), ("standoff_gap", float('inf')),
                 ("width", 0), ("height", 800.5), ("radius", -1),
                 ("radius", float('nan')), ("heading", float('inf')))
        for name, value in cases:
            with self.subTest(name=name, value=value):
                world = World(task_target="blue_package")
                owner = world.robot if name in ("radius", "heading") else world
                setattr(owner, name, value)
                before = (world.robot.position, world.robot.heading)
                with self.assertRaises(ValueError):
                    move_to(world, "blue_package")
                self.assertEqual(before, (world.robot.position, world.robot.heading))

    def test_invalid_scene_geometry_rejected(self):
        for radius, shape in ((25, "square"), (-1, "box"), (float('inf'), "circle")):
            with self.subTest(radius=radius, shape=shape), self.assertRaises(ValueError):
                SceneObject("target", Point(300, 300), radius, "blue", shape)
        with self.assertRaises(ValueError):
            Point(float('nan'), 0)
        with self.assertRaises(ValueError):
            Rectangle(20, 0, 10, 30)
        with self.assertRaises(ValueError):
            Rectangle(0, 0, float('inf'), 30)

    def test_stale_motion_leaves_pose_unchanged(self):
        world = World(task_target="blue_package")
        motion = plan_move_to(world, "blue_package")
        assign_task(world, "green_crate")
        move_to(world, "green_crate")
        before = (world.robot.position, world.robot.heading)
        with self.assertRaises(ValueError):
            apply_motion(world, motion)
        self.assertEqual(before, (world.robot.position, world.robot.heading))

    def test_map_boundary_accounts_for_robot_radius(self):
        world = World(task_target="blue_package")
        apply_motion(world, Motion(world.robot.position, Point(18, 18), 0))
        before = (world.robot.position, world.robot.heading)
        with self.assertRaises(ValueError):
            apply_motion(world, Motion(world.robot.position, Point(17, 18), 90))
        self.assertEqual(before, (world.robot.position, world.robot.heading))

    def test_duplicate_target_rejected(self):
        world = World(task_target="blue_package")
        world.objects += (world.objects[0],)
        before = (world.robot.position, world.robot.heading)
        with self.assertRaises(ValueError):
            move_to(world, "blue_package")
        self.assertEqual(before, (world.robot.position, world.robot.heading))

    def test_circle_clearance(self):
        world = World(task_target="blue_package")
        # Check planning geometry without executing entry into the zone.
        p = plan_move_to(world, "yellow_barrel").end
        self.assertAlmostEqual(hypot(p.x - 880, p.y - 490), 30 + 18 + 10)


class InspectionTests(unittest.TestCase):
    def test_move_then_inspect_each_shape_without_moving(self):
        for target in ("blue_package", "yellow_barrel", "green_crate"):
            with self.subTest(target=target):
                world = World(task_target=target)
                # Put the circle in safe space to isolate inspection mechanics.
                world.objects = tuple(replace(item, position=Point(500, 350))
                                      if item.object_id == "yellow_barrel" else item
                                      for item in world.objects)
                move_to(world, target)
                before = (world.robot.position, world.robot.heading)
                gap = inspect(world, target)
                self.assertAlmostEqual(inspect(world, target), gap)
                self.assertEqual(world.inspected_objects, {target})
                self.assertEqual(before, (world.robot.position, world.robot.heading))

    def test_surface_gap_boundaries_for_box_and_circle(self):
        for target in ("blue_package", "yellow_barrel"):
            for gap in (10, 25, 9.99, 25.01, -1):
                with self.subTest(target=target, gap=gap):
                    world = World(task_target=target)
                    item = next(item for item in world.objects if item.object_id == target)
                    world.robot.position = Point(item.position.x - item.radius - world.robot.radius - gap,
                                                 item.position.y)
                    if 10 <= gap <= 25:
                        self.assertAlmostEqual(inspect(world, target), gap)
                    else:
                        with self.assertRaises(ValueError):
                            inspect(world, target)
                        self.assertEqual(world.inspected_objects, set())

    def test_rejected_inspection_preserves_state(self):
        for case in ("far", "unknown", "duplicate", "overlap", "invalid_range"):
            with self.subTest(case=case):
                world = World(task_target="blue_package")
                target = "blue_package"
                if case == "unknown":
                    target = "missing"
                elif case == "duplicate":
                    world.objects += (world.objects[0],)
                elif case == "overlap":
                    world.robot.position = world.objects[0].position
                elif case == "invalid_range":
                    world.inspection_range = float('nan')
                before = (world.robot.position, world.robot.heading)
                with self.assertRaises(ValueError):
                    inspect(world, target)
                self.assertEqual(world.inspected_objects, set())
                self.assertEqual(before, (world.robot.position, world.robot.heading))

    def test_manual_inspection_command(self):
        world = World(task_target="blue_package")
        output = StringIO()
        with patch('sys.argv', ['simulator.py', '--interactive']), \
                patch('simulator.World', return_value=world), \
                patch('builtins.input', side_effect=['inspect blue_package', 'move_to blue_package',
                                                   'inspect blue_package', 'quit']), \
                patch('simulator.render'), redirect_stdout(output):
            main()
        self.assertIn('Command rejected:', output.getvalue())
        self.assertIn('Inspected blue_package', output.getvalue())
        self.assertEqual(world.inspected_objects, {'blue_package'})


class ZoneEnforcementTests(unittest.TestCase):
    def assert_rejected_unchanged(self, world, action):
        before = (world.robot.position, world.robot.heading, world.task_complete,
                  set(world.inspected_objects))
        with self.assertRaisesRegex(ValueError, "Restricted-zone check"):
            action()
        self.assertEqual(before, (world.robot.position, world.robot.heading, world.task_complete,
                                  world.inspected_objects))

    def test_barrel_move_rejected_without_changing_completed_task(self):
        world = World()
        assign_task(world, "blue_package")
        move_to(world, "blue_package")
        inspect(world, "blue_package")
        return_home(world)
        self.assertTrue(world.task_complete)
        self.assert_rejected_unchanged(world, lambda: apply_motion(
            world, plan_move_to(world, "yellow_barrel")))

    def test_crossing_with_destination_outside_is_rejected(self):
        world = World()
        world.robot.position = Point(650, 400)
        self.assert_rejected_unchanged(world, lambda: apply_motion(
            world, Motion(world.robot.position, Point(1080, 400), 0)))

    def test_return_home_denial_does_not_complete_task(self):
        world = World()
        assign_task(world, "blue_package")
        move_to(world, "blue_package")
        inspect(world, "blue_package")
        world.restricted_zone = Rectangle(200, 480, 240, 580)
        self.assert_rejected_unchanged(world, lambda: return_home(world))
        self.assertFalse(world.task_complete)

    def test_stationary_inside_zone_rejected_but_stop_available(self):
        world = World()
        world.robot.position = Point(800, 400)
        self.assert_rejected_unchanged(world, lambda: apply_motion(
            world, Motion(world.robot.position, world.robot.position, 0)))
        stop(world)
        self.assertEqual(world.robot.position, Point(800, 400))

    def test_manual_session_can_continue_after_denial(self):
        world = World()
        output = StringIO()
        with patch('sys.argv', ['simulator.py', '--interactive']), \
                patch('simulator.World', return_value=world), \
                patch('builtins.input', side_effect=['task yellow_barrel', 'move_to yellow_barrel',
                                                   'task blue_package', 'move_to blue_package', 'quit']), \
                patch('simulator.render') as draw, redirect_stdout(output):
            main()
        self.assertIn('Command rejected: Restricted-zone check', output.getvalue())
        self.assertEqual(draw.call_count, 2)
        inspect(world, "blue_package")


class AuthorizationTests(unittest.TestCase):
    def test_safe_path_to_wrong_target_is_denied_without_state_changes(self):
        world = World(task_target="blue_package")
        motion = plan_move_to(world, "green_crate")
        self.assertFalse(path_touches_zone(motion, world.restricted_zone, world.robot.radius))
        before = (world.robot.position, world.robot.heading, world.task_status)
        with self.assertRaisesRegex(ValueError, "Task authorization"):
            move_to(world, "green_crate")
        self.assertEqual(before, (world.robot.position, world.robot.heading, world.task_status))

    def test_assigned_target_still_cannot_enter_zone(self):
        world = World()
        assign_task(world, "yellow_barrel")
        before = (world.robot.position, world.robot.heading)
        with self.assertRaisesRegex(ValueError, "Restricted-zone check"):
            move_to(world, "yellow_barrel")
        self.assertEqual(before, (world.robot.position, world.robot.heading))

    def test_nearby_wrong_target_cannot_be_inspected(self):
        world = World(task_target="green_crate")
        move_to(world, "green_crate")
        inspect(world, "green_crate")  # Prove proximity is sufficient.
        assign_task(world, "blue_package")
        before = (world.robot.position, world.robot.heading, world.task_status)
        with self.assertRaisesRegex(ValueError, "Task authorization"):
            inspect(world, "green_crate")
        self.assertEqual(world.inspected_objects, set())
        self.assertEqual(before, (world.robot.position, world.robot.heading, world.task_status))

    def test_no_task_blocks_target_actions_but_allows_home_and_stop(self):
        world = World()
        # Use a valid nearby pose so inspection fails for authorization alone.
        world.robot.position = plan_move_to(world, "blue_package").end
        before = (world.robot.position, world.robot.heading)
        for action in (move_to, inspect):
            with self.subTest(action=action.__name__), self.assertRaisesRegex(
                    ValueError, "Task authorization: no task assigned"):
                action(world, "blue_package")
            self.assertEqual(before, (world.robot.position, world.robot.heading))
        stop(world)
        return_home(world)
        self.assertEqual(world.robot.position, world.home)
        self.assertFalse(world.task_complete)


class TaskCompletionTests(unittest.TestCase):
    def test_no_task_until_assigned_and_old_inspections_do_not_count(self):
        world = World()
        self.assertEqual(world.task_status, "No task assigned.")
        with self.assertRaisesRegex(ValueError, "Task authorization"):
            move_to(world, "blue_package")
        self.assertFalse(world.task_complete)
        assign_task(world, "blue_package")
        return_home(world)
        self.assertFalse(world.task_complete)
        self.assertEqual(world.inspected_objects, set())
        move_to(world, "blue_package")
        inspect(world, "blue_package")
        return_home(world)
        self.assertTrue(world.task_complete)
        with self.assertRaises(ValueError):
            assign_task(world, "missing")
        self.assertTrue(world.task_complete)
        self.assertEqual(world.task_target, "blue_package")
        assign_task(world, "blue_package")
        self.assertFalse(world.task_complete)
        self.assertEqual(world.inspected_objects, set())

    def test_inspect_then_return_completes_task(self):
        world = World()
        assign_task(world, "blue_package")
        self.assertFalse(world.task_complete)
        return_home(world)
        self.assertFalse(world.task_complete)
        move_to(world, "blue_package")
        inspect(world, "blue_package")
        self.assertFalse(world.task_complete)
        self.assertIn("return home next", world.task_status)
        stop(world)
        self.assertFalse(world.task_complete)
        return_home(world)
        self.assertTrue(world.task_complete)
        stop(world)
        return_home(world)
        self.assertTrue(world.task_complete)
        move_to(world, "blue_package")
        self.assertFalse(world.task_complete)
        return_home(world)
        self.assertTrue(world.task_complete)

    def test_wrong_target_or_failed_inspection_cannot_complete_task(self):
        world = World()
        assign_task(world, "blue_package")
        with self.assertRaises(ValueError):
            inspect(world, "blue_package")
        return_home(world)
        self.assertFalse(world.task_complete)
        with self.assertRaisesRegex(ValueError, "Task authorization"):
            move_to(world, "green_crate")
        with self.assertRaisesRegex(ValueError, "Task authorization"):
            inspect(world, "green_crate")
        return_home(world)
        self.assertFalse(world.task_complete)

    def test_failed_return_does_not_complete_task(self):
        world = World()
        assign_task(world, "blue_package")
        move_to(world, "blue_package")
        inspect(world, "blue_package")
        world.home = Point(0, 0)
        with self.assertRaises(ValueError):
            return_home(world)
        self.assertFalse(world.task_complete)

    def test_configured_target_and_separate_worlds(self):
        world = World(task_target="green_crate")
        move_to(world, "green_crate")
        inspect(world, "green_crate")
        return_home(world)
        self.assertTrue(world.task_complete)
        other = World()
        self.assertFalse(other.task_complete)
        self.assertEqual(other.inspected_objects, set())

    def test_manual_session_reports_completion(self):
        output = StringIO()
        with patch('sys.argv', ['simulator.py', '--interactive']), \
                patch('builtins.input', side_effect=['status', 'task blue_package',
                                                   'move_to blue_package', 'inspect blue_package',
                                                   'return_home', 'quit']), \
                patch('simulator.render'), redirect_stdout(output):
            main()
        self.assertIn('target inspected; return home next', output.getvalue())
        self.assertIn('Task: No task assigned.', output.getvalue())
        self.assertNotIn('Objective:', output.getvalue())
        self.assertIn('Complete: target inspected and robot returned home.', output.getvalue())


class RenderingTests(unittest.TestCase):
    def test_render_writes_readable_image_without_changing_pose(self):
        world = World()
        before = (world.robot.position, world.robot.heading)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "scene.png"
            render(world, output)
            with Image.open(output) as image:
                self.assertEqual(image.size, (world.width, world.height))
                image.load()
        self.assertEqual(before, (world.robot.position, world.robot.heading))

    def test_interactive_session_recovers_from_save_failure(self):
        world = World()
        output = StringIO()
        with patch('sys.argv', ['simulator.py', '--interactive']), \
                patch('simulator.World', return_value=world), \
                patch('builtins.input', side_effect=['task blue_package', 'move_to blue_package', 'return_home', 'quit']), \
                patch('simulator.render', side_effect=[None, OSError('disk full'), None]) as draw, \
                redirect_stdout(output):
            main()
        self.assertEqual(draw.call_count, 3)
        self.assertEqual(world.robot.position, world.home)
        self.assertIn('Map not updated: disk full', output.getvalue())
        self.assertIn('Robot at (300.2, 439.3)', output.getvalue())

    def test_initial_save_failure_has_nonzero_exit_status(self):
        with patch('sys.argv', ['simulator.py']), \
                patch('simulator.render', side_effect=PermissionError('read only')), \
                redirect_stdout(StringIO()), self.assertRaises(SystemExit) as error:
            main()
        self.assertEqual(error.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
