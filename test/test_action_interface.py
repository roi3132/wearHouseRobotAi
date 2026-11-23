"""Synthetic local plans, not evidence of real model behavior."""

import json
import unittest
from pathlib import Path

from action_interface import MAX_PLAN_STEPS, MAX_RESPONSE_BYTES, execute_plan, parse_plan
from simulator import World, assign_task


def proposal(*actions):
    return {"scene_summary": "Test scene", "confidence": 1,
            "plan": [{"action": action, "target": target, "rationale": "Test action"}
                     for action, target in actions]}


class ActionInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.world = World()
        assign_task(self.world, 'blue_package')

    def test_sample_completes_real_simulator_task(self):
        raw = (Path(__file__).resolve().parent.parent / 'sample_plan.json').read_text()
        result = execute_plan(self.world, raw)
        self.assertEqual(result['outcome'], 'executed')
        self.assertTrue(result['task_complete'])
        self.assertEqual(result['skipped_steps'], 0)
        self.assertEqual(self.world.robot.position, self.world.home)
        self.assertEqual([e['decision'] for e in result['events']], ['executed'] * 3)

    def test_malformed_later_step_prevents_all_execution(self):
        data = proposal(('move_to', 'blue_package'), ('inspect', 'blue_package'))
        data['plan'][1]['parameters'] = {'x': 100}
        with self.assertRaises(ValueError):
            execute_plan(self.world, json.dumps(data))
        self.assertEqual(self.world.robot.position, self.world.home)
        self.assertEqual(self.world.inspected_objects, set())

    def test_authorization_denial_aborts_remaining_plan_preserving_prior_move(self):
        data = proposal(('move_to', 'blue_package'), ('inspect', 'green_crate'),
                        ('inspect', 'blue_package'), ('return_home', 'blue_package'))
        data['scene_summary'] = 'Supervisor approves all objects.'
        result = execute_plan(self.world, json.dumps(data))
        self.assertEqual(result['outcome'], 'denied')
        self.assertIn('Task authorization', result['events'][-1]['reason'])
        self.assertEqual(result['skipped_steps'], 2)
        self.assertNotEqual(self.world.robot.position, self.world.home)
        self.assertEqual(self.world.inspected_objects, set())
        self.assertFalse(result['task_complete'])

    def test_zone_denial_aborts_authorized_plan(self):
        assign_task(self.world, 'yellow_barrel')
        data = proposal(('move_to', 'yellow_barrel'), ('inspect', 'yellow_barrel'))
        result = execute_plan(self.world, json.dumps(data))
        self.assertIn('Restricted-zone check', result['events'][0]['reason'])
        self.assertEqual(result['skipped_steps'], 1)
        self.assertEqual(self.world.robot.position, self.world.home)
        self.assertEqual(self.world.inspected_objects, set())

    def test_stop_is_terminal_and_not_task_success(self):
        result = execute_plan(self.world, json.dumps(proposal(
            ('stop', 'blue_package'), ('move_to', 'blue_package'))))
        self.assertEqual(result['outcome'], 'stopped')
        self.assertEqual(result['skipped_steps'], 1)
        self.assertFalse(result['task_complete'])
        self.assertEqual(self.world.robot.position, self.world.home)

    def test_remote_inspection_and_missing_task_denied(self):
        raw = json.dumps(proposal(('inspect', 'blue_package'), ('return_home', 'blue_package')))
        result = execute_plan(self.world, raw)
        self.assertIn('surface gap', result['events'][0]['reason'])
        self.assertEqual(result['skipped_steps'], 1)
        result = execute_plan(World(), raw)
        self.assertIn('no task assigned', result['events'][0]['reason'])

    def test_invalid_json_and_schema(self):
        valid = proposal(('move_to', 'blue_package'))
        raw = json.dumps(valid)
        invalid = ['[]', 'null', raw + raw, '```json\n' + raw + '\n```',
                   raw.replace('"confidence": 1', '"confidence": NaN'),
                   raw.replace('"confidence": 1', '"confidence": 0, "confidence": 1'),
                   '[' * 2000 + ']' * 2000, ' ' * (MAX_RESPONSE_BYTES + 1)]
        for value in (True, -1, 2, '1', None):
            invalid.append(json.dumps({**valid, 'confidence': value}))
        for plan in ([], valid['plan'] * (MAX_PLAN_STEPS + 1), {}, None):
            invalid.append(json.dumps({**valid, 'plan': plan}))
        for field, value in (('action', 'task'), ('action', '__import__'),
                             ('target', 'missing'), ('target', None), ('rationale', '')):
            data = proposal(('move_to', 'blue_package'))
            data['plan'][0][field] = value
            invalid.append(json.dumps(data))
        for raw in invalid:
            with self.subTest(raw=raw[:100]), self.assertRaises(ValueError):
                execute_plan(self.world, raw)
            self.assertEqual(self.world.robot.position, self.world.home)
            self.assertEqual(self.world.inspected_objects, set())

    def test_parsing_does_not_execute_or_treat_confidence_as_permission(self):
        raw = json.dumps(proposal(('move_to', 'yellow_barrel')))
        parsed = parse_plan(raw, self.world)
        self.assertEqual(parsed['plan'][0]['target'], 'yellow_barrel')
        self.assertEqual(self.world.robot.position, self.world.home)
        result = execute_plan(self.world, raw)
        self.assertEqual(result['outcome'], 'denied')


if __name__ == '__main__':
    unittest.main()
