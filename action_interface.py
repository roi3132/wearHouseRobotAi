"""Strict JSON proposal validation and bounded execution through trusted actions.

Input is the final model answer, not hidden reasoning. No code is evaluated.
All actions keep the assignment's action/target/rationale shape. For return_home
and stop, target must still be a known ID but has no execution meaning.
"""

import argparse
import json
from math import isfinite
from pathlib import Path

from simulator import World, assign_task, find_object, inspect, move_to, render, return_home, stop

MAX_RESPONSE_BYTES = 16_384
MAX_PLAN_STEPS = 12
ACTIONS = frozenset({"move_to", "inspect", "return_home", "stop"})


def unique_object(pairs: list[tuple]) -> dict:
    """Reject duplicate JSON keys instead of silently keeping the last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key!r}.")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON numeric constant: {value}.")


def require_fields(value: object, fields: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} must contain exactly: {', '.join(sorted(fields))}.")


def require_text(value: object, limit: int, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{label} must be nonempty text of at most {limit} characters.")


def parse_plan(raw: str, world: World) -> dict:
    """Validate the complete proposal without executing or changing state."""
    if not isinstance(raw, str) or len(raw.encode('utf-8')) > MAX_RESPONSE_BYTES:
        raise ValueError(f"Response must be JSON text of at most {MAX_RESPONSE_BYTES} bytes.")
    try:
        proposal = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Response must be one valid JSON object, without prose or fences.") from error
    require_fields(proposal, {"scene_summary", "confidence", "plan"}, "Proposal")
    require_text(proposal['scene_summary'], 1000, "scene_summary")
    confidence = proposal['confidence']
    if type(confidence) not in (int, float) or not 0 <= confidence <= 1 or not isfinite(confidence):
        raise ValueError("confidence must be a finite number between 0 and 1.")
    plan = proposal['plan']
    if not isinstance(plan, list) or not 1 <= len(plan) <= MAX_PLAN_STEPS:
        raise ValueError(f"plan must contain 1–{MAX_PLAN_STEPS} actions.")
    for step in plan:
        require_fields(step, {"action", "target", "rationale"}, "Action")
        require_text(step['action'], 32, "action")
        if step['action'] not in ACTIONS:
            raise ValueError(f"Unsupported action: {step['action']!r}.")
        require_text(step['target'], 100, "target")
        find_object(world, step['target'])
        require_text(step['rationale'], 500, "rationale")
    return proposal


def execute_plan(world: World, raw: str) -> dict:
    """Execute a fully validated proposal; a denial or stop ends the plan.

    Earlier successful actions are not rolled back. Summary, confidence and
    rationale are untrusted metadata and never grant permission.
    """
    proposal = parse_plan(raw, world)
    events = []
    outcome = "executed"
    for index, step in enumerate(proposal['plan']):
        action, target = step['action'], step['target']
        try:
            # Explicit dispatch prevents JSON from naming arbitrary functions.
            if action == 'move_to':
                move_to(world, target)
            elif action == 'inspect':
                inspect(world, target)
            elif action == 'return_home':
                return_home(world)
            else:  # parse_plan has already restricted this to stop.
                stop(world)
        except ValueError as error:
            stop(world)
            events.append({"step": index + 1, "action": action, "target": target,
                           "decision": "denied", "reason": str(error)})
            outcome = "denied"
            break
        events.append({"step": index + 1, "action": action, "target": target,
                       "decision": "executed", "position": {
                           "x": world.robot.position.x, "y": world.robot.position.y}})
        if action == 'stop':
            outcome = "stopped"
            break
    return {"proposal": proposal, "outcome": outcome, "events": events,
            "skipped_steps": len(proposal['plan']) - len(events),
            "defences_enabled": world.defences_enabled, "zone_contact": world.zone_contact,
            "task_complete": world.task_complete, "task_status": world.task_status}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True, help='Local JSON proposal to execute')
    parser.add_argument('--task', required=True, help='Trusted operator-assigned object ID')
    args = parser.parse_args()
    world = World()
    try:
        assign_task(world, args.task)
        # Bound the read as well as the parser's accepted input size.
        with args.plan.open('rb') as source:
            raw = source.read(MAX_RESPONSE_BYTES + 1).decode('utf-8')
        result = execute_plan(world, raw)
        render(world, Path('outputs/scene.png'))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Plan rejected or output failed: {error}\n")
    print(json.dumps(result, indent=2))
    if result['outcome'] == 'denied':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
