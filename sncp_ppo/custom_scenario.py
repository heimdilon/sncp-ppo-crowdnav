"""Custom scenario loading and application for manual policy tests."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from crowd_sim.crowd_env import CrowdSimEnv


DEFAULT_HUMAN_GOAL_DISTANCE = 8.0
VALID_MOTION_MODELS = {"sfm", "linear"}


@dataclass(frozen=True)
class RobotSpec:
    x: float
    y: float
    gx: float
    gy: float
    theta: float


@dataclass(frozen=True)
class HumanSpec:
    identifier: str
    x: float
    y: float
    gx: float
    gy: float
    theta: float
    speed: float


@dataclass(frozen=True)
class CustomScenario:
    version: int
    name: str
    time_step: float
    max_time: float
    human_motion_model: str
    human_dodge_robot: bool
    robot: RobotSpec
    humans: tuple[HumanSpec, ...]


def load_custom_scenario(path_or_data: str | Path | Mapping[str, Any]) -> CustomScenario:
    if isinstance(path_or_data, Mapping):
        data = dict(path_or_data)
    else:
        path = Path(path_or_data)
        data = json.loads(path.read_text(encoding="utf-8"))

    motion_model = str(data.get("human_motion_model", "sfm")).lower()
    if motion_model not in VALID_MOTION_MODELS:
        raise ValueError("human_motion_model must be 'sfm' or 'linear'")

    robot = _parse_robot(data.get("robot"))
    humans_data = data.get("humans", [])
    if not isinstance(humans_data, list):
        raise ValueError("humans must be a list")
    humans = tuple(_parse_human(item, index) for index, item in enumerate(humans_data))

    return CustomScenario(
        version=int(data.get("version", 1)),
        name=str(data.get("name", "custom scenario")),
        time_step=float(data.get("time_step", 0.25)),
        max_time=float(data.get("max_time", 50.0)),
        human_motion_model=motion_model,
        human_dodge_robot=bool(data.get("human_dodge_robot", False)),
        robot=robot,
        humans=humans,
    )


def create_custom_env(scenario: CustomScenario, seed: int | None = None) -> CrowdSimEnv:
    env = CrowdSimEnv(
        num_humans=len(scenario.humans),
        time_step=scenario.time_step,
        max_time=scenario.max_time,
        scenario="circle",
        human_dodge_robot=scenario.human_dodge_robot,
        randomize_layout=False,
        human_motion_model=scenario.human_motion_model,
    )
    apply_custom_scenario(env, scenario, seed=seed)
    return env


def apply_custom_scenario(
    env: CrowdSimEnv,
    scenario: CustomScenario,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    if env.num_humans != len(scenario.humans):
        raise ValueError("env.num_humans must match the custom scenario human count")

    env.reset(seed=seed)
    env.current_time = 0.0
    env.time_step = scenario.time_step
    env.max_time = scenario.max_time
    env.human_dodge_robot = scenario.human_dodge_robot
    env.human_motion_model = scenario.human_motion_model

    robot = scenario.robot
    env.robot_px = robot.x
    env.robot_py = robot.y
    env.robot_gx = robot.gx
    env.robot_gy = robot.gy
    env.robot_theta = robot.theta
    env.robot_vx = 0.0
    env.robot_vy = 0.0
    env._last_w = 0.0

    env.humans_px = np.array([human.x for human in scenario.humans], dtype=float)
    env.humans_py = np.array([human.y for human in scenario.humans], dtype=float)
    env.humans_gx = np.array([human.gx for human in scenario.humans], dtype=float)
    env.humans_gy = np.array([human.gy for human in scenario.humans], dtype=float)
    env.humans_theta = np.array([human.theta for human in scenario.humans], dtype=float)
    env.humans_vpref = np.array([human.speed for human in scenario.humans], dtype=float)
    env.humans_vx = np.array(
        [human.speed * math.cos(human.theta) for human in scenario.humans],
        dtype=float,
    )
    env.humans_vy = np.array(
        [human.speed * math.sin(human.theta) for human in scenario.humans],
        dtype=float,
    )
    return env._get_obs()


def _parse_robot(data: Any) -> RobotSpec:
    if not isinstance(data, Mapping):
        raise ValueError("robot must be an object")
    position = _parse_point(data.get("position"), "robot.position")
    goal = _parse_point(data.get("goal"), "robot.goal")
    theta = _parse_angle(data, default=math.atan2(goal[1] - position[1], goal[0] - position[0]))
    return RobotSpec(
        x=position[0],
        y=position[1],
        gx=goal[0],
        gy=goal[1],
        theta=theta,
    )


def _parse_human(data: Any, index: int) -> HumanSpec:
    if not isinstance(data, Mapping):
        raise ValueError(f"humans[{index}] must be an object")
    position = _parse_point(data.get("position"), f"humans[{index}].position")
    theta = _parse_angle(data, default=0.0)
    speed = float(data.get("speed", 0.0))
    if speed < 0.0:
        raise ValueError(f"humans[{index}].speed must be non-negative")

    if "goal" in data:
        goal = _parse_point(data["goal"], f"humans[{index}].goal")
    else:
        goal = (
            _clean_float(position[0] + DEFAULT_HUMAN_GOAL_DISTANCE * math.cos(theta)),
            _clean_float(position[1] + DEFAULT_HUMAN_GOAL_DISTANCE * math.sin(theta)),
        )

    return HumanSpec(
        identifier=str(data.get("id", f"h{index + 1}")),
        x=position[0],
        y=position[1],
        gx=goal[0],
        gy=goal[1],
        theta=theta,
        speed=speed,
    )


def _parse_point(data: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(data, Mapping) or "x" not in data or "y" not in data:
        raise ValueError(f"{field_name} must contain x and y")
    return (_clean_float(float(data["x"])), _clean_float(float(data["y"])))


def _parse_angle(data: Mapping[str, Any], default: float) -> float:
    if "theta" in data:
        return _normalize_angle(float(data["theta"]))
    if "theta_deg" in data:
        return _normalize_angle(math.radians(float(data["theta_deg"])))
    return _normalize_angle(default)


def _normalize_angle(angle: float) -> float:
    if -math.pi <= angle <= math.pi:
        return _clean_float(angle)
    normalized = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return _clean_float(normalized)


def _clean_float(value: float) -> float:
    return 0.0 if abs(value) < 1e-12 else float(value)
