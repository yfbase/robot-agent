"""Offline arm-tool tests: the STS3215 serial bus is faked, no hardware needed."""

import pytest
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel

import simple_arm
from robot_agent import RobotAgent
from robot_agent.tools import make_arm_tools
from simple_arm import Joint, Reg, SimpleArm

# Fake EEPROM travel: 888..3208 ticks, midpoint 2048 (= 0 deg), span ±102 deg
# (wide enough for every angle in SimpleArm.HIGH_POSE, like the real wrist).
MIN_RAW, MAX_RAW = 888, 3208
MID_RAW = (MIN_RAW + MAX_RAW) // 2


def deg_to_raw(deg):
    return round(deg * 4095 / 360 + MID_RAW)


class FakeBus:
    """Six fake motors: feedback reports the current goal (instant settling).

    Set ``fail_reads`` to simulate a disconnected/busy bus.
    """

    def __init__(self, port=None):
        self.goal = {mid: MID_RAW for mid in range(1, 7)}
        self.torque = {mid: 0 for mid in range(1, 7)}
        self.writes = []  # (mid, addr, data) in order
        self.fail_reads = False

    def read_reg(self, mid, addr, count):
        if self.fail_reads:
            raise TimeoutError(f"ID{mid} no response")
        if addr == Reg.CONFIG:
            buf = bytearray(34)
            buf[9:11] = MIN_RAW.to_bytes(2, "little")
            buf[11:13] = MAX_RAW.to_bytes(2, "little")
            return bytes(buf)
        if addr == Reg.PRESENT_FEEDBACK:
            return (
                self.goal[mid].to_bytes(2, "little")
                + b"\x00\x00\x00\x00"  # speed, load
                + bytes([120, 30, 0, 0, 0])  # voltage, temperature, reserved, fault, moving
            )
        if addr == Reg.GOAL_POSITION:
            return self.goal[mid].to_bytes(2, "little")
        if addr == Reg.TORQUE_ENABLE:
            return bytes([self.torque[mid]])
        raise AssertionError(f"unexpected read addr={addr}")

    def write_reg(self, mid, addr, data):
        self.writes.append((mid, addr, bytes(data)))
        if addr == Reg.GOAL_POSITION:
            self.goal[mid] = int.from_bytes(data, "little")
        elif addr == Reg.TORQUE_ENABLE:
            self.torque[mid] = data[0]

    def close(self):
        pass


@pytest.fixture
def arm(monkeypatch):
    monkeypatch.setattr(simple_arm, "StsBus", FakeBus)
    return SimpleArm()


@pytest.fixture
def tools_by_name(arm):
    return {tool.__name__: tool for tool in make_arm_tools(arm)}


def test_get_servo_angles_initial(tools_by_name):
    result = tools_by_name["get_servo_angles"]()
    assert result == {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
        "gripper_opening_percent": 50.0,  # midpoint of travel = half open
    }


def test_set_servo_angles_moves_only_named_joints(arm, tools_by_name):
    result = tools_by_name["set_servo_angles"](shoulder_lift=-40, wrist_flex=95)
    assert result["settled"] is True
    assert result["shoulder_lift"] == -40.0
    assert result["wrist_flex"] == 95.0
    assert result["shoulder_pan"] == 0.0  # untouched
    assert arm.bus.goal[2] == deg_to_raw(-40)
    assert arm.bus.goal[4] == deg_to_raw(95)
    assert arm.bus.goal[1] == MID_RAW  # pan never commanded
    assert all(arm.bus.torque.values())  # enable() turned all six on


def test_set_servo_angles_without_args_is_an_error(tools_by_name):
    result = tools_by_name["set_servo_angles"]()
    assert isinstance(result, str) and result.startswith("ERROR")


def test_set_gripper_opens_and_clamps(arm, tools_by_name):
    result = tools_by_name["set_gripper"](100)
    assert result == {"gripper_opening_percent": 100.0, "settled": True}
    assert arm.bus.goal[6] == MAX_RAW
    assert tools_by_name["set_gripper"](150)["gripper_opening_percent"] == 100.0
    assert tools_by_name["set_gripper"](-5)["gripper_opening_percent"] == 0.0
    assert arm.bus.goal[6] == MIN_RAW


def test_reset_arm_reaches_high_pose_and_lifts_before_pan(arm, tools_by_name):
    result = tools_by_name["reset_arm"]()
    assert result["settled"] is True
    # Expected angles come from HIGH_POSE itself (tuned on the real arm),
    # not hard-coded here; joints absent from HIGH_POSE keep their position.
    for joint, deg in SimpleArm.HIGH_POSE.items():
        assert result[joint.name.lower()] == pytest.approx(deg, abs=0.1)
    assert arm.bus.goal[6] != MID_RAW  # gripper was commanded open

    # Ordering: every lift/gripper goal write precedes the pan motion write.
    def last_goal_write(mid):
        return max(
            i for i, (m, a, _) in enumerate(arm.bus.writes) if m == mid and a == Reg.GOAL_POSITION
        )

    if Joint.SHOULDER_PAN in SimpleArm.HIGH_POSE:
        pan_idx = last_goal_write(Joint.SHOULDER_PAN.mid)
        for joint in SimpleArm.HIGH_POSE:
            if joint is not Joint.SHOULDER_PAN:
                assert last_goal_write(joint.mid) < pan_idx
        assert last_goal_write(6) < pan_idx  # gripper opens before the pan too


def test_bus_failure_returns_error_string(arm, tools_by_name):
    arm.bus.fail_reads = True
    for name in ("get_servo_angles", "set_gripper", "reset_arm"):
        tool = tools_by_name[name]
        result = tool(50) if name == "set_gripper" else tool()
        assert isinstance(result, str) and result.startswith("ERROR")
    result = tools_by_name["set_servo_angles"](shoulder_pan=10)
    assert isinstance(result, str) and result.startswith("ERROR")


def test_agent_can_call_arm_tools(arm):
    """End-to-end with pydantic-ai's TestModel: schemas build, tools execute."""
    agent = RobotAgent(model=TestModel(), tools=make_arm_tools(arm))
    result = agent.chat_sync("Reset the arm and report its angles.")
    assert isinstance(result.output, str) and result.output
    called = {
        part.tool_name
        for message in result.messages
        for part in message.parts
        if isinstance(part, ToolCallPart)
    }
    assert called == {"get_servo_angles", "set_servo_angles", "set_gripper", "reset_arm"}