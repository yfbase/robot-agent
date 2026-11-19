#!/usr/bin/env python3
"""极简 SO101 从臂控制，两层结构：

- StsBus    —— STS3215 总线协议层：串口、帧收发、校验和、寄存器读写；只认电机 ID。
- SimpleArm —— 关节语义层：Joint 枚举、自身限位、角度/开度换算、使能与运动。

只依赖 pyserial 与 construct。二进制布局用 construct 声明式定义
（STS_FEEDBACK / STS_GOAL / STS_CONFIG），寄存器地址由 Reg 枚举标识。

保留了什么：
- 响应包校验和验证（否则读回来的位置不可信）；
- 逐轴单播读写（本链路群读不可用），读操作失败自动重试一次（写绝不重试）；
- 退出时把六个电机扭矩关闭（close / with 退出）。

去掉了什么（相对于 arm_api.py）：
- 负载到阈即停、单次步长限制、限位收边夹紧、写入回读校验、
  临时参数保存/恢复、电压/温度/负载越界报错——目标与安全由使用者负责。
  （电机固件仍会按自身 EEPROM 限位截断目标，这是舵机自身行为。）

用法：
    from simple_arm import SimpleArm, Joint

    arm = SimpleArm()                              # 自动找 1a86 适配器；或 SimpleArm('COM5')
    arm.read_all()                                 # dict，键为 Joint 枚举
    arm.enable()                                   # 使能全部；单轴 arm.enable(Joint.GRIPPER)
    arm.enable(Joint.ELBOW_FLEX | Joint.WRIST_FLEX)# 组合使能（| 运算）
    arm.set_goal_raw(Joint.ELBOW_FLEX, 2050)       # 直接写目标刻度，无任何检查（仅单关节）
    arm.move(Joint.ELBOW_FLEX, 30)                 # 角度（0° = 该轴 EEPROM 行程中点）
    arm.gripper(40)                                # 夹爪开度 0(闭)..100(开)
    arm.wait(Joint.ELBOW_FLEX | Joint.WRIST_FLEX)  # 等多轴各自到目标 ±tol
    arm.disable(Joint.ELBOW_FLEX | Joint.GRIPPER)  # 组合卸力
    arm.close()                                    # 全部卸力 + 关串口

Joint 为 Flag 枚举：enable/disable/wait 接受 `|` 组合，默认 Joint（全部六轴）；
move/set_goal_raw/read_one 仅接受单个成员（is_single 判断）。

命令行：python simple_arm.py [串口]   # 打印状态后执行 main() 中的测试动作
"""
import os
import sys
import time
from enum import Flag, IntEnum

import serial
from serial.tools import list_ports
from construct import Adapter, Int8ul, Int16ul, Pointer, Struct


class Joint(Flag):
    """六轴标识；位标志枚举，可用 | 组合（如 Joint.ELBOW_FLEX | Joint.GRIPPER）。
    mid 属性即电机 ID（总线 ID 1..6）；is_single 判断是否恰好为单个成员。"""
    SHOULDER_PAN = 1
    SHOULDER_LIFT = 2
    ELBOW_FLEX = 4
    WRIST_FLEX = 8
    WRIST_ROLL = 16
    GRIPPER = 32

    @property
    def mid(self):
        """电机 ID（1..6）。仅对单个成员有意义；组合值请先迭代展开。"""
        return self.value.bit_length()

    @property
    def is_single(self):
        """是否恰好为单个成员（位值为 2 的幂即单个成员；组合或空为 False）。"""
        v = self.value
        return v != 0 and v & (v - 1) == 0


class Reg(IntEnum):
    """STS3215 寄存器地址；IntEnum 成员即整数，可直接用于协议包与切片。
    反馈区（PRESENT_FEEDBACK 起 15 字节）内部布局：
    [0:2] 位置  [2:4] 速度  [4:6] 负载  [6] 电压(0.1V)  [7] 温度℃  [9] 报警位  [10] 运动中。"""
    CONFIG = 0             # 34 字节配置区快照（含限位）
    MIN_POSITION = 9       # 2 字节：位置下限（EEPROM）
    MAX_POSITION = 11      # 2 字节：位置上限（EEPROM）
    TORQUE_ENABLE = 40     # 1 字节：扭矩使能（0/1）
    ACCELERATION = 41      # 1 字节：加速度
    GOAL_POSITION = 42     # 2 字节：目标位置
    GOAL_SPEED = 46        # 2 字节：速度上限
    OUTPUT_LIMIT = 48      # 2 字节：输出上限
    PRESENT_FEEDBACK = 56  # 15 字节：实时反馈区


class StsSigned(Adapter):
    """STS3215 有符号幅值编码：符号位在第 bits 位，幅值为低 bits 位。"""
    def __init__(self, subcon, bits):
        super().__init__(subcon)
        self.bits = bits

    def _decode(self, obj, ctx, path):
        mask = (1 << self.bits) - 1
        return -(obj & mask) if obj & (1 << self.bits) else obj

    def _encode(self, obj, ctx, path):
        mask = (1 << self.bits) - 1
        return ((1 << self.bits) if obj < 0 else 0) | (abs(obj) & mask)


# 反馈区（寄存器 56 起）：位置/速度 15 位有符号，负载 10 位有符号。
STS_FEEDBACK = Struct(
    'position' / StsSigned(Int16ul, 15),
    'speed' / StsSigned(Int16ul, 15),
    'load' / StsSigned(Int16ul, 10),
    'voltage_x10' / Int8ul,
    'temperature' / Int8ul,
    'reserved' / Int8ul,
    'fault' / Int8ul,
    'moving' / Int8ul,
)

# 目标位置寄存器（2 字节，15 位有符号）。
STS_GOAL = Struct('value' / StsSigned(Int16ul, 15))

# 配置区（寄存器 0 起，34 字节）：Pointer 直接按寄存器地址取字段，允许间隙。
STS_CONFIG = Struct(
    'min_position' / Pointer(Reg.MIN_POSITION, Int16ul),
    'max_position' / Pointer(Reg.MAX_POSITION, Int16ul),
    'p_gain' / Pointer(21, Int8ul),
    'homing_offset' / Pointer(31, StsSigned(Int16ul, 11)),
    'mode' / Pointer(33, Int8ul),
)


def _packet(mid, instr, params=b''):
    body = bytes([mid, len(params) + 2, instr]) + params
    return b'\xff\xff' + body + bytes([(~sum(body)) & 0xFF])


class StsBus:
    """STS3215 总线协议层：串口、帧收发、校验和、寄存器读写。只认电机 ID。"""

    BAUD = 1_000_000
    TIMEOUT = 0.15
    RETRY = 1   # 读操作失败重试次数；写操作绝不重试

    def __init__(self, port=None):
        self.ser = serial.Serial(port or self._find_port(), self.BAUD, timeout=self.TIMEOUT,
                                 bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE)

    @staticmethod
    def _find_port():
        if os.name != 'nt':
            return '/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086137-if00'
        hits = [p.device for p in list_ports.comports() if p.vid == 0x1A86]
        if len(hits) != 1:
            raise OSError(f'期望恰好一个 1a86 串口适配器，实际 {hits}；请显式传 port='
                          + '/'.join(hits) if hits else '未发现串口设备')
        return hits[0]

    def read_reg(self, mid, addr, count):
        err, data = self._exchange_retry(mid, 2, bytes([addr, count]))
        if err or len(data) != count:
            raise RuntimeError(f'ID{mid} 读 addr={addr} 失败 err={err} len={len(data)}')
        return data

    def write_reg(self, mid, addr, data):
        err, _ = self._exchange_retry(mid, 3, bytes([addr]) + bytes(data), want=0)
        if err:
            raise RuntimeError(f'ID{mid} 写 addr={addr} 失败 err={err}')

    def _exchange_retry(self, mid, instr, params=b'', want=None):
        """读失败（超时/坏帧）重试 RETRY 次；写只执行一次。"""
        attempts = 1 + (self.RETRY if instr == 2 else 0)
        for attempt in range(attempts):
            try:
                return self._exchange(mid, instr, params, want)
            except (TimeoutError, RuntimeError):
                if attempt == attempts - 1:
                    raise

    def _exchange(self, mid, instr, params=b'', want=None):
        if want is None:
            want = params[1] if instr == 2 else 0
        self.ser.reset_input_buffer()
        self.ser.write(_packet(mid, instr, params))
        deadline = time.monotonic() + self.TIMEOUT
        buf = bytearray()
        while time.monotonic() < deadline:
            chunk = self.ser.read(512)
            if chunk:
                buf.extend(chunk)
                frame = self._extract(buf, mid, want)
                if frame is not None:
                    return frame
        raise TimeoutError(f'ID{mid} 无有效响应（收到 {bytes(buf).hex()}）')

    @staticmethod
    def _extract(buf, mid, want):
        while True:
            if len(buf) < 4:
                return None
            if buf[0] != 0xFF or buf[1] != 0xFF:
                del buf[0]
                continue
            length = buf[3]
            if not 2 <= length <= 100 or len(buf) < length + 4:
                return None
            frame = bytes(buf[:length + 4])
            if sum(frame[2:]) & 0xFF != 255:
                del buf[0]
                continue
            del buf[:length + 4]
            if frame[2] == mid and length == want + 2:
                return frame[4], frame[5:-1]

    def close(self):
        self.ser.close()


class SimpleArm:
    """关节语义层：Joint 枚举、自身 EEPROM 限位、角度/开度换算、使能与运动。"""

    # 运动参数：写入电机 RAM（断电自动还原），enable() 时生效。
    SPEED = 180             # 速度上限（全部轴）
    ACCELERATION = 10       # 加速度
    ARM_OUTPUT = 300        # 五个臂轴的输出上限
    GRIPPER_OUTPUT = 100    # 夹爪的输出上限（小、易顶死，给更小）

    def __init__(self, port=None):
        self.bus = StsBus(port)
        self.limits = {j: self._read_limits(j.mid) for j in Joint}

    def _read_limits(self, mid):
        cfg = STS_CONFIG.parse(self.bus.read_reg(mid, Reg.CONFIG, 34))
        return cfg.min_position, cfg.max_position

    # ---------- 换算：角度/开度 <-> 编码器刻度，各只写一遍 ----------

    def _to_raw(self, joint, deg):
        lo, hi = self.limits[joint]
        return round(deg * 4095 / 360 + (lo + hi) / 2)

    def _to_deg(self, joint, raw):
        lo, hi = self.limits[joint]
        return (raw - (lo + hi) / 2) * 360 / 4095

    def _pct_to_raw(self, percent):
        lo, hi = self.limits[Joint.GRIPPER]
        return round(lo + (hi - lo) * percent / 100)

    def _raw_to_pct(self, raw):
        lo, hi = self.limits[Joint.GRIPPER]
        return (raw - lo) * 100 / (hi - lo)

    # ---------- 读 ----------

    def read_one(self, joint):
        """单轴完整状态。joint 须为单个 Joint 成员。"""
        if not joint.is_single:
            raise ValueError(f'该操作只接受单个关节，收到组合：{[m.name for m in joint]}')
        mid = joint.mid
        fb = STS_FEEDBACK.parse(self.bus.read_reg(mid, Reg.PRESENT_FEEDBACK, 15))
        gripper = joint is Joint.GRIPPER
        return dict(
            id=mid, joint=joint, raw=fb.position,
            deg=None if gripper else self._to_deg(joint, fb.position),
            gripper_pct=self._raw_to_pct(fb.position) if gripper else None,
            load=fb.load,
            voltage_V=fb.voltage_x10 / 10, temperature_C=fb.temperature,
            goal_raw=STS_GOAL.parse(self.bus.read_reg(mid, Reg.GOAL_POSITION, 2)).value,
            torque_on=self.bus.read_reg(mid, Reg.TORQUE_ENABLE, 1) == b'\x01',
            moving=bool(fb.moving))

    def read_all(self):
        return {j: self.read_one(j) for j in Joint}

    def read_raw(self):
        return [self.read_one(j)['raw'] for j in Joint]

    # ---------- 使能 / 卸力 ----------

    def enable(self, joints=Joint):
        """使能指定轴：joints 为 Joint（可用 | 组合），默认 Joint（全部六轴）。
        写 RAM 运动参数（SPEED/ACCELERATION/输出上限，断电自动还原），
        预置当前位姿为目标后开扭矩。"""
        for member in joints:
            mid = member.mid
            cur = STS_FEEDBACK.parse(self.bus.read_reg(mid, Reg.PRESENT_FEEDBACK, 15)).position
            self.bus.write_reg(mid, Reg.ACCELERATION, bytes([self.ACCELERATION]))
            self.bus.write_reg(mid, Reg.GOAL_SPEED, self.SPEED.to_bytes(2, 'little'))
            output = self.GRIPPER_OUTPUT if member is Joint.GRIPPER else self.ARM_OUTPUT
            self.bus.write_reg(mid, Reg.OUTPUT_LIMIT, output.to_bytes(2, 'little'))
            self.bus.write_reg(mid, Reg.GOAL_POSITION, cur.to_bytes(2, 'little'))
            self.bus.write_reg(mid, Reg.TORQUE_ENABLE, b'\x01')

    def disable(self, joints=Joint):
        """卸力指定轴：joints 为 Joint（可用 | 组合），默认 Joint（全部六轴）。"""
        for member in joints:
            self.bus.write_reg(member.mid, Reg.TORQUE_ENABLE, b'\x00')

    # ---------- 运动 ----------

    def set_goal_raw(self, joint, raw):
        """直接写目标刻度（仅单关节）。无范围检查、无限位收边（固件仍按自身 EEPROM 限位截断）。"""
        if not joint.is_single:
            raise ValueError(f'该操作只接受单个关节，收到组合：{[m.name for m in joint]}')
        self.bus.write_reg(joint.mid, Reg.GOAL_POSITION, int(raw).to_bytes(2, 'little'))

    def move(self, joint, deg):
        """角度目标：0° = 该轴 EEPROM 行程中点（仅单关节）。"""
        self.set_goal_raw(joint, self._to_raw(joint, deg))

    def gripper(self, percent):
        """夹爪开度 0(闭)..100(开)。"""
        self.set_goal_raw(Joint.GRIPPER, self._pct_to_raw(percent))

    def wait(self, joints=Joint, *, tol=8, timeout=3.):
        """轮询等待这些轴各自进入目标 ±tol 刻度：joints 为 Joint（可用 | 组合），
        默认 Joint（全部六轴）。只看位置，不看负载；error_ticks 以 Joint 成员为键。"""
        members = list(joints)
        deadline = time.monotonic() + timeout
        while True:
            errors = {}
            for member in members:
                mid = member.mid
                raw = STS_FEEDBACK.parse(self.bus.read_reg(mid, Reg.PRESENT_FEEDBACK, 15)).position
                goal = STS_GOAL.parse(self.bus.read_reg(mid, Reg.GOAL_POSITION, 2)).value
                errors[member] = raw - goal
            if all(abs(e) <= tol for e in errors.values()):
                return dict(settled=True, error_ticks=errors)
            if time.monotonic() > deadline:
                return dict(settled=False, error_ticks=errors)
            time.sleep(.05)

    # ---------- 收尾 ----------

    def close(self):
        try:
            self.disable()
        finally:
            self.bus.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else None
    with SimpleArm(port) as arm:
        for joint, row in arm.read_all().items():
            value = f"{row['deg']:+.1f}°" if row['deg'] is not None else f"{row['gripper_pct']:.1f}%"
            print(f"{row['id']} {joint.name}: raw={row['raw']} {value} 扭矩开={row['torque_on']} "
                  f"{row['temperature_C']}°C {row['voltage_V']}V 负载{row['load']} "
                  f"目标{row['goal_raw']} 限位{arm.limits[joint][0]}..{arm.limits[joint][1]}")
        arm.enable(Joint.GRIPPER)
        arm.gripper(20)
        print(arm.wait(Joint.GRIPPER))     # {'settled': True/False, 'error_ticks': ...}
        print(arm.read_one(Joint.GRIPPER)) # 顺带看停住时的负载和温度


if __name__ == '__main__':
    main()
