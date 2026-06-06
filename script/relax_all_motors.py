#!/usr/bin/env python
"""
放松 XLerobot 全部舵机（Torque Off + Lock Off）

用途：
    - 程序异常退出后舵机锁死，需要手动放松
    - 启动 xlerobot_host 前确认舵机状态
    - 某台舵机无响应（There is no status packet!）时先重置

原理：
    1. 广播发送 Torque_Enable=0 到所有舵机（总线上的舵机都会接收）
    2. 广播发送 Lock=0 到所有舵机
    3. 再逐个发送一次（不需要响应包，不会报错）
    4. 最后尝试 ping 确认哪些舵机在线

用法：
    # 使用默认端口（/dev/serial/by-id/...）
    python script/relax_all_motors.py

    # 指定端口
    python script/relax_all_motors.py \
        --port1=/dev/ttyACM0 \
        --port2=/dev/ttyACM1

    # 只放松 bus1（左臂+头部）
    python script/relax_all_motors.py --only-bus1

    # 只放松 bus2（右臂+底盘）
    python script/relax_all_motors.py --only-bus2
"""

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# STS3215 控制表地址
TORQUE_ENABLE_ADDR = 40
LOCK_ADDR = 55


def find_serial_ports():
    """查找可用的串口设备。"""
    # 优先稳定路径
    stable = sorted(glob.glob("/dev/serial/by-id/*"))
    if stable:
        return stable
    # 回退到动态路径
    dynamic = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    return dynamic


def auto_detect_ports():
    """自动检测两个串口。"""
    ports = find_serial_ports()
    if len(ports) >= 2:
        return ports[0], ports[1]
    elif len(ports) == 1:
        print("[WARN] 只检测到一个串口，bus1 和 bus2 将共用同一端口")
        return ports[0], ports[0]
    else:
        raise RuntimeError("未检测到任何串口设备。请检查 USB 连接。")


def relax_motors(port: str, motor_ids: list[int], baudrate: int = 1_000_000) -> list[int]:
    """
    放松指定总线上的舵机。

    Args:
        port: 串口路径
        motor_ids: 该总线上的舵机 ID 列表
        baudrate: 波特率

    Returns:
        在线的舵机 ID 列表
    """
    import scservo_sdk as scs

    port_handler = scs.PortHandler(port)
    packet_handler = scs.PacketHandler(protocol_version=0)

    if not port_handler.openPort():
        print(f"  [ERROR] 无法打开串口 {port}")
        return []

    port_handler.setBaudRate(baudrate)
    print(f"  [OK] 串口已打开: {port} @ {baudrate} baud")

    # 1. 广播 Torque_Enable=0（所有在线舵机会收到）
    packet_handler.write1ByteTxOnly(port_handler, scs.BROADCAST_ID, TORQUE_ENABLE_ADDR, 0)
    print(f"  [OK] 广播 Torque_Enable=0 (ID={scs.BROADCAST_ID})")

    # 2. 广播 Lock=0
    packet_handler.write1ByteTxOnly(port_handler, scs.BROADCAST_ID, LOCK_ADDR, 0)
    print(f"  [OK] 广播 Lock=0 (ID={scs.BROADCAST_ID})")

    # 3. 逐个发送（不需要响应，即使舵机锁死也能发送）
    for motor_id in motor_ids:
        packet_handler.write1ByteTxOnly(port_handler, motor_id, TORQUE_ENABLE_ADDR, 0)
        packet_handler.write1ByteTxOnly(port_handler, motor_id, LOCK_ADDR, 0)

    print(f"  [OK] 已逐个发送 Torque_Enable=0 + Lock=0 到 IDs: {motor_ids}")

    # 4. 尝试 ping 确认在线状态
    print(f"  [INFO] 正在检测舵机在线状态...")
    online_ids = []
    for motor_id in motor_ids:
        try:
            model, comm, error = packet_handler.ping(port_handler, motor_id)
            if comm == scs.COMM_SUCCESS:
                online_ids.append(motor_id)
                print(f"    [OK]   ID={motor_id} 在线 (model={model})")
            else:
                print(f"    [WARN] ID={motor_id} 无响应 ({packet_handler.getTxRxResult(comm)})")
        except Exception as e:
            print(f"    [WARN] ID={motor_id} ping 异常: {e}")

    port_handler.closePort()
    print(f"  [INFO] 串口已关闭")

    return online_ids


def main():
    parser = argparse.ArgumentParser(description="放松 XLerobot 全部舵机")
    parser.add_argument("--port1", type=str, default=None, help="bus1 串口（左臂+头部）")
    parser.add_argument("--port2", type=str, default=None, help="bus2 串口（右臂+底盘）")
    parser.add_argument("--only-bus1", action="store_true", help="只放松 bus1")
    parser.add_argument("--only-bus2", action="store_true", help="只放松 bus2")
    parser.add_argument("--baudrate", type=int, default=1_000_000, help="波特率")
    args = parser.parse_args()

    # 默认端口（xlerobot 配置中的默认值）
    default_port1 = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D042095-if00"
    default_port2 = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D045917-if00"

    if args.port1:
        port1 = args.port1
    else:
        try:
            port1 = default_port1 if glob.glob(default_port1) else auto_detect_ports()[0]
        except RuntimeError:
            port1 = None

    if args.port2:
        port2 = args.port2
    else:
        try:
            port2 = default_port2 if glob.glob(default_port2) else auto_detect_ports()[1]
        except RuntimeError:
            port2 = None

    print("=" * 50)
    print("XLerobot 舵机放松工具")
    print("=" * 50)
    print(f"Bus1 (左臂+头部): {port1 or '未指定'}")
    print(f"Bus2 (右臂+底盘): {port2 or '未指定'}")
    print()

    # bus1 电机: 左臂 ID 1-6, 头部 ID 7-8
    bus1_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    # bus2 电机: 右臂 ID 1-6, 底盘 ID 7-9
    bus2_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    all_online = {}

    if not args.only_bus2 and port1:
        print("[Bus1] 左臂(1-6) + 头部(7-8)")
        online1 = relax_motors(port1, bus1_ids, args.baudrate)
        all_online["bus1"] = online1
        print()

    if not args.only_bus1 and port2:
        print("[Bus2] 右臂(1-6) + 底盘(7-9)")
        online2 = relax_motors(port2, bus2_ids, args.baudrate)
        all_online["bus2"] = online2
        print()

    # 汇总
    print("=" * 50)
    print("结果汇总")
    print("=" * 50)
    total = len(bus1_ids) + len(bus2_ids)
    online_total = 0
    if "bus1" in all_online:
        missing1 = [i for i in bus1_ids if i not in all_online["bus1"]]
        print(f"Bus1 在线: {len(all_online['bus1'])}/{len(bus1_ids)}  IDs: {all_online['bus1']}")
        if missing1:
            print(f"       缺失: {missing1}")
        online_total += len(all_online["bus1"])
    if "bus2" in all_online:
        missing2 = [i for i in bus2_ids if i not in all_online["bus2"]]
        print(f"Bus2 在线: {len(all_online['bus2'])}/{len(bus2_ids)}  IDs: {all_online['bus2']}")
        if missing2:
            print(f"       缺失: {missing2}")
        online_total += len(all_online["bus2"])

    print(f"\n总计在线: {online_total}/{total}")

    if online_total < total:
        print("\n[提示] 如果有舵机缺失：")
        print("  1. 检查该舵机的电源线和信号线是否松动")
        print("  2. 检查总线末端是否有 120Ω 终端电阻")
        print("  3. 尝试断电重启后再运行本脚本")
        print("  4. 如果某个舵机 ID 冲突，需要重新配置 ID")
    else:
        print("\n[OK] 所有舵机已放松，现在可以手动转动机械臂了。")


if __name__ == "__main__":
    main()
