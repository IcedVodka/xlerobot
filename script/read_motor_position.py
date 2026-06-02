#!/usr/bin/env python
"""
读取舵机位置 - 高性能版（GroupSyncRead 批量读取 + 实时刷新）

用法:
    python read_motor_position.py --port /dev/ttyACM0
    python read_motor_position.py --port /dev/ttyACM0 --watch
    python read_motor_position.py --port /dev/ttyACM0 --watch --interval 0.05
"""

import argparse
import sys
import time


def _model_number_to_protocol(model_nb: int) -> int:
    """根据型号编号推断协议版本"""
    protocol0_models = {777, 2825, 11272}  # sts3215, sts3250, sm85-12bl
    protocol1_models = {1284, 1285}         # scs0009, scs0015
    if model_nb in protocol0_models:
        return 0
    if model_nb in protocol1_models:
        return 1
    # 默认尝试 protocol 0
    return 0


def _model_number_to_name(nb: int) -> str:
    names = {777: "STS3215", 2825: "STS3250", 11272: "SM85-12BL", 1284: "SCS0009", 1285: "SCS0015"}
    return names.get(nb, f"Unknown({nb})")


class MotorReader:
    """高性能舵机位置读取器，保持连接，支持批量读取"""

    def __init__(self, port: str):
        import scservo_sdk as scs

        self.port = port
        self.port_handler = scs.PortHandler(port)
        self.ph0 = scs.PacketHandler(0)  # Protocol 0
        self.ph1 = scs.PacketHandler(1)  # Protocol 1
        self._sync_reader = None  # GroupSyncRead，延迟初始化
        self.motors = []          # [(id, model_nb, protocol), ...]
        self._connected = False

    def connect(self, baudrate: int) -> bool:
        if not self.port_handler.openPort():
            return False
        self.port_handler.setBaudRate(baudrate)
        self._connected = True
        return True

    def disconnect(self):
        if self._connected:
            self.port_handler.closePort()
            self._connected = False

    def scan(self, target_baudrate: int | None = None) -> list:
        """扫描端口，返回 [(id, model_nb, protocol, baudrate), ...]"""
        import scservo_sdk as scs

        scan_baudrates = [target_baudrate] if target_baudrate else [
            1_000_000, 500_000, 250_000, 128_000, 115_200,
            57_600, 38_400, 19_200, 14_400, 9_600,
        ]

        found = []
        seen_ids = set()

        for br in scan_baudrates:
            self.port_handler.setBaudRate(br)

            # Broadcast Ping (Protocol 0)
            txpacket = [0] * 6
            txpacket[scs.PKT_ID] = scs.BROADCAST_ID
            txpacket[scs.PKT_LENGTH] = 2
            txpacket[scs.PKT_INSTRUCTION] = scs.INST_PING

            result = self.ph0.txPacket(self.port_handler, txpacket)
            if result != scs.COMM_SUCCESS:
                continue

            # 接收响应
            status_length = 6
            wait_length = status_length * (scs.MAX_ID + 1)
            tx_time = (1000.0 / self.port_handler.getBaudRate()) * 10.0
            self.port_handler.setPacketTimeoutMillis((wait_length * tx_time) + (3.0 * scs.MAX_ID) + 16.0)

            rxpacket = []
            while not self.port_handler.isPacketTimeout():
                rxpacket += self.port_handler.readPort(wait_length - len(rxpacket))
                if len(rxpacket) >= wait_length:
                    break
            self.port_handler.is_using = False

            # 解析响应
            while len(rxpacket) >= status_length:
                idx = 0
                while idx < len(rxpacket) - 1 and not (rxpacket[idx] == 0xFF and rxpacket[idx + 1] == 0xFF):
                    idx += 1
                if idx >= len(rxpacket) - 1:
                    break
                if len(rxpacket) - idx < status_length:
                    break

                checksum = sum(rxpacket[idx + 2:idx + status_length - 1]) & 0xFF
                checksum = ~checksum & 0xFF
                if rxpacket[idx + status_length - 1] == checksum:
                    motor_id = rxpacket[idx + scs.PKT_ID]
                    if motor_id != scs.BROADCAST_ID and motor_id not in seen_ids:
                        model_nb, comm, _ = self.ph0.read2ByteTxRx(self.port_handler, motor_id, 3)
                        if comm == scs.COMM_SUCCESS:
                            protocol = _model_number_to_protocol(model_nb)
                            found.append((motor_id, model_nb, protocol, br))
                            seen_ids.add(motor_id)
                        else:
                            # protocol 0 读型号失败，尝试 protocol 1
                            model_nb, comm, _ = self.ph1.read2ByteTxRx(self.port_handler, motor_id, 3)
                            if comm == scs.COMM_SUCCESS:
                                found.append((motor_id, model_nb, 1, br))
                                seen_ids.add(motor_id)
                        rxpacket = rxpacket[idx + status_length:]
                    else:
                        rxpacket = rxpacket[idx + status_length:]
                else:
                    rxpacket = rxpacket[idx + 2:]

        self.motors = [(m[0], m[1], m[2]) for m in found]  # (id, model, protocol)
        return found

    def read_positions_fast(self) -> dict[int, int | None]:
        """高性能读取：Protocol0 用 GroupSyncRead 批量读，Protocol1 逐个读"""
        import scservo_sdk as scs

        results = {}
        p0_motors = [m for m in self.motors if m[2] == 0]
        p1_motors = [m for m in self.motors if m[2] == 1]

        # --- Protocol 0: GroupSyncRead 批量读取 ---
        if p0_motors:
            if self._sync_reader is None:
                self._sync_reader = scs.GroupSyncRead(self.port_handler, self.ph0, 56, 2)

            self._sync_reader.clearParam()
            for mid, _, _ in p0_motors:
                self._sync_reader.addParam(mid)

            comm = self._sync_reader.txRxPacket()
            if comm == scs.COMM_SUCCESS:
                for mid, _, _ in p0_motors:
                    if self._sync_reader.isAvailable(mid, 56, 2):
                        pos = self._sync_reader.getData(mid, 56, 2)
                        results[mid] = pos
                    else:
                        results[mid] = None
            else:
                # GroupSyncRead 失败，逐个 fallback
                for mid, _, _ in p0_motors:
                    pos, comm, _ = self.ph0.read2ByteTxRx(self.port_handler, mid, 56)
                    results[mid] = pos if comm == scs.COMM_SUCCESS else None

        # --- Protocol 1: 逐个读取 ---
        for mid, _, _ in p1_motors:
            pos, comm, _ = self.ph1.read2ByteTxRx(self.port_handler, mid, 56)
            results[mid] = pos if comm == scs.COMM_SUCCESS else None

        return results


def main():
    parser = argparse.ArgumentParser(description="读取舵机位置（高性能版）")
    parser.add_argument("--port", type=str, required=True, help="串口路径")
    parser.add_argument("--baudrate", type=int, default=None, help="指定波特率（不指定则扫描）")
    parser.add_argument("--watch", action="store_true", help="实时刷新模式")
    parser.add_argument("--interval", type=float, default=0.05, help="刷新间隔秒数（默认 0.05 = 20Hz）")
    args = parser.parse_args()

    reader = MotorReader(args.port)

    print(f"扫描端口: {args.port}")
    if not reader.connect(args.baudrate or 1_000_000):
        print(f"无法打开端口: {args.port}")
        sys.exit(1)

    found = reader.scan(args.baudrate)

    if not found:
        print("未发现任何舵机")
        reader.disconnect()
        sys.exit(1)

    # 统一波特率
    baudrates = sorted(set(m[3] for m in found))
    use_br = baudrates[0]
    if len(baudrates) > 1:
        print(f"注意: 发现多个波特率 {baudrates}，使用 {use_br}")
    reader.port_handler.setBaudRate(use_br)

    print(f"\n发现 {len(found)} 个舵机:")
    for mid, model_nb, protocol, br in found:
        name = _model_number_to_name(model_nb)
        print(f"  ID={mid:3d}  型号={name:12s}  Protocol={protocol}  波特率={br:,}")

    if not args.watch:
        # 单次读取
        positions = reader.read_positions_fast()
        print("\n当前位置:")
        for mid, _, _ in reader.motors:
            pos = positions.get(mid)
            print(f"  ID {mid:3d}: {pos if pos is not None else 'ERR'}")
        reader.disconnect()
        return

    # --- 实时监控模式 ---
    print(f"\n实时监控模式 (目标 {1/args.interval:.0f}Hz)，按 Ctrl+C 退出\n")

    # 打印表头
    header_parts = [f"ID{m[0]:2d}" for m in reader.motors]
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    try:
        while True:
            t0 = time.perf_counter()
            positions = reader.read_positions_fast()
            dt = time.perf_counter() - t0

            # 构建状态行
            parts = []
            for mid, _, _ in reader.motors:
                pos = positions.get(mid)
                if pos is not None:
                    parts.append(f"{pos:5d}")
                else:
                    parts.append("  ERR")

            hz = 1.0 / dt if dt > 0 else 999
            status_line = " | ".join(parts)
            print(f"\r[{hz:5.1f}Hz] {status_line}", end="", flush=True)

            # 精确睡眠
            sleep_time = args.interval - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n已退出")
    finally:
        reader.disconnect()


if __name__ == "__main__":
    main()
