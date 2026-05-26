import zmq
import time
import sys

def main(bind_ip="0.0.0.0", port=5555):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{bind_ip}:{port}")

    print(f"[SERVER] Listening on tcp://{bind_ip}:{port}")
    print("[SERVER] Waiting for client messages...\n")

    count = 0
    while True:
        try:
            message = socket.recv_string()
            count += 1
            print(f"[SERVER] Received ({count}): {message}")

            reply = f"ACK-{count}: got '{message}'"
            socket.send_string(reply)
            print(f"[SERVER] Replied: {reply}\n")

        except KeyboardInterrupt:
            print("\n[SERVER] Shutting down...")
            break

    socket.close()
    context.term()

if __name__ == "__main__":
    bind_ip = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5555
    main(bind_ip, port)
