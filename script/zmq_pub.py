import zmq
import time
import sys

def main(bind_ip="0.0.0.0", port=5556):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://{bind_ip}:{port}")

    print(f"[PUB] Broadcasting on tcp://{bind_ip}:{port}")
    print("[PUB] Sending messages every 1 second...\n")

    count = 0
    while True:
        try:
            count += 1
            message = f"broadcast #{count} at {time.time():.2f}"
            socket.send_string(message)
            print(f"[PUB] Sent: {message}")
            time.sleep(1)

        except KeyboardInterrupt:
            print("\n[PUB] Shutting down...")
            break

    socket.close()
    context.term()

if __name__ == "__main__":
    bind_ip = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5556
    main(bind_ip, port)
