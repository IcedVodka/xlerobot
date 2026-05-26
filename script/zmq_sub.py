import zmq
import sys

def main(pub_ip="localhost", port=5556):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{pub_ip}:{port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    print(f"[SUB] Subscribed to tcp://{pub_ip}:{port}")
    print("[SUB] Listening for broadcasts...\n")

    count = 0
    while True:
        try:
            message = socket.recv_string()
            count += 1
            print(f"[SUB] Received ({count}): {message}")

        except KeyboardInterrupt:
            print("\n[SUB] Shutting down...")
            break

    socket.close()
    context.term()

if __name__ == "__main__":
    pub_ip = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5556
    main(pub_ip, port)
