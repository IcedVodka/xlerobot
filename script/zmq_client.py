import zmq
import time
import sys

def main(server_ip="localhost", port=5555):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{server_ip}:{port}")

    print(f"[CLIENT] Connected to tcp://{server_ip}:{port}")
    print("[CLIENT] Sending messages every 2 seconds...\n")

    count = 0
    while True:
        try:
            count += 1
            message = f"Hello from client, msg #{count}"
            socket.send_string(message)
            print(f"[CLIENT] Sent: {message}")

            reply = socket.recv_string()
            print(f"[CLIENT] Received: {reply}\n")

            time.sleep(2)

        except KeyboardInterrupt:
            print("\n[CLIENT] Shutting down...")
            break

    socket.close()
    context.term()

if __name__ == "__main__":
    server_ip = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5555
    main(server_ip, port)
