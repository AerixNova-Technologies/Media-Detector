import socket

def check_port(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)
    result = sock.connect_ex((ip, port))
    sock.close()
    return result == 0

ips = ["157.48.122.14", "157.51.100.27"]
ports_to_check = [80, 554, 8000, 48961, 48962, 27906]

for ip in ips:
    print(f"Checking ports on {ip}...")
    for port in ports_to_check:
        is_open = check_port(ip, port)
        status = "OPEN" if is_open else "CLOSED"
        print(f"  Port {port}: {status}")
