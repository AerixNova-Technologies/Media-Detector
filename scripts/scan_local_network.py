import socket
import threading
from queue import Queue

def scan_port(ip, port, results):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((ip, port)) == 0:
                results.put((ip, port))
    except:
        pass

def scan_network():
    # Attempting to scan the 192.168.31.x subnet
    base_ip = "192.168.31."
    ports = [554, 8000, 80]
    results = Queue()
    threads = []

    print(f"Scanning {base_ip}0/24 for ports {ports}...")
    
    for i in range(1, 255):
        ip = base_ip + str(i)
        for port in ports:
            t = threading.Thread(target=scan_port, args=(ip, port, results))
            t.start()
            threads.append(t)
            
            # Limit number of active threads
            if len(threads) > 100:
                for t in threads:
                    t.join()
                threads = []

    for t in threads:
        t.join()

    found = []
    while not results.empty():
        found.append(results.get())
    
    if found:
        print("\nFound devices:")
        for ip, port in found:
            print(f"  {ip}:{port}")
    else:
        print("\nNo devices found in this range.")

if __name__ == "__main__":
    scan_network()
