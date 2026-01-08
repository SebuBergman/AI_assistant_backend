import socket

host = "db.qrddxaypivwpxyrixxuk.supabase.co"
try:
    ip = socket.gethostbyname(host)
    print(f"Resolved {host} to {ip}")
except Exception as e:
    print(f"DNS failed: {e}")
