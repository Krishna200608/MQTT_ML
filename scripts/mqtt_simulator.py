# mqtt_simulator.py
import time
import json
import argparse
from paho.mqtt import client as mqtt_client
import random
import socket

def publish_loop(broker='127.0.0.1', port=1883, client_id='sim1', topic='sensors/temp', rate=1.0):
    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, client_id=client_id) # type: ignore
    client.connect(broker, port, keepalive=60)
    client.loop_start()
    try:
        while True:
            payload = {
                "device_id": client_id,
                "ts": int(time.time()),
                "temperature": round(20 + random.random()*5, 2),
                "humidity": round(30 + random.random()*10, 2)
            }
            client.publish(topic, json.dumps(payload))
            print("Published:", payload)
            time.sleep(rate)
    except KeyboardInterrupt:
        print("Stopped simulator")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--broker', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=1883)
    parser.add_argument('--client', default='sim1')
    parser.add_argument('--topic', default='sensors/temp')
    parser.add_argument('--rate', type=float, default=1.0)
    args = parser.parse_args()
    publish_loop(args.broker, args.port, args.client, args.topic, args.rate)
