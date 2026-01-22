
import random
import time
from ai2thor.controller import Controller

print("Testing AI2-THOR Controller Initialization...")
port = random.randint(9000, 10000)
print(f"Port: {port}")

try:
    c = Controller(scene="FloorPlan1", headless=True, port=port)
    print("SUCCESS: Controller initialized!")
    c.stop()
except Exception as e:
    print(f"FAILURE: {e}")
