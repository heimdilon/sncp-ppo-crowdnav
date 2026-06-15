import numpy as np
import time

def list_comp(n, pos, vel, rad):
    for i in range(n):
        neighbors = [
            (pos[j], vel[j], float(rad[j]))
            for j in range(n) if j != i
        ]

def array_mask(n, pos, vel, rad):
    for i in range(n):
        mask = np.arange(n) != i
        p = pos[mask]
        v = vel[mask]
        r = rad[mask]

start = time.time()
n = 100
pos = np.random.rand(n, 2)
vel = np.random.rand(n, 2)
rad = np.random.rand(n)
for _ in range(1000):
    list_comp(n, pos, vel, rad)
print("list comp:", time.time() - start)

start = time.time()
for _ in range(1000):
    array_mask(n, pos, vel, rad)
print("array mask:", time.time() - start)
