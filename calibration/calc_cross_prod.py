import numpy as np
import math

oz = np.array([73, 260, -448], dtype=np.float32)
oy = np.array([-20.79, 715.62, -448], dtype=np.float32)
ox = np.cross(oy, oz)

oz1 = oz/np.linalg.norm(oz)
oy1 = oy/np.linalg.norm(oy)
ox1 = ox/np.linalg.norm(ox)

print(ox1)
print(oy1)
print(oz1)

B = np.array([ox1, oy1, oz1])
print(B)

primes = np.array([73, 260, -448], dtype=np.float32)
secs = np.array([0, 0, 523.1], dtype=np.float32)

print(np.dot(primes, np.linalg.inv(B)))
print(np.dot(secs, B))

