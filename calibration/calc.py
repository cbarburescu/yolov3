import numpy as np
import math

primes = np.array([73, 260, -448, 1], dtype=np.float32)
secs = np.array([0, 0, 523.1], dtype=np.float32)
phi_x = 148.94
phi_y = 360-15.68


def rot_mat(phi, axis="x"):
    sin_phi = math.sin(math.radians(phi))
    cos_phi = math.cos(math.radians(phi))
    if axis == "x":
        ret = np.array([
            [1, 0, 0],
            [0, cos_phi, -sin_phi],
            [0, sin_phi, cos_phi],
        ], dtype=np.float32)
    elif axis == "y":
        ret = np.array([
            [cos_phi, 0, sin_phi],
            [0, 1, 0],
            [-sin_phi, 0, cos_phi],
        ], dtype=np.float32)
    elif axis == "z":
        ret = np.array([
            [cos_phi, -sin_phi, 0],
            [sin_phi, cos_phi, 0],
            [0, 0, 1],
        ], dtype=np.float32)

    return ret


# print(rot_mat(phi_x, "x") @ rot_mat(phi_y, "y") @ primes @ np.linalg.inv(rot_mat(phi_y, "y")) @ np.linalg.inv(rot_mat(phi_x, "x")))
# print(np.linalg.inv(rot_mat(phi_y, "y")) @ np.linalg.inv(rot_mat(phi_x, "x")) @ primes @ rot_mat(phi_x, "x") @ rot_mat(phi_y, "y"))


def rot_mat_homo(phi, axis="x"):
    sin_phi = math.sin(math.radians(phi))
    cos_phi = math.cos(math.radians(phi))
    if axis == "x":
        ret = np.array([
            [0, 0, 1, 0],
            [cos_phi, sin_phi, 0, 0],
            [-sin_phi, cos_phi, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    elif axis == "y":
        ret = np.array([
            [-sin_phi, cos_phi, 0, 0],
            [0, 0, 1, 0],
            [cos_phi, sin_phi, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    elif axis == "z":
        ret = np.array([
            [cos_phi, sin_phi, 0, 0],
            [-sin_phi, cos_phi, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    return ret


print(rot_mat_homo(phi_x, "x") @ rot_mat_homo(phi_y, "y") @ primes @ np.linalg.inv(rot_mat_homo(phi_y, "y")) @ np.linalg.inv(rot_mat_homo(phi_x, "x")))
print(np.linalg.inv(rot_mat_homo(phi_y, "y")) @ np.linalg.inv(rot_mat_homo(phi_x, "x")) @ primes @ rot_mat_homo(phi_x, "x") @ rot_mat_homo(phi_y, "y"))