# https://docs.opencv.org/3.3.0/dc/dbb/tutorial_py_calibration.html

import argparse
import glob
import os
import pdb
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.social_distancing import BasisTransforms, ScaleTransform


def populate_mouse_points(event, x, y, flags, param):
        # Used to mark 4 points on the frame zero of the video that will be warped
        # Used to mark 2 points on the frame zero of the video that are 2 meters away
        if event == cv2.EVENT_LBUTTONDOWN:
            # mouseX, mouseY = x, y
            cv2.circle(calibration_image, (x, y), 5, (0, 255, 0), -1)
            mouse_pts.append((x, y))
            print(f"Point detected; mouse points: {mouse_pts}")

if __name__ == "__main__":
    simples = np.array([73, 260, 0], dtype=np.float32)
    primes = np.array([73, 260, -448], dtype=np.float32)
    secs = np.array([0, 0, 523.1], dtype=np.float32)

    basistransform = BasisTransforms()
    mouse_pts = []

    # print(basistransform.simples_to_seconds(simples))
    # print(basistransform.seconds_to_simples(secs))

    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-image", default="images/perspective/perspective_calibration_balcony2.jpeg", help="image to test calibration")
    parser.add_argument("--save-dir", default=os.path.join(script_dir, "camera_params"), help="dir with camera calibration parameters")
    parser.add_argument("--cam-shape", nargs=2, default=[1600, 1201], type=float, help="w, h from camera calibration images")
    parser.add_argument("--draw-points", action="store_true", help="Draw mouse points for calibration")
    parser.add_argument("--dry-run", action="store_true", help="Do not write new perspective params in --save-dir")
    opt = parser.parse_args()

    # test camera calibration against all points, calculating XYZ
    # load perspective calibration images
    calibration_image = cv2.imread(opt.calibration_image)
    h, w, _ = calibration_image.shape

    # load camera calibration
    cam_mtx = np.load(os.path.join(opt.save_dir, 'cam_mtx.npy'))
    dist = np.load(os.path.join(opt.save_dir, 'dist.npy'))
    newcam_mtx = np.load(os.path.join(opt.save_dir, 'newcam_mtx.npy'))
    roi = np.load(os.path.join(opt.save_dir, 'roi.npy'))


    # load center points from New Camera matrix
    c_scaler = ScaleTransform(src=opt.cam_shape, dst=(w, h))
    cx, cy = c_scaler.scale((newcam_mtx[0, 2], newcam_mtx[1, 2]))
    # cx = newcam_mtx[0, 2] * w / opt.cam_shape[0]
    # cy = newcam_mtx[1, 2] * h / opt.cam_shape[1]
    fx = newcam_mtx[0, 0]
    fy = newcam_mtx[1, 1]
    print("cx: "+str(cx)+",cy "+str(cy)+",fx "+str(fx))


    print(cx, cy)
    print(fx, fy)
    cv2.circle(calibration_image, (int(cx), int(cy)), 10, (0,255,0), -1)
    
    if not opt.draw_points:
        while True:
            cv2.imshow("Perspective Calibration", calibration_image)
            k = cv2.waitKey(0)
            if k == ord("q"):
                cv2.destroyAllWindows()
                break

    # MANUALLY INPUT YOUR MEASURED POINTS HERE
    # ENTER (X,Y,d*)
    # d* is the distance from your point to the camera lens. (d* = Z for the camera center)
    # we will calculate Z in the next steps after extracting the new_cam matrix


    # world center + 9 world points


    X_center = 75  # 184.93
    Y_center = 255 # 115.0
    Z_center = 523.1
    worldPoints = np.array([[X_center, Y_center, Z_center],
                            [40, 241, 448],
                            [71, 241, 448],
                            [102.5, 241, 448],
                            [40, 272, 448],
                            [71, 272, 448],
                            [102.5, 272, 448],
                            [40, 303, 448],
                            [71, 303, 448],
                            [102.5, 303, 448]], dtype=np.float32)
    # worldPoints = np.array([[X_center, Y_center, Z_center],
    #                         [102.5, 303, 523.1],
    #                         [71, 303, 523.1],
    #                         [40, 303, 523.1],
    #                         [102.5, 272, 523.1],
    #                         [71, 272, 523.1],
    #                         [40, 272, 523.1],
    #                         [102.5, 241, 523.1],
    #                         [71, 241, 523.1],
    #                         [40, 241, 523.1]], dtype=np.float32)
    # worldPoints = basistransform.simples_to_seconds(worldPoints)


    # MANUALLY INPUT THE DETECTED IMAGE COORDINATES HERE

    # [u,v] center + 9 Image points
    imagePoints = np.array([[cx, cy]], dtype=np.float32)

    if opt.draw_points:
        cv2.namedWindow("Draw")
        cv2.setMouseCallback("Draw", populate_mouse_points)
        while True:
            cv2.imshow("Draw", calibration_image)
            cv2.waitKey(1)
            if len(mouse_pts) == 10:
                cv2.destroyWindow("Draw")
                break

        mouse_pts = np.array(mouse_pts[:-1], dtype=np.float32)
        imagePoints = np.concatenate((imagePoints, mouse_pts), axis=0)

    else:
        extraPoints = np.array([(2088, 1889),
                                (2284, 1870),
                                (2481, 1855),
                                (2068, 1724),
                                (2260, 1712),
                                (2449, 1688),
                                (2051, 1572),
                                (2235, 1562),
                                (2420, 1545)], dtype=np.float32)

        imagePoints = np.concatenate((imagePoints, extraPoints), axis=0)

    print(imagePoints)
    total_points_used = imagePoints.shape[0]

    # FOR REAL WORLD POINTS, CALCULATE Z from d*

    for i in range(1, total_points_used):
        # start from 1, given for center Z=d*
        # to center of camera
        wX = worldPoints[i, 0]-X_center
        wY = worldPoints[i, 1]-Y_center
        wd = np.sqrt(np.sum(np.power(worldPoints[i], 2)))

        d1 = np.sqrt(np.square(wX)+np.square(wY))
        wZ = np.sqrt(np.square(wd)-np.square(d1))
        worldPoints[i, 2] = wZ

    print(worldPoints)


    # print(ret)
    print("Camera Matrix")
    print(cam_mtx, end="\n"+"-"*70+"\n")
    print("Distortion Coeff")
    print(dist, end="\n"+"-"*70+"\n")

    print("Region of Interest")
    print(roi, end="\n"+"-"*70+"\n")
    print("New Camera Matrix")
    print(newcam_mtx, end="\n"+"-"*70+"\n")
    inverse_newcam_mtx = np.linalg.inv(newcam_mtx)
    print("Inverse New Camera Matrix")
    print(inverse_newcam_mtx, end="\n"+"-"*70+"\n")

    print(">==> Calibration Loaded")


    print("solvePNP")
    ret, rvec1, tvec1 = cv2.solvePnP(worldPoints, imagePoints, newcam_mtx, dist)

    print("pnp rvec1 - Rotation")
    print(rvec1, end="\n"+"-"*70+"\n")

    print("pnp tvec1 - Translation")
    print(tvec1, end="\n"+"-"*70+"\n")

    print("R - rodrigues vecs")
    R_mtx, jac = cv2.Rodrigues(rvec1)
    print(R_mtx, end="\n"+"-"*70+"\n")

    print("R|t - Extrinsic Matrix")
    Rt = np.column_stack((R_mtx, tvec1))
    print(Rt, end="\n"+"-"*70+"\n")

    print("newCamMtx*R|t - Projection Matrix")
    P_mtx = newcam_mtx.dot(Rt)
    print(P_mtx, end="\n"+"-"*70+"\n")

    if not opt.dry_run:
        save_names = {
            'inverse_newcam_mtx.npy': inverse_newcam_mtx,
            'rvec1.npy': rvec1,
            'tvec1.npy': tvec1,
            'R_mtx.npy': R_mtx,
            'Rt.npy': Rt,
            'P_mtx.npy': P_mtx,
        }
        for k, v in save_names.items():
            np.save(os.path.join(opt.save_dir, k), v)
        

    # [XYZ1]


    # LETS CHECK THE ACCURACY HERE


    s_arr = np.array([0], dtype=np.float32)
    s_describe = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    for i in range(0, total_points_used):
        print("=======POINT # " + str(i) + " =========================")

        print("Forward: From World Points, Find Image Pixel")
        XYZ1 = np.array([[worldPoints[i, 0], worldPoints[i, 1],
                        worldPoints[i, 2], 1]], dtype=np.float32)
        XYZ1 = XYZ1.T
        print("{{-- XYZ1")
        print(XYZ1, end="\n"+"-"*70+"\n")
        suv1 = P_mtx.dot(XYZ1)
        print("//-- suv1")
        print(suv1, end="\n"+"-"*70+"\n")
        s = suv1[2, 0]
        # pdb.set_trace()
        uv1 = suv1/s
        print(">==> uv1 - Image Points")
        print(uv1, end="\n"+"-"*70+"\n")
        print(">==> s - Scaling Factor")
        print(s, end="\n"+"-"*70+"\n")
        s_arr = np.array([s/total_points_used+s_arr[0]], dtype=np.float32)
        s_describe[i] = s
        if not opt.dry_run:
            np.save(os.path.join(opt.save_dir, 's_arr.npy'), s_arr)

        print("Solve: From Image Pixels, find World Points")

        uv_1 = np.array(
            [[imagePoints[i, 0], imagePoints[i, 1], 1]], dtype=np.float32)
        uv_1 = uv_1.T
        print(">==> uv1")
        print(uv_1, end="\n"+"-"*70+"\n")
        suv_1 = s*uv_1
        print("//-- suv1")
        print(suv_1, end="\n"+"-"*70+"\n")

        print("get camera coordinates, multiply by inverse Camera Matrix, subtract tvec1")
        xyz_c = inverse_newcam_mtx.dot(suv_1)
        xyz_c = xyz_c-tvec1
        print("      xyz_c")
        inverse_R_mtx = np.linalg.inv(R_mtx)
        XYZ = inverse_R_mtx.dot(xyz_c)
        print("{{-- XYZ")
        print(XYZ, end="\n"+"-"*70+"\n")


    s_mean, s_std = np.mean(s_describe), np.std(s_describe)

    print(">>>>>>>>>>>>>>>>>>>>> S RESULTS")
    print("Mean: " + str(s_mean))
    #print("Average: " + str(s_arr[0]))
    print("Std: " + str(s_std))

    print(">>>>>> S Error by Point")

    for i in range(0, total_points_used):
        print("Point "+str(i))
        print("S: " + str(s_describe[i])+" Mean: " +
            str(s_mean) + " Error: " + str(s_describe[i]-s_mean))
