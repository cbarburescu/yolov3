import cv2
import os
import numpy as np
import pdb

from scipy.spatial.distance import pdist, squareform


class BasisTransforms:
    def __init__(self):
        self.oz = np.array([73, 260, -448], dtype=np.float32)
        self.oy = np.array([-20.79, 715.62, -448], dtype=np.float32)
        self.ox = np.cross(self.oy, self.oz)

        self.oz = self.oz/np.linalg.norm(self.oz)
        self.oy = self.oy/np.linalg.norm(self.oy)
        self.ox = self.ox/np.linalg.norm(self.ox)

        self.B = np.array([self.ox, self.oy, self.oz])

    def simples_to_primes(self, simples):
        # simples - 1x3 vector or Nx3 array of vectors in simple basis coords
        ret = np.copy(simples)
        ret[..., 2] -= 448
        return ret

    def primes_to_simples(self, primes):
        # primes - 1x3 vector or Nx3 array of vectors in prime basis coords
        ret = np.copy(primes)
        ret[..., 2] += 448
        return ret

    def primes_to_seconds(self, primes):
        # primes - 1x3 vector or Nx3 array of vectors in prime basis coords
        return np.dot(primes, np.linalg.inv(self.B))

    def seconds_to_primes(self, seconds):
        # seconds - 1x3 vector or Nx3 array of vectors in second basis coords
        return np.dot(seconds, self.B)

    def simples_to_seconds(self, simples):
        # simples - 1x3 vector or Nx3 array of vectors in simple basis coords
        return self.primes_to_seconds(self.simples_to_primes(simples))

    def seconds_to_simples(self, seconds):
        # seconds - 1x3 vector or Nx3 array of vectors in second basis coords
        return self.primes_to_simples(self.seconds_to_primes(seconds))


# class ScaleTransforms:
#     def __init__(self, cam_shape=(1201, 1600), perspective_shape=(1201, 1600), roi_shape):
#         # h, w for all shapes





class SocialDistancingSystem:
    def __init__(self, dataset, scale_bird, camera_calibration_dir=None, bg_color=(41, 41, 41), mock=None):
        # Init social distancing parameters
        self.pedestrians_detected = 0
        self.pedestrians_per_sec = 0
        self.two_meter_violations = 0
        self.abs_two_meter_violations = 0
        self.total_two_meter_violations = 0
        self.onefive_meter_violations = 0
        self.abs_onefive_meter_violations = 0
        self.total_onefive_meter_violations = 0
        self.pairs = 0
        self.sh_index = 1
        self.sc_index = 1
        self.scale_w, self.scale_h = scale_bird
        self.camera_calibration_dir = camera_calibration_dir
        self.auto_perspective = camera_calibration_dir is not None
        self.dataset = dataset
        self.mock = mock

        self.person_color = (194, 33, 12)

        # Init bird's eye view
        next(iter(self.dataset))

        if not self.auto_perspective:
            # Set camera perspective
            self.set_camera_perspective()

        else:
            # self.scale_cx = self.dataset.w / 1600 
            # self.scale_cy = self.dataset.h / 1201

            self.scale_cx = 1600 / self.dataset.w 
            self.scale_cy = 1201 / self.dataset.h

            # self.scale_cx = 1 
            # self.scale_cy = 1

            self.basistransforms = BasisTransforms()
            self.load_3d_reconstruction_params()
            self.set_roi()

        # Set distance threshold for realworld coordinates
        self.set_distance_threshold()

    def set_camera_perspective(self):
        src = np.float32(np.array(self.dataset.mouse_pts[:4]))
        dst = np.float32([
            [0, 0],
            [self.dataset.w, 0],
            [self.dataset.w, self.dataset.h],
            [0, self.dataset.h]])

        self.camera_perspective = cv2.getPerspectiveTransform(src, dst)

    def load_3d_reconstruction_params(self):
        self.newcam_mtx = np.load(os.path.join(
            self.camera_calibration_dir, 'newcam_mtx.npy'))
        self.inverse_newcam_mtx = np.load(os.path.join(
            self.camera_calibration_dir, 'inverse_newcam_mtx.npy'))
        self.tvec1 = np.load(os.path.join(
            self.camera_calibration_dir, 'tvec1.npy'))
        self.R_mtx = np.load(os.path.join(
            self.camera_calibration_dir, 'R_mtx.npy'))
        self.inverse_R_mtx = np.linalg.inv(self.R_mtx)
        self.scalingfactor = np.load(os.path.join(
            self.camera_calibration_dir, 's_arr.npy'))[0]

    def set_roi(self):
        w, h = self.dataset.w, self.dataset.h
        edge_points = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)

        self.roi = []
        
        for edge_point in edge_points:
            self.roi.append(self.image_to_realworld(edge_point))

        max_x = max([p[0] for p in self.roi])
        # max_x *= 1.2

        max_y = max([p[1] for p in self.roi])
        # max_y *= 1.2

        self.scale_roi_w = w / max_x
        self.scale_roi_h = h / max_y

        # self.scale_roi_w = 1
        # self.scale_roi_h = 1

        # pdb.set_trace()

        self.roi = np.array(self.roi)
        self.roi = self.roi.astype(np.int32)

        # self.roi[:,[0, 1]] = self.roi[:,[1, 0]]

        self.scaled_roi = np.zeros_like(self.roi, dtype=np.float64)
        # self.scaled_roi[:, 0] = self.roi[:, 0] * self.scale_roi_w / 1.2
        # self.scaled_roi[:, 1] = self.roi[:, 1] * self.scale_roi_h / 1.2
        self.scaled_roi = self.scaled_roi.astype(np.int32)

    def set_distance_threshold(self):
        if self.auto_perspective:
            self.dist_thres = 200
        else:
            dist_pts = np.array([self.dataset.mouse_pts[4:]], dtype=np.float32)
            # warped_dist = cv2.perspectiveTransform(
            #     dist_pts, self.camera_perspective)[0]
            warped_dist = self.image_to_realworld(dist_pts)

            self.dist_thres = np.sqrt(
                (warped_dist[0][0] - warped_dist[1][0]) ** 2
                + (warped_dist[0][1] - warped_dist[1][1]) ** 2
            )

    def image_to_realworld(self, src):
        if self.auto_perspective:
            src[..., 0] *= self.scale_cx
            src[... ,1] *= self.scale_cy
            points = np.expand_dims(np.append(src, 1), axis=0)  # [[u, v , 1]]
            uv_1 = np.array(points, dtype=np.float32)
            uv_1 = uv_1.T
            suv_1 = self.scalingfactor*uv_1
            xyz_c = self.inverse_newcam_mtx.dot(suv_1)
            xyz_c = xyz_c-self.tvec1
            xyz = self.inverse_R_mtx.dot(xyz_c)

            ret = xyz.T
            # ret = self.basistransforms.seconds_to_simples(ret)
            ret = self.basistransforms.simples_to_seconds(ret)
        else:
            ret = cv2.perspectiveTransform(src, self.camera_perspective)

        # first and single point, x and y coords
        return np.squeeze(ret[0][:][:2])

    def realword_to_image(self, src):
        if self.auto_perspective:
            point = np.expand_dims(np.append(src, -448),
                                   axis=0)  # [[u, v , 1]]
            xyz = np.array(points, dtype=np.float32)
            xyz = xyz.T
            r_xyz = self.R_mtx @ xyz
            r_xyz_t = r_xyz + self.tvec1
            uv_1 = 1 / self.scalingfactor * self.newcam_matrix @ r_xyz_t

            ret = uv_1.T

            return np.squeeze(ret[0][:][:2])
        else:
            raise ValueError(
                "realword_to_image only works in auto_perspective mode")

    def plot_points_on_bird_eye_view(self, frame, pedestrian_boxes):
        frame_h = frame.shape[0]
        frame_w = frame.shape[1]

        node_radius = 10
        thickness_node = 20
        solid_back_color = (10, 10, 10)

        self.bird_im = np.zeros(
            (int(frame_h * self.scale_h), int(frame_w * self.scale_w), 3), np.uint8
        )

        # pdb.set_trace()
        self.bird_im[:] = solid_back_color
        if self.auto_perspective:
            cv2.polylines(self.bird_im, [self.scaled_roi], True, (0, 255, 255), thickness=3) 
        self.warped_pts = []

        detections = []
        self.mock_detec_points = []
        if not self.dataset.video_flag and self.mock:
            all_mock_detec_points = {
                "perspective_calibration_balcony.jpeg": [(819, 848), (300, 921)],
            }
            file = os.path.basename(self.dataset.files[self.dataset.count-1])
            if file in all_mock_detec_points.keys():
                self.mock_detec_points = np.array(all_mock_detec_points[file], dtype=np.float32)
            else:
                self.mock = False

            if not self.mock:
                cv2.namedWindow("Draw")
                cv2.setMouseCallback("Draw", self.populate_mock_detec_points)
                self.calibration_image = frame
                while True:
                    cv2.imshow("Draw", self.calibration_image)
                    cv2.waitKey(1)
                    if len(self.mock_detec_points) == 3:
                        cv2.destroyWindow("Draw")
                        break
                self.mock_detec_points = np.array(self.mock_detec_points[:-1], dtype=np.float32)

            for i in range(len(self.mock_detec_points)):
                mid_point_x = self.mock_detec_points[i][0]
                mid_point_y = self.mock_detec_points[i][1]
                pts = np.array([[[mid_point_x, mid_point_y]]], dtype="float32")
                detections.append(pts)
        else:
            for i in range(len(pedestrian_boxes)):
                mid_point_x = int(
                    (pedestrian_boxes[i][0] +
                    pedestrian_boxes[i][2]) / 2
                )

                mid_point_y = int(pedestrian_boxes[i][3])

                pts = np.array([[[mid_point_x, mid_point_y]]], dtype="float32")
                detections.append(pts)
        
            
        for i in range(len(detections)):
            cv2.circle(
                frame,
                tuple(detections[i][0, 0]),
                node_radius,
                self.person_color,
                thickness_node,
            )

            warped_pt = self.image_to_realworld(detections[i])
            warped_pt_scaled = [int(warped_pt[0] * self.scale_w),
                                int(warped_pt[1] * self.scale_h)]

            if self.auto_perspective:
                warped_pt_scaled[0] *= self.scale_roi_w
                warped_pt_scaled[1] *= self.scale_roi_h

            warped_pt_scaled = tuple([int(c) for c in warped_pt_scaled])
            self.warped_pts.append(warped_pt_scaled)
            cv2.circle(
                self.bird_im,
                warped_pt_scaled,
                node_radius,
                self.person_color,
                thickness_node,
            )

    def plot_lines(self, points, dist, dist_thres, color, lineThickness=4, initial_y_offset=75):
        dd = np.where(dist < dist_thres)
        point_pairs = []
        for i in range(int(np.ceil(len(dd[0]) / 2))):
            if dd[0][i] != dd[1][i]:
                point1 = dd[0][i]
                point2 = dd[1][i]
                d = dist[point1][point2]
                
                point_pairs.append([point1, point2])
                cv2.line(
                    self.bird_im,
                    (points[point1][0], points[point1][1]),
                    (points[point2][0], points[point2][1]),
                    color,
                    lineThickness,
                )

                last_y = self.put_text(self.bird_im, f'd_thres={str(self.dist_thres)}', text_color=color, text_offset_y=75)
                self.put_text(self.bird_im, f'       d={str(d)}', text_color=color, text_offset_y=last_y)

    def plot_lines_between_nodes(self):
        # pdb.set_trace()
        points = np.array(self.warped_pts, dtype=np.float64)
        if self.auto_perspective:
            points[:, 0] /= self.scale_roi_w 
            points[:, 1] /= self.scale_roi_h
        points = points.astype(np.int32)
        dist_condensed = pdist(points)
        dist = squareform(dist_condensed)

        self.two_meter_violations = len(np.where(dist_condensed <
                                                 self.dist_thres)[0])
        self.onefive_meter_violations = len(np.where(dist_condensed <
                                                 (self.dist_thres // 4 * 3))[0])
        self.pairs += len(dist_condensed)

        points = points.astype(np.float64)
        if self.auto_perspective:
            points[:, 0] *= self.scale_roi_w 
            points[:, 1] *= self.scale_roi_h
        points = points.astype(np.int32)

        lineThickness = 4
        # All lines - green
        self.plot_lines(points, dist, 9999999, (10, 201, 26), lineThickness=lineThickness)
        # 2m threshold lines - orange
        self.plot_lines(points, dist, self.dist_thres, (52, 152, 235), lineThickness=lineThickness)
        # 1.5m threshold lines - red
        self.plot_lines(points, dist, self.dist_thres / 4 * 3, (17, 17, 212), lineThickness=lineThickness)

    def calculate_stay_at_home_index(self):
        normally_people = 10
        self.pedestrians_per_sec = np.round(
            self.pedestrians_detected / self.dataset.frame, 1)
        self.sh_index = 1 - self.pedestrians_per_sec / normally_people

    def update_stats(self, num_pedestrians):
        self.pedestrians_detected += num_pedestrians
        self.total_two_meter_violations += self.two_meter_violations / self.dataset.fps
        self.abs_two_meter_violations += self.two_meter_violations
        self.total_onefive_meter_violations += self.onefive_meter_violations / self.dataset.fps
        self.abs_onefive_meter_violations += self.onefive_meter_violations
        if self.pairs != 0:
            self.sc_index = 1 - self.abs_two_meter_violations / self.pairs

        self.calculate_stay_at_home_index()

    def put_text(self, frame, text, text_color=(255, 255, 255) , text_offset_x=-1, text_offset_y=25):
        font_scale = 0.8
        font = cv2.FONT_HERSHEY_SIMPLEX
        rectangle_bgr = (35, 35, 35)
        (text_width, text_height) = cv2.getTextSize(
            text, font, fontScale=font_scale, thickness=1
        )[0]
        # set the text start position
        if text_offset_x == -1:
            text_offset_x = frame.shape[1] - 400
        # make the coords of the box with a small padding of two pixels
        box_coords = (
            (text_offset_x, text_offset_y + 5),
            (text_offset_x + text_width + 2, text_offset_y - text_height - 2),
        )
        if text_color == (255, 255, 255):
            cv2.rectangle(
                frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED
            )
        cv2.putText(
            frame,
            text,
            (text_offset_x, text_offset_y),
            font,
            fontScale=font_scale,
            color=text_color,
            thickness=1,
        )

        return 2 * text_height + text_offset_y

    def draw_info(self, frame):
        last_h = 75

        text = f"# 1.5m violations: {int(self.total_onefive_meter_violations)}"
        last_h = self.put_text(frame, text, text_offset_y=last_h)

        text = f"# 2m violations: {int(self.total_two_meter_violations)}"
        last_h = self.put_text(frame, text, text_offset_y=last_h)

        text = f"Stay-at-home Index: {str(np.round(100 * self.sh_index, 1))}%"
        last_h = self.put_text(frame, text, text_offset_y=last_h)

        text = f"Social-distancing Index: {str(np.round(100 * self.sc_index, 1))}%"
        self.put_text(frame, text, text_offset_y=last_h)

    def merge_ims(self, frame):
        if self.auto_perspective:
            cv2.circle(
                frame,
                (int(self.newcam_mtx[0,2]), int(self.newcam_mtx[1,2])),
                10,
                (0, 255, 0),
                20,
            )

        return np.concatenate((frame, self.bird_im), axis=0)

    def populate_mock_detec_points(self, event, x, y, flags, param):
        # Used to mark 4 points on the frame zero of the video that will be warped
        # Used to mark 2 points on the frame zero of the video that are 2 meters away
        if event == cv2.EVENT_LBUTTONDOWN:
            # mouseX, mouseY = x, y
            cv2.circle(self.calibration_image, (x, y), 5, (0, 255, 0), -1)
            self.mock_detec_points.append((x, y))
            print(f"Point detected; mouse points: {self.mock_detec_points}")