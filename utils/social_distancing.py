import cv2
import numpy as np
import pdb

from scipy.spatial.distance import pdist, squareform


class SocialDistancingSystem:
    def __init__(self, dataset, scale_bird, camera_calibration_dir=None, bg_color=(41, 41, 41)):
        # Init social distancing parameters
        self.pedestrians_detected = 0
        self.pedestrians_per_sec = 0
        self.two_meter_violations = 0
        self.abs_two_meter_violations = 0
        self.total_two_meter_violations = 0
        self.pairs = 0
        self.sh_index = 1
        self.sc_index = 1
        self.scale_w, self.scale_h = scale_bird
        self.auto_perspective = camera_calibration_dir is not None
        self.dataset = dataset

        if not self.auto_perspective:
            # Init bird's eye view
            next(iter(self.dataset))

            # Set camera perspective
            self.set_camera_perspective()

            # Compute distance threshold
            self.set_distance_threshold()
        else:
            self.load_3d_reconstruction_params()

    def set_camera_perspective(self):
        src = np.float32(np.array(self.dataset.mouse_pts[:4]))
        dst = np.float32([
            [0, 0],
            [self.dataset.w, 0],
            [self.dataset.w, self.dataset.h],
            [0, self.dataset.h]])

        self.camera_perspective = cv2.getPerspectiveTransform(src, dst)

    def load_3d_reconstruction_params(self):
        self.inverse_newcam_mtx = np.load(os.path.join(
            self.camera_calibration_dir, 'inverse_newcam_mtx.npy'))
        self.tvec1 = np.load(os.path.join(
            self.camera_calibration_dir, 'tvec1.npy'))
        R_mtx = np.load(os.path.join(self.camera_calibration_dir, 'R_mtx.npy'))
        self.inverse_R_mtx = self.inverse_R_mtx = np.linalg.inv(R_mtx)
        self.scalingfactor = np.load(os.path.join(self.camera_calibration_dir, 's_arr.npy'))[0]

    def change_perspective(self, src):
        if self.auto_perspective:
            points = np.expand_dims(np.append(src, 1))  # [[u, v , 1]]
            uv_1 = np.array(points, dtype=np.float32)
            uv_1 = uv_1.T
            suv_1 = self.scalingfactor*uv_1
            xyz_c = self.inverse_newcam_mtx.dot(suv_1)
            xyz_c = xyz_c-self.tvec1
            xyz = self.inverse_R_mtx.dot(xyz_c)

            ret = xyz
        else:
            ret = cv2.perspectiveTransform(src, self.camera_perspective)

        return np.squeeze(ret[0][:][:2])  # first and single point, x and y coords

    def set_distance_threshold(self):
        dist_pts = np.array([self.dataset.mouse_pts[4:]], dtype=np.float32)
        # warped_dist = cv2.perspectiveTransform(
        #     dist_pts, self.camera_perspective)[0]
        warped_dist = self.change_perspective(dist_pts)

        self.dist_thres = np.sqrt(
            (warped_dist[0][0] - warped_dist[1][0]) ** 2
            + (warped_dist[0][1] - warped_dist[1][1]) ** 2
        )

    def plot_points_on_bird_eye_view(self, frame, pedestrian_boxes):
        frame_h = frame.shape[0]
        frame_w = frame.shape[1]

        node_radius = 10
        color_node = (192, 133, 156)
        thickness_node = 20
        solid_back_color = (41, 41, 41)

        self.bird_im = np.zeros(
            (int(frame_h * self.scale_h), int(frame_w * self.scale_w), 3), np.uint8
        )
        self.bird_im[:] = solid_back_color
        self.warped_pts = []
        for i in range(len(pedestrian_boxes)):
            mid_point_x = int(
                (pedestrian_boxes[i][0] +
                 pedestrian_boxes[i][2]) / 2
            )
            # mid_point_y = int(
            #     (pedestrian_boxes[i][1] +
            #      pedestrian_boxes[i][3]) / 2
            # )
            mid_point_y = int(pedestrian_boxes[i][3])

            pts = np.array([[[mid_point_x, mid_point_y]]], dtype="float32")
            # warped_pt = cv2.perspectiveTransform(
            #     pts, self.camera_perspective)[0]
            warped_pt = self.change_perspective(pts)
            warped_pt_scaled = (int(warped_pt[0] * self.scale_w),
                                int(warped_pt[1] * self.scale_h))

            self.warped_pts.append(warped_pt_scaled)
            cv2.circle(
                self.bird_im,
                warped_pt_scaled,
                node_radius,
                color_node,
                thickness_node,
            )

    def plot_lines_between_nodes(self):
        p = np.array(self.warped_pts)
        dist_condensed = pdist(p)
        dist = squareform(dist_condensed)

        self.two_meter_violations = len(np.where(dist_condensed <
                                                 self.dist_thres)[0])
        self.pairs += len(dist_condensed)

        # Close enough: 3 meter mark
        dd = np.where(dist < self.dist_thres / 2 * 3)
        close_p = []
        color_10 = (80, 172, 110)
        lineThickness = 4
        for i in range(int(np.ceil(len(dd[0]) / 2))):
            if dd[0][i] != dd[1][i]:
                point1 = dd[0][i]
                point2 = dd[1][i]

                close_p.append([point1, point2])

                cv2.line(
                    self.bird_im,
                    (p[point1][0], p[point1][1]),
                    (p[point2][0], p[point2][1]),
                    color_10,
                    lineThickness,
                )

        # Really close: 2 meter mark
        dd = np.where(dist < self.dist_thres)
        danger_p = []
        color_6 = (52, 92, 227)
        for i in range(int(np.ceil(len(dd[0]) / 2))):
            if dd[0][i] != dd[1][i]:
                point1 = dd[0][i]
                point2 = dd[1][i]

                danger_p.append([point1, point2])
                cv2.line(
                    self.bird_im,
                    (p[point1][0], p[point1][1]),
                    (p[point2][0], p[point2][1]),
                    color_6,
                    lineThickness,
                )

    def calculate_stay_at_home_index(self):
        normally_people = 10
        self.pedestrians_per_sec = np.round(
            self.pedestrians_detected / self.dataset.frame, 1)
        self.sh_index = 1 - self.pedestrians_per_sec / normally_people

    def update_stats(self, num_pedestrians):
        self.pedestrians_detected += num_pedestrians
        self.total_two_meter_violations += self.two_meter_violations / self.dataset.fps
        self.abs_two_meter_violations += self.two_meter_violations
        if self.pairs != 0:
            self.sc_index = 1 - self.abs_two_meter_violations / self.pairs

        self.calculate_stay_at_home_index()

    def put_text(self, frame, text, text_offset_y=25):
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        rectangle_bgr = (35, 35, 35)
        (text_width, text_height) = cv2.getTextSize(
            text, font, fontScale=font_scale, thickness=1
        )[0]
        # set the text start position
        text_offset_x = frame.shape[1] - 400
        # make the coords of the box with a small padding of two pixels
        box_coords = (
            (text_offset_x, text_offset_y + 5),
            (text_offset_x + text_width + 2, text_offset_y - text_height - 2),
        )
        cv2.rectangle(
            frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED
        )
        cv2.putText(
            frame,
            text,
            (text_offset_x, text_offset_y),
            font,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=1,
        )

        return 2 * text_height + text_offset_y

    def draw_info(self, frame):
        last_h = 50

        text = f"# 2m violations: {int(self.total_two_meter_violations)}"
        last_h = self.put_text(frame, text, text_offset_y=last_h)

        text = f"Stay-at-home Index: {str(np.round(100 * self.sh_index, 1))}%"
        last_h = self.put_text(frame, text, text_offset_y=last_h)

        text = f"Social-distancing Index: {str(np.round(100 * self.sc_index, 1))}%"
        self.put_text(frame, text, text_offset_y=last_h)

    def merge_ims(self, frame):
        # print(self.bird_im.shape, frame.shape)
        scale = self.dataset.h / self.bird_im.shape[0]
        self.bird_im = cv2.resize(
            self.bird_im, dsize=(0, 0), fx=scale, fy=scale)

        return np.concatenate((frame, self.bird_im), axis=1)
