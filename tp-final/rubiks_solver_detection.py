import cv2
from rubik_solver import utils
from color_utils import color_detector
from config import config
from ui_utils import (
    draw_preview_stickers,
    draw_current_stickers,
    draw_contours,
    draw_current_color_to_calibrate,
    draw_calibrated_colors,
    draw_scanned_successfully,
    draw_2d_cube_state
)
from constants import (
    CUBE_PALETTE,
    CALIBRATE_MODE_KEY,
    ERROR_INCORRECTLY_SCANNED,
    ERROR_ALREADY_SOLVED,
    ERROR_MISSING_SIDES,
)


class Webcam:

    def __init__(self):
        self.cam = cv2.VideoCapture(0)

        self.colors_to_calibrate = ['red', 'green', 'orange', 'blue', 'yellow', 'white']
        self.average_sticker_colors = {}
        # all 6 sides result
        self.result_state = {}

        # initial snapshot(current saved detection) & preview(current detection) states are all white
        self.snapshot_state = [(255, 255, 255), (255, 255, 255), (255, 255, 255),
                               (255, 255, 255), (255, 255, 255), (255, 255, 255),
                               (255, 255, 255), (255, 255, 255), (255, 255, 255)]
        self.preview_state = [(255, 255, 255), (255, 255, 255), (255, 255, 255),
                              (255, 255, 255), (255, 255, 255), (255, 255, 255),
                              (255, 255, 255), (255, 255, 255), (255, 255, 255)]

        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.calibrate_mode = False
        self.calibrated_colors = {}
        self.current_calibrate_index = 0
        self.done_calibrating = False

    # filters cube inside contours, those who have a square-ish shape
    @staticmethod
    def find_contours(dilated_frame):
        contours, hierarchy = cv2.findContours(dilated_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # this array will contain all of the square-ish shapes
        final_contours = []

        # 1 - Find all shapes that are square-ish
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.1 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:  # aprox poly has 4 sides
                area = cv2.contourArea(contour)
                (x, y, w, h) = cv2.boundingRect(approx)

                # width vs height to check if is square
                ratio = w / float(h)

                # check if contour is close to a square
                if 0.8 <= ratio <= 1.2 and 30 <= w <= 400 and area / (w * h) > 0.8:
                    final_contours.append((x, y, w, h))

        # no full side found
        if len(final_contours) < 9:
            return []

        # 2 - Find the contour that has 9 neighbors (including itself)
        found = False
        contour_neighbors = {}
        for index, contour in enumerate(final_contours):
            (x, y, w, h) = contour
            contour_neighbors[index] = []
            # shape x & y centers
            center_x = x + w / 2
            center_y = y + h / 2

            delta = 1.1  # este delta esta alpedo? multiplica por uno

            # check the 8 neighbours if are present on the square-ish shapes list.
            # if has 8 neighbours, this is the center sticker

            x_left = (center_x - w * delta)
            y_top = (center_y - h * delta)
            x_right = (center_x + w * delta)
            y_bottom = (center_y + h * delta)
            neighbor_positions = [
                # top left
                [x_left, y_top],

                # top center
                [center_x, y_top],

                # top right
                [x_right, y_top],

                # mid left
                [x_left, center_y],

                # center
                [center_x, center_y],

                # mid right
                [x_right, center_y],

                # bottom left
                [x_left, y_bottom],

                # bottom center
                [center_x, y_bottom],

                # bottom right
                [x_right, y_bottom],
            ]

            for neighbor in final_contours:  # iterate detected square-ish shapes and see if the neighbours are included
                (detected_x, detected_y, detected_w, detected_h) = neighbor
                for (neighbour_x, neighbour_y) in neighbor_positions:  # these are the centers of each neighbour
                    # center is between top left and bottom right
                    if detected_x < neighbour_x < detected_x + detected_w and detected_y < neighbour_y < detected_y + detected_h:
                        contour_neighbors[index].append(neighbor)

        # 3 - We have all neighbours, need to check which contour has 9 neighbours
        for (contour, neighbors) in contour_neighbors.items():
            if len(neighbors) == 9:
                found = True
                final_contours = neighbors
                break

        if not found:
            return []

        # 4 - Sort contours top left to bottom right

        # Sort y
        y_sorted = sorted(final_contours, key=lambda item: item[1])

        # Sort each y on x
        top_row = sorted(y_sorted[0:3], key=lambda item: item[0])
        middle_row = sorted(y_sorted[3:6], key=lambda item: item[0])
        bottom_row = sorted(y_sorted[6:9], key=lambda item: item[0])

        sorted_contours = top_row + middle_row + bottom_row
        return sorted_contours

    # check if all colors are scanned (all colors should sum 9 occurrences)
    def scanned_successfully(self):
        color_count = {}
        for side, preview in self.result_state.items():
            for bgr in preview:
                key = str(bgr)
                if not key in color_count:
                    color_count[key] = 1
                else:
                    color_count[key] = color_count[key] + 1
        invalid_colors = [k for k, v in color_count.items() if v != 9]
        return len(invalid_colors) == 0

    # calculate average colors
    def update_preview_state(self, frame, contours):
        # max_average_rounds = 8
        for index, (x, y, w, h) in enumerate(contours):
            # this part does not do much, when calculated over 8 times each sticker, it calculates an average of those detections
            # if index in self.average_sticker_colors and len(self.average_sticker_colors[index]) == max_average_rounds:
            #     sorted_items = {}
            #     for bgr in self.average_sticker_colors[index]:
            #         key = str(bgr)
            #         if key in sorted_items:
            #             sorted_items[key] += 1
            #         else:
            #             sorted_items[key] = 1
            #     most_common_color = max(sorted_items, key=lambda i: sorted_items[i])
            #     self.average_sticker_colors[index] = []
            #     self.preview_state[index] = eval(most_common_color)
            #     break

            # define a region of interest from contour center and get dominant color
            roi = frame[y + 12:y + h - 12, x + 12:x + w - 12]
            avg_bgr = color_detector.get_dominant_color(roi)
            closest_color = color_detector.get_closest_color(avg_bgr)['color_bgr']

            # update the state of the current preview
            self.preview_state[index] = closest_color
            if index in self.average_sticker_colors:
                self.average_sticker_colors[index].append(closest_color)
            else:
                self.average_sticker_colors[index] = [closest_color]

    # set saved state as the current preview state
    def update_snapshot_state(self, frame):
        self.snapshot_state = list(self.preview_state)
        center_color_name = color_detector.get_closest_color(self.snapshot_state[4])['color_name']
        self.result_state[center_color_name] = self.snapshot_state
        draw_current_stickers(self, frame)

    # reset calibration
    def reset_calibrate_mode(self):
        self.calibrated_colors = {}
        self.current_calibrate_index = 0
        self.done_calibrating = False

    # convert result to solver library input
    def get_result_notation(self):
        notation = dict(self.result_state)
        items = {}
        color_seq = ''
        for side, preview in notation.items():
            for sticker_index, bgr in enumerate(preview):
                if side in items:
                    items[side] = items[side] + color_detector.convert_bgr_to_color_initial(bgr)
                else:
                    items[side] = color_detector.convert_bgr_to_color_initial(bgr)

        for side in ['yellow', 'blue', 'red', 'green', 'orange', 'white']:
            color_seq += items[side]

        return color_seq

    # check if is already solved
    def state_already_solved(self):
        for side in ['white', 'red', 'green', 'yellow', 'orange', 'blue']:
            center = self.result_state[side][4]

            # Compare the center color to neighbours, all should be same as center
            for sticker in self.result_state[side]:
                if center != sticker:
                    return False
        return True

    def run(self):
        while True:
            _, frame = self.cam.read()
            key = cv2.waitKey(10) & 0xff

            if key == 27:  # escape
                break

            if not self.calibrate_mode:
                if key == 32:  # space
                    self.update_snapshot_state(frame)

            # Toggle calibrate mode.
            if key == ord(CALIBRATE_MODE_KEY):
                self.reset_calibrate_mode()
                self.calibrate_mode = not self.calibrate_mode

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred_frame = cv2.blur(gray_frame, (3, 3))
            canny_frame = cv2.Canny(blurred_frame, 30, 60, 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
            dilated_frame = cv2.dilate(canny_frame, kernel)
            cv2.imshow("dilated", dilated_frame)

            contours = self.find_contours(dilated_frame)
            if len(contours) == 9:
                draw_contours(self, frame, contours)
                if not self.calibrate_mode:
                    self.update_preview_state(frame, contours)
                elif key == 32 and self.done_calibrating == False:
                    current_color = self.colors_to_calibrate[self.current_calibrate_index]
                    (x, y, w, h) = contours[4]
                    roi = frame[y + 7:y + h - 7, x + 14:x + w - 14]
                    avg_bgr = color_detector.get_dominant_color(roi)
                    self.calibrated_colors[current_color] = avg_bgr
                    self.current_calibrate_index += 1
                    self.done_calibrating = self.current_calibrate_index == len(self.colors_to_calibrate)
                    if self.done_calibrating:
                        color_detector.set_cube_color_pallete(self.calibrated_colors)
                        config.set_setting(CUBE_PALETTE, color_detector.cube_color_palette)

            if self.calibrate_mode:
                draw_current_color_to_calibrate(self, frame)
                draw_calibrated_colors(self, frame)
            else:
                draw_preview_stickers(self, frame)
                draw_current_stickers(self, frame)
                draw_2d_cube_state(self, frame)
                if len(self.result_state.keys()) == 6:
                    draw_scanned_successfully(self, frame, self.scanned_successfully())

            cv2.imshow("Visiar - Rubik's cube solver", frame)

        self.cam.release()
        cv2.destroyAllWindows()

        if len(self.result_state.keys()) != 6:
            return ERROR_MISSING_SIDES

        if not self.scanned_successfully():
            return ERROR_INCORRECTLY_SCANNED

        if self.state_already_solved():
            return ERROR_ALREADY_SOLVED

        return utils.solve(self.get_result_notation(), 'Kociemba')


webcam = Webcam()
print(webcam.run())
