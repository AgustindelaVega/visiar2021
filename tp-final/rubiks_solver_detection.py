import cv2
from rubik_solver import utils
from color_utils import color_detector
from config import config
from constants import (
    COLOR_PLACEHOLDER,
    CUBE_PALETTE,
    MINI_STICKER_AREA_TILE_SIZE,
    MINI_STICKER_AREA_TILE_GAP,
    MINI_STICKER_AREA_OFFSET,
    STICKER_AREA_TILE_SIZE,
    STICKER_AREA_TILE_GAP,
    STICKER_AREA_OFFSET,
    STICKER_CONTOUR_COLOR,
    CALIBRATE_MODE_KEY,
    TEXT_SIZE,
    E_INCORRECTLY_SCANNED,
    E_ALREADY_SOLVED
)


class Webcam:

    def __init__(self):
        self.cam = cv2.VideoCapture(0)

        self.colors_to_calibrate = ['red', 'green', 'orange', 'blue', 'yellow', 'white']
        self.average_sticker_colors = {}
        self.result_state = {}

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
        self.current_color_to_calibrate_index = 0
        self.done_calibrating = False

    # draw main grid of stickers
    def draw_stickers(self, frame, stickers, offset_x, offset_y):
        index = -1
        for row in range(3):
            for col in range(3):
                index += 1
                x1 = (offset_x + STICKER_AREA_TILE_SIZE * col) + STICKER_AREA_TILE_GAP * col
                y1 = (offset_y + STICKER_AREA_TILE_SIZE * row) + STICKER_AREA_TILE_GAP * row
                x2 = x1 + STICKER_AREA_TILE_SIZE
                y2 = y1 + STICKER_AREA_TILE_SIZE

                # shadow
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 0),
                    -1
                )

                # foreground color
                cv2.rectangle(
                    frame,
                    (x1 + 1, y1 + 1),
                    (x2 - 1, y2 - 1),
                    color_detector.get_prominent_color(stickers[index]),
                    -1
                )

    # draw the preview of the scanned and saved stickers
    def draw_preview_stickers(self, frame):
        self.draw_stickers(frame, self.preview_state, STICKER_AREA_OFFSET, STICKER_AREA_OFFSET)

    # draw the current scanned sticker
    def draw_current_stickers(self, frame):
        y = STICKER_AREA_TILE_SIZE * 3 + STICKER_AREA_TILE_GAP * 2 + STICKER_AREA_OFFSET * 2
        self.draw_stickers(frame, self.snapshot_state, STICKER_AREA_OFFSET, y)

    # filters cube inside contours, those who have a square-ish shape
    def find_contours(self, dilatedFrame):
        contours, hierarchy = cv2.findContours(dilatedFrame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_contours = []

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.1 * perimeter, True)
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                (x, y, w, h) = cv2.boundingRect(approx)

                # Find aspect ratio of boundary rectangle around the countours.
                ratio = w / float(h)

                # Check if contour is close to a square.
                if ratio >= 0.8 and ratio <= 1.2 and w >= 30 and w <= 100 and area / (w * h) > 0.4:
                    final_contours.append((x, y, w, h))

        # Return early if we didn't found 9 or more contours.
        if len(final_contours) < 9:
            return []

        # Step 2/4: Find the contour that has 9 neighbors (including itself)
        found = False
        contour_neighbors = {}
        for index, contour in enumerate(final_contours):
            (x, y, w, h) = contour
            contour_neighbors[index] = []
            center_x = x + w / 2
            center_y = y + h / 2
            radius = 1.5

            # Create 9 positions for the current contour which are the
            # neighbors. We'll use this to check how many neighbors each contour
            # has. The only way all of these can match is if the current contour
            # is the center of the cube. If we found the center, we also know
            # all the neighbors, thus knowing all the contours and thus knowing
            # this shape can be considered a 3x3x3 cube. When we've found those
            # contours, we sort them and return them.
            neighbor_positions = [
                # top left
                [(center_x - w * radius), (center_y - h * radius)],

                # top middle
                [center_x, (center_y - h * radius)],

                # top right
                [(center_x + w * radius), (center_y - h * radius)],

                # middle left
                [(center_x - w * radius), center_y],

                # center
                [center_x, center_y],

                # middle right
                [(center_x + w * radius), center_y],

                # bottom left
                [(center_x - w * radius), (center_y + h * radius)],

                # bottom middle
                [center_x, (center_y + h * radius)],

                # bottom right
                [(center_x + w * radius), (center_y + h * radius)],
            ]

            for neighbor in final_contours:
                (x2, y2, w2, h2) = neighbor
                for (x3, y3) in neighbor_positions:
                    # The neighbor_positions are located in the center of each
                    # contour instead of top-left corner.
                    # logic: (top left < center pos) and (bottom right > center pos)
                    if (x2 < x3 and y2 < y3) and (x2 + w2 > x3 and y2 + h2 > y3):
                        contour_neighbors[index].append(neighbor)

        # Step 3/4: Now that we know how many neighbors all contours have, we'll
        # loop over them and find the contour that has 9 neighbors, which
        # includes itself. This is the center piece of the cube. If we come
        # across it, then the 'neighbors' are actually all the contours we're
        # looking for.
        for (contour, neighbors) in contour_neighbors.items():
            if len(neighbors) == 9:
                found = True
                final_contours = neighbors
                break

        if not found:
            return []

        # Step 4/4: When we reached this part of the code we found a cube-like
        # contour. The code below will sort all the contours on their X and Y
        # values from the top-left to the bottom-right.

        # Sort contours on the y-value first.
        y_sorted = sorted(final_contours, key=lambda item: item[1])

        # Split into 3 rows and sort each row on the x-value.
        top_row = sorted(y_sorted[0:3], key=lambda item: item[0])
        middle_row = sorted(y_sorted[3:6], key=lambda item: item[0])
        bottom_row = sorted(y_sorted[6:9], key=lambda item: item[0])

        sorted_contours = top_row + middle_row + bottom_row
        return sorted_contours

    # check if all colors are scanned
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

    # draw the contours of each color
    def draw_contours(self, frame, contours):
        if self.calibrate_mode:
            # Only show the center piece contour.
            (x, y, w, h) = contours[4]
            cv2.rectangle(frame, (x, y), (x + w, y + h), STICKER_CONTOUR_COLOR, 2)
        else:
            for index, (x, y, w, h) in enumerate(contours):
                cv2.rectangle(frame, (x, y), (x + w, y + h), STICKER_CONTOUR_COLOR, 2)

    # calculate average colors
    def update_preview_state(self, frame, contours):
        max_average_rounds = 8
        for index, (x, y, w, h) in enumerate(contours):
            if index in self.average_sticker_colors and len(self.average_sticker_colors[index]) == max_average_rounds:
                sorted_items = {}
                for bgr in self.average_sticker_colors[index]:
                    key = str(bgr)
                    if key in sorted_items:
                        sorted_items[key] += 1
                    else:
                        sorted_items[key] = 1
                most_common_color = max(sorted_items, key=lambda i: sorted_items[i])
                self.average_sticker_colors[index] = []
                self.preview_state[index] = eval(most_common_color)
                break

            roi = frame[y + 12:y + h - 12, x + 12:x + w - 12]
            cv2.imshow("test", roi)
            avg_bgr = color_detector.get_dominant_color(roi)
            closest_color = color_detector.get_closest_color(avg_bgr)['color_bgr']
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
        self.draw_current_stickers(frame)

    # show text on the given frame
    def render_text(self, frame, text, pos, color=(255, 255, 255)):
        self.get_text_size(text)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA,
                    bottomLeftOrigin=None)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color, thickness=1, lineType=cv2.LINE_AA,
                    bottomLeftOrigin=None)

    def get_text_size(self, text, size=TEXT_SIZE):
        return (size, size), size

    # draw scanned sides count
    def draw_scanned_sides(self, frame):
        text = 'scannedSides: ' + str(len(self.result_state.keys()))
        self.render_text(frame, text, (20, self.height - 20))

    # draw current color to calibrate
    def draw_current_color_to_calibrate(self, frame):
        y_offset = 20
        font_size = int(TEXT_SIZE * 1.25)
        if self.done_calibrating:
            messages = [
                'calibratedSuccessfully',
                'quitCalibrateMode: ' + CALIBRATE_MODE_KEY,
            ]
            for index, text in enumerate(messages):
                (textsize_width, textsize_height), _ = self.get_text_size(text, font_size)
                y = y_offset + (textsize_height + 10) * index
                self.render_text(frame, text, (int(self.width / 2 - textsize_width / 2), y))
        else:
            current_color = self.colors_to_calibrate[self.current_color_to_calibrate_index]
            text = 'currentCalibratingSide: ' + current_color
            (textsize_width, textsize_height), _ = self.get_text_size(text, font_size)
            self.render_text(frame, text, (int(self.width / 2 - textsize_width / 2), y_offset))

    # draw the history of calibrated colors
    def draw_calibrated_colors(self, frame):
        offset_y = 20
        for index, (color_name, color_bgr) in enumerate(self.calibrated_colors.items()):
            x1 = 90
            y1 = int(offset_y + STICKER_AREA_TILE_SIZE * index)
            x2 = x1 + STICKER_AREA_TILE_SIZE
            y2 = y1 + STICKER_AREA_TILE_SIZE

            # shadow
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 0),
                -1
            )

            # foreground
            cv2.rectangle(
                frame,
                (x1 + 1, y1 + 1),
                (x2 - 1, y2 - 1),
                tuple([int(c) for c in color_bgr]),
                -1
            )
            self.render_text(frame, color_name, (20, y1 + 20))

    # reset calibration
    def reset_calibrate_mode(self):
        self.calibrated_colors = {}
        self.current_color_to_calibrate_index = 0
        self.done_calibrating = False

    # draw cube state
    def draw_2d_cube_state(self, frame):
        grid = {
            'white': [1, 2],
            'orange': [3, 1],
            'green': [2, 1],
            'red': [1, 1],
            'blue': [0, 1],
            'yellow': [1, 0],
        }

        side_offset = MINI_STICKER_AREA_TILE_GAP * 3

        side_size = MINI_STICKER_AREA_TILE_SIZE * 3 + MINI_STICKER_AREA_TILE_GAP * 2

        offset_x = self.width - (side_size * 4) - (side_offset * 3) - MINI_STICKER_AREA_OFFSET
        offset_y = self.height - (side_size * 3) - (side_offset * 2) - MINI_STICKER_AREA_OFFSET

        for side, (grid_x, grid_y) in grid.items():
            index = -1
            for row in range(3):
                for col in range(3):
                    index += 1
                    x1 = int((offset_x + MINI_STICKER_AREA_TILE_SIZE * col) + (MINI_STICKER_AREA_TILE_GAP * col) + (
                            (side_size + side_offset) * grid_x))
                    y1 = int((offset_y + MINI_STICKER_AREA_TILE_SIZE * row) + (MINI_STICKER_AREA_TILE_GAP * row) + (
                            (side_size + side_offset) * grid_y))
                    x2 = int(x1 + MINI_STICKER_AREA_TILE_SIZE)
                    y2 = int(y1 + MINI_STICKER_AREA_TILE_SIZE)

                    foreground_color = COLOR_PLACEHOLDER
                    if side in self.result_state:
                        foreground_color = color_detector.get_prominent_color(self.result_state[side][index])

                    # shadow
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 0),
                        -1
                    )

                    # foreground color
                    cv2.rectangle(
                        frame,
                        (x1 + 1, y1 + 1),
                        (x2 - 1, y2 - 1),
                        foreground_color,
                        -1
                    )

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
            # Get the center color of the current side.
            center_bgr = self.result_state[side][4]

            # Compare the center color to all neighbors. If we come across a
            # different color, then we can assume the cube isn't solved yet.
            for bgr in self.result_state[side]:
                if center_bgr != bgr:
                    return False
        return True

    def draw_scanned_successfully(self, frame, successfully):
        text = 'Scanned Successfully!' if successfully else 'Scanned Failed!'
        self.render_text(frame, text, (20, self.height - 40))

    def run(self):
        while True:
            _, frame = self.cam.read()
            key = cv2.waitKey(10) & 0xff

            # Quit on escape.
            if key == 27:
                break

            if not self.calibrate_mode:
                # Update the snapshot when space bar is pressed.
                if key == 32:
                    self.update_snapshot_state(frame)

            # Toggle calibrate mode.
            if key == ord(CALIBRATE_MODE_KEY):
                self.reset_calibrate_mode()
                self.calibrate_mode = not self.calibrate_mode

            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurredFrame = cv2.blur(grayFrame, (3, 3))
            cannyFrame = cv2.Canny(blurredFrame, 30, 60, 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            dilatedFrame = cv2.dilate(cannyFrame, kernel)
            cv2.imshow("dilated", dilatedFrame)

            contours = self.find_contours(dilatedFrame)
            if len(contours) == 9:
                self.draw_contours(frame, contours)
                if not self.calibrate_mode:
                    self.update_preview_state(frame, contours)
                elif key == 32 and self.done_calibrating == False:
                    current_color = self.colors_to_calibrate[self.current_color_to_calibrate_index]
                    (x, y, w, h) = contours[4]
                    roi = frame[y + 7:y + h - 7, x + 14:x + w - 14]
                    avg_bgr = color_detector.get_dominant_color(roi)
                    self.calibrated_colors[current_color] = avg_bgr
                    self.current_color_to_calibrate_index += 1
                    self.done_calibrating = self.current_color_to_calibrate_index == len(self.colors_to_calibrate)
                    if self.done_calibrating:
                        color_detector.set_cube_color_pallete(self.calibrated_colors)
                        config.set_setting(CUBE_PALETTE, color_detector.cube_color_palette)

            if self.calibrate_mode:
                self.draw_current_color_to_calibrate(frame)
                self.draw_calibrated_colors(frame)
            else:
                self.draw_preview_stickers(frame)
                self.draw_current_stickers(frame)
                self.draw_scanned_sides(frame)
                self.draw_2d_cube_state(frame)
                if len(self.result_state.keys()) == 6:
                    self.draw_scanned_successfully(frame, self.scanned_successfully())

            cv2.imshow("Visiar - Rubik's cube solver", frame)

        self.cam.release()
        cv2.destroyAllWindows()

        if len(self.result_state.keys()) != 6:
            return E_INCORRECTLY_SCANNED

        if not self.scanned_successfully():
            return E_INCORRECTLY_SCANNED

        if self.state_already_solved():
            return E_ALREADY_SOLVED

        return utils.solve(self.get_result_notation(), 'Kociemba')


webcam = Webcam()
print(webcam.run())
