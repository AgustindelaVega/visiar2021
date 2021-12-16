import cv2
from color_utils import color_detector
from constants import (
    COLOR_PLACEHOLDER,
    STICKER_AREA_TILE_GAP,
    MINI_STICKER_AREA_TILE_SIZE,
    MINI_STICKER_AREA_TILE_GAP,
    MINI_STICKER_AREA_OFFSET,
    STICKER_AREA_TILE_SIZE,
    STICKER_AREA_OFFSET_X,
    STICKER_CONTOUR_COLOR,
    CALIBRATE_MODE_KEY,
    TEXT_SIZE,
    STICKER_AREA_OFFSET_Y
)


# draw main grid of stickers
def draw_stickers(frame, stickers, offset_x, offset_y):
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
def draw_preview_stickers(webcam, frame):
    draw_stickers(frame, webcam.preview_state, STICKER_AREA_OFFSET_X, STICKER_AREA_OFFSET_Y)


# draw the current scanned sticker
def draw_current_stickers(webcam, frame):
    y = STICKER_AREA_OFFSET_Y + int(STICKER_AREA_OFFSET_Y / 4)
    draw_stickers(frame, webcam.snapshot_state, STICKER_AREA_OFFSET_X, y)


# draw a list of sticker contours
def draw_contours(webcam, frame, contours):
    if webcam.calibrate_mode:
        # Only show the center piece contour
        draw_sticker(frame, contours[4])
    else:
        for contour in contours:
            draw_sticker(frame, contour)


# draw the contour of a given sticker
def draw_sticker(frame, contour):
    (x, y, w, h) = contour
    cv2.rectangle(frame, (x, y), (x + w, y + h), STICKER_CONTOUR_COLOR, 2)


# show text on the given frame
def render_text(frame, text, pos, color=(255, 255, 255)):
    get_text_size()
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA,
                bottomLeftOrigin=None)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color, thickness=1, lineType=cv2.LINE_AA,
                bottomLeftOrigin=None)


def get_text_size(size=TEXT_SIZE):
    return (size, size), size


# draw current color to calibrate
def draw_current_color_to_calibrate(webcam, frame):
    y_offset = 40
    font_size = int(TEXT_SIZE * 1.25)
    if webcam.done_calibrating:
        messages = [
            'Calibrated Successfully!',
            'To exit Calibration mode press: ' + CALIBRATE_MODE_KEY,
        ]
        for index, text in enumerate(messages):
            (textsize_width, textsize_height), _ = get_text_size(font_size)
            y = y_offset + (textsize_height + 10) * int(index * 3)
            render_text(frame, text, (int(webcam.width / 2 - textsize_width / 2), y))
    else:
        current_color = webcam.colors_to_calibrate[webcam.current_calibrate_index]
        draw_current_sticker_color_to_calibrate(webcam, frame, color_detector.convert_name_to_bgr(current_color))
        text = 'Please scan side '
        (textsize_width, textsize_height), _ = get_text_size(font_size)
        render_text(frame, text, (int(webcam.width - 180 - textsize_width / 2), y_offset))


def draw_current_sticker_color_to_calibrate(webcam, frame, current_color):
    x1 = webcam.width - 90
    y1 = 20
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
        current_color,
        -1
    )


# draw the history of calibrated colors
def draw_calibrated_colors(webcam, frame):
    offset_y = 20
    for index, (color_name, color_bgr) in enumerate(webcam.calibrated_colors.items()):
        x1 = 90
        y1 = int(offset_y + STICKER_AREA_TILE_SIZE * index)
        x2 = x1 + STICKER_AREA_TILE_SIZE
        y2 = y1 + STICKER_AREA_TILE_SIZE

        # shadow
        cv2.rectangle(
            frame,
            (x1, y1 + 20),
            (x2, y2 - 10),
            (0, 0, 0),
            -1
        )

        # foreground
        cv2.rectangle(
            frame,
            (x1 + 1, y1 + 19),
            (x2 - 1, y2 - 9),
            tuple([int(c) for c in color_bgr]),
            -1
        )
        render_text(frame, color_name, (20, y1 + 20))


def draw_scanned_successfully(webcam, frame, successfully):
    text = 'Scanned Successfully!' if successfully else 'Scanned Failed!'
    render_text(frame, text, (20, webcam.height - 30))


# draw cube state
def draw_2d_cube_state(webcam, frame):
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

    offset_x = webcam.width - (side_size * 4) - (side_offset * 3) - MINI_STICKER_AREA_OFFSET
    offset_y = webcam.height - (side_size * 3) - (side_offset * 2) - MINI_STICKER_AREA_OFFSET

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
                if side in webcam.result_state:
                    foreground_color = color_detector.get_prominent_color(webcam.result_state[side][index])

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
