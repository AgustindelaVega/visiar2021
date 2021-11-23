import cv2
import math
from color_utils import get_colour_name, convert_rgb_to_names, nearest_colour

webcam_window = 'Webcam-Image-Window'
segmented_image_window = 'Segmented-Image-Window'


def main():
    cv2.namedWindow(webcam_window)
    cv2.namedWindow(segmented_image_window)

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        height = math.floor((frame.shape[0]) / 16)
        width = math.floor((frame.shape[1]) / 19)
        # frame = cv2.imread('./1.jpeg')

        start_x = math.floor(width * 6)
        start_y = math.floor(height * 6)
        end_x = math.floor(width * 6 * 2)
        end_y = math.floor(height * 6 * 2)

        range_x = math.floor((end_x - start_x) / 3)
        range_y = math.floor((end_y - start_y) / 3)
        color = (255, 255, 0)
        cv2.line(frame, (start_x, start_y), (start_x, end_y), color, 2, 1)
        cv2.line(frame, (start_x, start_y), (end_x, start_y), color, 2, 1)
        cv2.line(frame, (start_x, end_y), (end_x, end_y), color, 2, 1)
        cv2.line(frame, (end_x, start_y), (end_x, end_y), color, 2, 1)

        cv2.line(frame, (start_x + range_x, start_y), (start_x + range_x, end_y), color, 2, 1)
        cv2.line(frame, (start_x + range_x * 2, start_y), (start_x + range_x * 2, end_y), color, 2, 1)

        cv2.line(frame, (start_x, start_y + range_y), (end_x, start_y + range_y), color, 2, 1)
        cv2.line(frame, (start_x, start_y + range_y * 2), (end_x, start_y + range_y * 2), color, 2, 1)

        points_range_x = math.floor((end_x - start_x) / 6)
        points_range_y = math.floor((end_y - start_y) / 6)

        points = [
            (start_x + points_range_x, start_y + points_range_y),
            (start_x + points_range_x, start_y + points_range_y * 3),
            (start_x + points_range_x, start_y + points_range_y * 5),
            (start_x + points_range_x * 3, start_y + points_range_y),
            (start_x + points_range_x * 3, start_y + points_range_y * 3),
            (start_x + points_range_x * 3, start_y + points_range_y * 5),
            (start_x + points_range_x * 5, start_y + points_range_y),
            (start_x + points_range_x * 5, start_y + points_range_y * 3),
            (start_x + points_range_x * 5, start_y + points_range_y * 5),

        ]

        colors = []
        for idx, point in enumerate(points):
            col = frame[(point[1], point[0])]
            colors.append(col)
            print(col, "-", nearest_colour(col))

        cv2.imshow(webcam_window, frame)

        if cv2.waitKey(10) & 0xFF == ord('p'):
            print("here")
            # grab_cut(frame.copy())

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == '__main__':
    main()