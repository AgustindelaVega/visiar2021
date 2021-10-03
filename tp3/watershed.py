import cv2
import sys
import numpy as np
from PIL import ImageColor

base_colours = ['#37AB65', '#3DF735', '#AD6D70', '#EC2504', '#8C0B90', '#C0E4FF', '#27B502', '#7C60A8', '#CF95D7', '#37AB65']
binary_window = 'Binary-Window'


def watershed(img):
    np.set_printoptions(threshold=sys.maxsize)

    markers = cv2.watershed(img, np.int32(seeds))

    img[markers == -1] = [0, 0, 255]
    for n in range(1, 10):
        img[markers == n] = ImageColor.getcolor(base_colours[n], "RGB")

    cv2.imshow("img", img)

    cv2.waitKey()


def click_event(event, x, y, _flags, _params):
    if event == cv2.EVENT_LBUTTONDOWN:
        val = int(chr(selected_key))
        points.append(((x, y), val, selected_key))
        cv2.circle(seeds, (x, y), 7, (val, val, val), thickness=-1)


def main():
    global points
    global seeds
    global frame
    global selected_key
    selected_key = 49
    points = []
    # seeds = np.zeros((1198, 1198), np.uint8)
    seeds = np.zeros((480, 640), np.uint8)
    cv2.namedWindow("frame")
    cv2.namedWindow("seeds")

    cap = cv2.VideoCapture(0)
    cv2.setMouseCallback("frame", click_event)

    while True:
        _, frame = cap.read()
        # frame = cv2.imread('./1.jpeg')
        frame_copy = frame.copy()
        seeds_copy = seeds.copy()

        for point in points:
            color = ImageColor.getcolor(base_colours[int(chr(point[2]))], "RGB")
            val = int(chr(point[2])) * 20

            cv2.circle(frame_copy, (point[0][0], point[0][1]), 7, val, thickness=-1)
            cv2.circle(seeds_copy, (point[0][0], point[0][1]), 7, val, thickness=-1)
            cv2.putText(frame_copy, chr(point[2]), (point[0][0] - 20, point[0][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)

        cv2.imshow("frame", frame_copy)
        map = cv2.applyColorMap(seeds_copy, cv2.COLORMAP_JET)
        cv2.imshow("seeds", map)

        key = cv2.waitKey(100) & 0xFF
        if key == 32:
            watershed(frame.copy())
            points = []
            seeds = np.zeros((480, 640), np.uint8)

        if ord('1') <= key <= ord('9'):
            selected_key = key

        if key == ord('q'):
            break

    cap.release()


if __name__ == '__main__':
    main()
