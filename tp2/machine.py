import cv2
import os

from tp1.frame_editor import denoise
from tp1.frame_editor import draw_contours, draw_name
from tp1.contour import get_biggest_contours
from math import copysign, log10
from train import train_model
import csv
import numpy as np
from tp1.trackbar import create_trackbar, get_trackbar_value

colors = [(255, 102, 153), (153, 255, 102), (204, 153, 0)]

block_size_trackbar = 'Threshold-Block-Size-Trackbar'
denoise_radius_trackbar = 'Denoise-Radius-Trackbar'
binary_window = 'Binary-Window'

min_area_trackbar = 'Min-Area-Trackbar'
drawed_window = 'Drawed-Window'


def get_denoised(frame, trackbar):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if trackbar:
        block_size = int(np.ceil(get_trackbar_value(block_size_trackbar, binary_window)) // 2 * 2 + 3)
        denoise_radius = get_trackbar_value(denoise_radius_trackbar, binary_window) + 1
    else:
        block_size = 501
        denoise_radius = 10
    threshold_frame = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                            block_size,
                                            10)
    return denoise(threshold_frame, cv2.MORPH_ELLIPSE, denoise_radius)


def get_moments(contour):
    return get_hu_moments(contour)


def get_contours(frame, trackbar):
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if trackbar:
        value = get_trackbar_value(min_area_trackbar, drawed_window)
    else:
        value = 10000
    return get_biggest_contours(contours, 3, value)


def get_hu_moments(contour):
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments)
    for i in range(len(hu_moments)):
        if hu_moments[i] != 0:
            hu_moments[i] = -1 * copysign(1.0, hu_moments[i]) * log10(abs(hu_moments[i]))
    return hu_moments


def evaluate_tree(tree, moments):
    data = np.array([moments], dtype=np.float32)
    return tree.predict(data)[0]


def int_to_label(int_value):
    file2 = open(r'etiquetas.csv', 'r')
    reader = csv.reader(file2, delimiter=',')
    for row in reader:
        if int(row[0]) == int_value:
            return row[1]


def main():
    if os.path.exists(r'./tree.yml'):
        trained_tree = cv2.ml.DTrees_load(r'./tree.yml')
    else:
        trained_tree = train_model()
        trained_tree.save(r'./tree.yml')

    cv2.namedWindow(binary_window)
    create_trackbar(block_size_trackbar, binary_window, slider_max=480*640, initial_value=1)
    create_trackbar(denoise_radius_trackbar, binary_window, slider_max=20, initial_value=1)

    cv2.namedWindow(drawed_window)
    create_trackbar(min_area_trackbar, drawed_window, slider_max=480*640, initial_value=5000)

    cap = cv2.VideoCapture(2)

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 0)

        denoised_frame = get_denoised(frame, True)
        contours = get_contours(denoised_frame, True)
        drawed = frame

        for idx, contour in enumerate(contours):
            moments = get_moments(contour)
            drawed = draw_contours(drawed, contour, colors[idx], 5)
            prediction = int_to_label(evaluate_tree(trained_tree, moments))
            draw_name(drawed, (contour, prediction), (0, 153, 255), thickness=3)

        cv2.imshow(drawed_window, drawed)
        cv2.imshow(binary_window, denoised_frame)
        if cv2.waitKey(1) & 0xFF:
            continue

if __name__ == '__main__':
    main()
