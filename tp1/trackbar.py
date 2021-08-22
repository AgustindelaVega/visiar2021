import cv2


def create_trackbar(trackbar_name, window_name, slider_max, initial_value=0):
    cv2.createTrackbar(trackbar_name, window_name, initial_value, slider_max, on_trackbar)


def on_trackbar(val):
    pass


def get_trackbar_value(trackbar_name, window_name):
    return cv2.getTrackbarPos(trackbar_name, window_name)
