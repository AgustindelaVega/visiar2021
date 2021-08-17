import cv2
import imutils


def denoise(frame, method, radius):
    kernel = cv2.getStructuringElement(method, (radius, radius))
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing


def apply_color_convertion(frame, color):
    return cv2.cvtColor(frame, color)


def adaptive_threshold(frame, slider_max, adaptative, binary, trackbar_value):
    return cv2.adaptiveThreshold(frame, slider_max, adaptative, binary, trackbar_value, 0)


def draw_contours(frame, contours, color, thickness):
    # -1 for all contours
    cv2.drawContours(frame, contours, -1, color, thickness)
    return frame


def draw_name(frame, value, color, thickness):
    M = cv2.moments(value[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    cv2.putText(frame, value[1], (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)


