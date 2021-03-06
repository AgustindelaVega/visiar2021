import cv2


def denoise(frame, method, radius):
    kernel = cv2.getStructuringElement(method, (radius, radius))
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)  # erosion then dilation / cleans white dots
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)  # dilation then erosion / cleans black dots / not needed
    return closing


def draw_contours(frame, contours, color, thickness):
    cv2.drawContours(frame, contours, -1, color, thickness)
    return frame


def draw_name(frame, value, color, thickness):
    contour = value[0]
    contour_name = value[1]

    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    cv2.putText(frame, contour_name, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)


