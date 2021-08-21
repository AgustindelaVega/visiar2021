import cv2


def get_biggest_contours(contours, amount):
    return sorted(contours, key=cv2.contourArea, reverse=True)[0:amount]


def get_external_contours(contours, hierarchy):
    external_contours = []
    for idx, contour in enumerate(contours):
        if hierarchy[0, idx, 3] == -1:
            external_contours.append(contour)

    return external_contours


def compare_contours(contour_to_compare, saved_contours, max_diff):
    for contour in saved_contours:  # contour is [contour, name] tuple
        if cv2.matchShapes(contour_to_compare, contour[0], cv2.CONTOURS_MATCH_I2, 0) < max_diff:
            return True, (contour_to_compare ,contour[1])
    return False, None
