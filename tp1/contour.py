import cv2


def get_biggest_contours(contours, amount, min_area):
    for cont in contours:
        print(cv2.contourArea(cont))

    contours = filter(lambda x: (cv2.contourArea(x) >= min_area), contours)
    return sorted(contours, key=cv2.contourArea, reverse=True)[0:amount]


def compare_contours(contour_to_compare, saved_contours, max_diff):
    for contour in saved_contours:  # contour is [contour, name] tuple
        if cv2.matchShapes(contour_to_compare, contour[0], cv2.CONTOURS_MATCH_I2, 0) < max_diff:
            return True, (contour_to_compare, contour[1])
    return False, None
