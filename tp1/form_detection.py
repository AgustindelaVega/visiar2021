import cv2

from tp1.contour import get_contours, get_biggest_contour, compare_contours
from tp1.frame_editor import apply_color_convertion, adaptive_threshold, denoise, draw_contours, draw_name
from tp1.trackbar import create_trackbar, get_trackbar_value


def main():

    window_name = 'Window'
    trackbar_name = 'Trackbar'
    slider_max = 151
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(2)
    biggest_contour = None
    color_white = (255, 255, 255)
    color_green = (255, 0, 0)
    create_trackbar(trackbar_name, window_name, slider_max)
    # saved_hu_moments = load_hu_moments(file_name="hu_moments.txt")
    saved_contours = []

    while True:
        ret, frame = cap.read()
        gray_frame = apply_color_convertion(frame=frame, color=cv2.COLOR_RGB2GRAY)
        trackbar_val = get_trackbar_value(trackbar_name=trackbar_name, window_name=window_name)
        adapt_frame = adaptive_threshold(frame=gray_frame, slider_max=slider_max,
                                         adaptative=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         binary=cv2.THRESH_BINARY,
                                         trackbar_value=trackbar_val)
        frame_denoised = denoise(frame=adapt_frame, method=cv2.MORPH_ELLIPSE, radius=10)
        contours = get_contours(frame=frame_denoised, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            biggest_contour = get_biggest_contour(contours=contours)
            # hu_moments = get_hu_moments(contour=biggest_contour)

            val = compare_contours(contour_to_compare=biggest_contour, saved_contours=saved_contours, max_diff=1)
            if val[0]:
                draw_contours(frame=frame, contours=biggest_contour, color=color_green, thickness=30)
                draw_name(frame=frame, value=val[1], color=color_green, thickness=3)
            draw_contours(frame=frame, contours=biggest_contour, color=color_white, thickness=3)

        cv2.imshow(window_name, frame_denoised)
        cv2.imshow('Window - Contours', frame)
        if cv2.waitKey(1) & 0xFF == ord('k'):
            if biggest_contour is not None:
                val = input("Please enter object name: ")
                # val = "control"
                # save_moment(hu_moments=hu_moments, file_name="hu_moments.txt")
                saved_contours.append((biggest_contour, val))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


main()
