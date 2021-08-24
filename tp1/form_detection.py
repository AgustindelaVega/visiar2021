import cv2

from tp1.contour import get_biggest_contours, compare_contours
from tp1.frame_editor import denoise, draw_contours, draw_name
from tp1.trackbar import create_trackbar, get_trackbar_value

threshold_window = 'Window - Threshold'
threshold_trackbar = 'Trackbar'

denoise_window = 'Window'
structuring_element_size_trackbar = 'Trackbar - StructuringElement'
min_area_trackbar = 'Trackbar - MinArea'

contours_window = 'Window - Contours'
contours_trackbar = 'Trackbar - Contours'

white = (255, 255, 255)
pink = (204, 0, 163)
green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)

colors = [green, red, blue]

contours_amount = 3


def main():

    cv2.namedWindow(threshold_window)
    cv2.namedWindow(denoise_window)
    cv2.namedWindow(contours_window)

    # create_trackbar(threshold_trackbar, threshold_window, slider_max=255)
    create_trackbar(structuring_element_size_trackbar, denoise_window, slider_max=20, initial_value=5)
    create_trackbar(contours_trackbar, contours_window, slider_max=1000)
    create_trackbar(min_area_trackbar, denoise_window, slider_max=480*640, initial_value=10000)

    biggest_contour = None  # Current frame biggest contour
    saved_contours = []

    cap = cv2.VideoCapture(2)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 0)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # threshold_val = int(get_trackbar_value(main_window_trackbar, main_window) / 2) * 2 + 3  # No need to use when using otsu threshold, it determines it automatically

        _, threshold_frame = cv2.threshold(gray_frame, 255, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        trackbar_structuring_element_val = get_trackbar_value(structuring_element_size_trackbar, denoise_window) + 1
        frame_denoised = denoise(threshold_frame, cv2.MORPH_ELLIPSE, trackbar_structuring_element_val)

        contours, _ = cv2.findContours(frame_denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # only external contours and no approximation

        trackbar_min_area_val = get_trackbar_value(min_area_trackbar, denoise_window)
        biggest_contours = get_biggest_contours(contours, contours_amount, trackbar_min_area_val)

        if len(biggest_contours) > 0:
            biggest_contour = biggest_contours[0]
            trackbar_contour_val = get_trackbar_value(contours_trackbar, contours_window) / 1000

            for idx, contour in enumerate(biggest_contours):
                val = compare_contours(contour, saved_contours, max_diff=trackbar_contour_val)
                if val[0]:
                    draw_contours(frame, contour, colors[idx], thickness=30)
                    draw_name(frame, val[1], pink, thickness=3)
                else:
                    draw_contours(frame, contour, red, thickness=3)

        cv2.imshow(threshold_window, threshold_frame)
        cv2.imshow(denoise_window, frame_denoised)
        cv2.imshow(contours_window, frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            if biggest_contour is not None:
                saved_contours.append((biggest_contour, 'Circulo'))
                print('Saved "Circulo"')

        if cv2.waitKey(1) & 0xFF == ord('r'):
            if biggest_contour is not None:
                saved_contours.append((biggest_contour, 'Rectangulo'))
                print('Saved "Rectangulo"')

        if cv2.waitKey(1) & 0xFF == ord('t'):
            if biggest_contour is not None:
                saved_contours.append((biggest_contour, 'Triangulo'))
                print('Saved "Triangulo"')

        if cv2.waitKey(1) & 0xFF == ord('k'):
            if biggest_contour is not None:
                val = input()
                saved_contours.append((biggest_contour, val))
                print('Saved ', val)

        if cv2.waitKey(1) & 0xFF == ord('f'):
            saved_contours = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


main()
