import cv2

from tp1.contour import get_biggest_contours, compare_contours, get_external_contours
from tp1.frame_editor import denoise, draw_contours, draw_name
from tp1.trackbar import create_trackbar, get_trackbar_value
main_window = 'Window'
main_window_trackbar = 'Trackbar'
contours_window = 'Window - Contours'
contours_window_trackbar = 'Trackbar - Contours'


def main():

    cv2.namedWindow(main_window)
    cv2.namedWindow(contours_window)
    cap = cv2.VideoCapture(2)
    biggest_contour = None
    color_white = (255, 255, 255)
    color_green = (0, 255, 0)
    color_red = (0, 0, 255)
    color_blue = (255, 0, 0)
    colors = [color_green, color_red, color_blue]
    create_trackbar(main_window_trackbar, main_window, 151)
    create_trackbar(contours_window_trackbar, contours_window, 1000)
    saved_contours = []

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 0)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        trackbar_val = get_trackbar_value(main_window_trackbar, main_window)
        threshold_frame = cv2.adaptiveThreshold(gray_frame, 151, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, trackbar_val, 0)

        frame_denoised = denoise(threshold_frame, cv2.MORPH_ELLIPSE, 10)
        contours, hierarchy = cv2.findContours(frame_denoised, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            external_contours = get_external_contours(contours, hierarchy)
            biggest_contours = get_biggest_contours(external_contours, 3)
            biggest_contour = biggest_contours[0]
            trackbar_contour_val = get_trackbar_value(main_window_trackbar, contours_window) / 1000

            for idx, contour in enumerate(biggest_contours):
                val = compare_contours(contour_to_compare=contour, saved_contours=saved_contours, max_diff=trackbar_contour_val)
                if val[0]:
                    draw_contours(frame=frame, contours=contour, color=colors[idx], thickness=30)
                    draw_name(frame=frame, value=val[1], color=color_white, thickness=3)
                else:
                    draw_contours(frame=frame, contours=contour, color=color_white, thickness=3)

        cv2.imshow(main_window, frame_denoised)
        cv2.imshow(contours_window, frame)

        if cv2.waitKey(1) & 0xFF == ord('i'):
            if biggest_contour is not None:
                val = 'Circulo'
                saved_contours.append((biggest_contour, val))
                print('Saved "Circulo"')

        if cv2.waitKey(1) & 0xFF == ord('u'):
            if biggest_contour is not None:
                val = 'Cuadrado'
                saved_contours.append((biggest_contour, val))
                print('Saved "Cuadrado"')

        if cv2.waitKey(1) & 0xFF == ord('t'):
            if biggest_contour is not None:
                val = 'Triangulo'
                saved_contours.append((biggest_contour, val))
                print('Saved "Triangulo"')

        if cv2.waitKey(1) & 0xFF == ord('r'):
            saved_contours = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


main()
