import cv2

from tp1.contour import get_biggest_contours, compare_contours, get_external_contours
from tp1.frame_editor import denoise, draw_contours, draw_name
from tp1.trackbar import create_trackbar, get_trackbar_value
main_window = 'Window'
main_window_trackbar = 'Trackbar'
contours_window = 'Window - Contours'
contours_window_trackbar = 'Trackbar - Contours'

white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)

colors = [green, red, blue]

contours_amount = 3

# TODO:
# es buena idea incluir una barra de desplazamiento para ajustar el tamaño del elemento estructural
# Filtrar contornos que se pueden descartar de antemano, por ejemplo, por tener un área muy pequeña
# establecer un umbral de distancia máxima de validez, es conveniente una barra deslizante para ajustar este valor
# contorno en rojo para objetos desconocidos


def main():

    cv2.namedWindow(main_window)
    cv2.namedWindow(contours_window)

    create_trackbar(main_window_trackbar, main_window, slider_max=255)
    create_trackbar(contours_window_trackbar, contours_window, slider_max=1000)

    biggest_contour = None  # Current frame biggest contour
    saved_contours = []

    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 0)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        threshold_val = int(get_trackbar_value(main_window_trackbar, main_window) / 2) * 2 + 3  # No need to use when using otsu threshold, it determines it automatically

        _, threshold_frame = cv2.threshold(gray_frame, threshold_val, maxval=255, type=cv2.THRESH_BINARY)

        frame_denoised = denoise(threshold_frame, cv2.MORPH_ELLIPSE, 10)
        contours, _ = cv2.findContours(frame_denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # only external contours and no approximation

        if len(contours) > 0:
            biggest_contours = get_biggest_contours(contours, contours_amount)
            biggest_contour = biggest_contours[0]
            trackbar_contour_val = get_trackbar_value(contours_window_trackbar, contours_window) / 1000

            for idx, contour in enumerate(biggest_contours):
                val = compare_contours(contour, saved_contours, max_diff=trackbar_contour_val)
                if val[0]:
                    draw_contours(frame, contour, colors[idx], thickness=30)
                    draw_name(frame, val[1], black, thickness=3)
                else:
                    draw_contours(frame, contour, white, thickness=3)

        cv2.imshow(main_window, frame_denoised)
        cv2.imshow(contours_window, frame)

        if cv2.waitKey(1) & 0xFF == ord('i'):
            if biggest_contour is not None:
                saved_contours.append((biggest_contour, 'Circulo'))
                print('Saved "Circulo"')

        if cv2.waitKey(1) & 0xFF == ord('u'):
            if biggest_contour is not None:
                saved_contours.append((biggest_contour, 'Cuadrado'))
                print('Saved "Cuadrado"')

        if cv2.waitKey(1) & 0xFF == ord('t'):
            if biggest_contour is not None:
                saved_contours.append((biggest_contour, 'Triangulo'))
                print('Saved "Triangulo"')

        if cv2.waitKey(1) & 0xFF == ord('k'):
            if biggest_contour is not None:
                val = input()
                saved_contours.append((biggest_contour, val))
                print('Saved ', val)

        if cv2.waitKey(1) & 0xFF == ord('r'):
            saved_contours = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


main()
