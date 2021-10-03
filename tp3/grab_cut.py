import cv2
import numpy as np

binary_window = 'Binary-Window'
selected_image_window = 'Selected-Image-Window'


def grab_cut(img):
    # si usamos el metodo de GC_INIT_WITH_RECT no es necesario camara por eso hacemos una matriz de 0
    mask = np.zeros(img.shape[:2], np.uint8)

    # These are arrays used by the algorithm internally. You just create two np.float64 type zero arrays
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # usamos roi para agarrar el rect
    rect = cv2.selectROI("img", img, fromCenter=False, showCrosshair=False)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    img = img * mask2[:, :, np.newaxis]

    cv2.imshow(selected_image_window, img)
    cv2.waitKey()


def main():
    cv2.namedWindow(binary_window)
    cv2.namedWindow(selected_image_window)

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        # frame = cv2.imread('./1.jpeg')

        cv2.imshow(binary_window, frame)

        if cv2.waitKey(10) & 0xFF == ord('p'):
            grab_cut(frame.copy())

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == '__main__':
    main()