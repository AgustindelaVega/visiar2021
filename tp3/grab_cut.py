import cv2
import numpy as np

webcam_window = 'Webcam-Image-Window'
segmented_image_window = 'Segmented-Image-Window'


def grab_cut(img):
    # creamos una mascara con puros ceros
    mask = np.zeros(img.shape[:2], np.uint8)

    # Arrays usados internamente por grabCut para crear los modelos de background y foreground
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # usamos roi para agarrar el area de interes
    rect = cv2.selectROI("Select-ROI", img, fromCenter=False, showCrosshair=False)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    img = img * mask2[:, :, np.newaxis]

    cv2.imshow(segmented_image_window, img)
    cv2.waitKey()


def main():
    cv2.namedWindow(webcam_window)
    cv2.namedWindow(segmented_image_window)

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        # frame = cv2.imread('./1.jpeg')

        cv2.imshow(webcam_window, frame)

        if cv2.waitKey(10) & 0xFF == ord('p'):
            grab_cut(frame.copy())

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == '__main__':
    main()