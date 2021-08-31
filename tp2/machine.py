import glob
import cv2
from tp1.frame_editor import denoise
from tp1.frame_editor import draw_contours
from math import copysign, log10


def main():

    for idx, img in enumerate(glob.glob(r'./shapes/triangle/*.png')):
        frame = cv2.imread(img)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        threshold_frame = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 511, 10)
        frame_denoised = denoise(threshold_frame, cv2.MORPH_ELLIPSE, 10)
        contours, _ = cv2.findContours(frame_denoised, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_NONE)  # only external contours and no approximation
        for cont in contours:
            draw_contours(frame, cont, (255, 0, 0), 5)

        cv2.imshow('test', frame_denoised)
        cv2.imshow('test2', frame)
        cv2.waitKey(1000)


def get_hu_moments(contour):
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments)
    for i in range(len(hu_moments)):
        hu_moments[i] = -1 * copysign(1.0, hu_moments[i]) * log10(abs(hu_moments[i]))
    return hu_moments


main()