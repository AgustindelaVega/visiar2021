import csv
import cv2
import numpy
import glob
from machine import get_moments, get_contours, get_denoised
from tp1.frame_editor import draw_contours


def write_descriptors():
    file = open(r'descriptores.csv', 'w')
    writer = csv.writer(file)
    cv2.namedWindow('descriptors-window')

    for idx, img in enumerate(glob.glob(r'./imagenes/*.png')):
        frame = cv2.imread(img)
        denoised_frame = get_denoised(frame, False)
        contour = get_contours(denoised_frame, False)[0]
        drawed = frame
        drawed = draw_contours(drawed, contour, (255, 0, 0), 5)
        cv2.imshow('descriptors-window', drawed)
        hu_moments = get_moments(contour)
        hu_moments = hu_moments.flatten()
        file2 = open(r'supervision.csv', 'r')
        reader = csv.reader(file2, delimiter=',')
        for row in reader:
            if row[0] == img[11:-4]:
                writer.writerow(numpy.insert(hu_moments, 0, row[1]))

        if cv2.waitKey(100) & 0xFF:
            continue


write_descriptors()
