import csv
import cv2
import numpy
import glob
from machine import get_moments, get_contours


def write_descriptors():
    file = open(r'descriptores.csv', 'w')
    writer = csv.writer(file)

    for idx, img in enumerate(glob.glob(r'./imagenes/*.png')):
        contour = get_contours(cv2.imread(img))[0]
        hu_moments = get_moments(contour)
        hu_moments = hu_moments.flatten()
        file2 = open(r'supervision.csv', 'r')
        reader = csv.reader(file2, delimiter=',')
        for row in reader:
            if row[0] == img[11:-4]:
                writer.writerow(numpy.insert(hu_moments, 0, row[1]))


write_descriptors()
