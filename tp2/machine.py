import glob
import cv2
import numpy

from tp1.frame_editor import denoise
from tp1.frame_editor import draw_contours, draw_name
from tp1.contour import get_biggest_contours
from math import copysign, log10
import csv
import numpy as np


def write_hu_moments():
    file = open(r'descriptores.csv', 'w')
    writer = csv.writer(file)

    for idx, img in enumerate(glob.glob(r'./imagenes/*.png')):
        hu_moments = get_moments(cv2.imread(img))
        hu_moments = hu_moments.flatten()
        file2 = open(r'supervision.csv', 'r')
        reader = csv.reader(file2, delimiter=',')
        for row in reader:
            if row[0] == img[11:-4]:
                writer.writerow(numpy.insert(hu_moments, 0, row[1]))


def get_denoised(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    threshold_frame = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 511,
                                            10)
    return denoise(threshold_frame, cv2.MORPH_ELLIPSE, 5)


def get_moments(contours):
    return get_hu_moments(contours[0])


def get_contours(frame):
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return get_biggest_contours(contours, 1, 0)


def get_hu_moments(contour):
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments)
    for i in range(len(hu_moments)):
        hu_moments[i] = -1 * copysign(1.0, hu_moments[i]) * log10(abs(hu_moments[i]))
    return hu_moments


def load_training_set():
    train_data = []
    train_labels = []
    with open(r'./descriptores.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            class_label = row[0]
            hu_moments = row[1:len(row)]
            floats = []
            for n in hu_moments:
                floats.append(float(n))  # tiene los momentos de Hu transformados a float.
            train_data.append(np.array(floats, dtype=np.float32))  # momentos de Hu
            train_labels.append(np.array([float(class_label)], dtype=np.int32))  # Resultados
            # Valores y resultados se necesitan por separados
    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)
    return train_data, train_labels
# transforma los arrays a arrays de forma numpy


# llama la funcion de arriba, se manda a entrenar y devuelve el modelo entrenado
def train_model():
    train_data, train_labels = load_training_set()

    tree = cv2.ml.DTrees_create()
    tree.setCVFolds(1)
    tree.setMaxDepth(10)
    tree.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    return tree


def evaluate_tree(tree, moments):
    data = np.array([moments], dtype=np.float32)
    return tree.predict(data)[0]


def int_to_label(int_value):
    if int_value == 0:
        return '5-point-star'
    if int_value == 1:
        return 'rectangle'
    if int_value == 2:
        return 'triangle'
    else:
        raise Exception('unkown class_label')


def main():
    trained_tree = train_model()
    cv2.namedWindow("binary-window")
    cv2.namedWindow("draw-window")
    cap = cv2.VideoCapture(2)

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 0)
        denoised_frame = get_denoised(frame)
        contours = get_contours(denoised_frame)
        moments = get_moments(contours)
        drawed = draw_contours(frame, contours, (255, 0, 0), 5)
        prediction = int_to_label(evaluate_tree(trained_tree, moments))
        print(prediction)
        draw_name(drawed, (contours[0], prediction), (0, 255, 0), thickness=3)
        cv2.imshow("draw-window", drawed)
        cv2.imshow("binary-window", denoised_frame)
        if cv2.waitKey(1) & 0xFF:
            continue


main()
