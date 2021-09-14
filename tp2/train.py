import csv
import cv2
import numpy as np


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
                floats.append(float(n))
            train_data.append(np.array(floats, dtype=np.float32))
            train_labels.append(np.array([float(class_label)], dtype=np.int32))

    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)
    return train_data, train_labels


def train_model():
    train_data, train_labels = load_training_set()

    tree = cv2.ml.DTrees_create()
    tree.setCVFolds(1)
    tree.setMaxDepth(10)
    tree.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    return tree


if __name__ == '__main__':
    train_model()
