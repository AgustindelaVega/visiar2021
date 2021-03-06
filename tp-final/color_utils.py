import numpy as np
import cv2
from helpers import ciede2000, bgr2lab
from constants import COLOR_PLACEHOLDER


class ColorDetection:

    def __init__(self):
        self.prominent_color_palette = {
            'red'   : (0, 0, 255),
            'orange': (0, 165, 255),
            'blue'  : (255, 0, 0),
            'green' : (0, 255, 0),
            'white' : (255, 255, 255),
            'yellow': (0, 255, 255)
        }

        # Load colors from config and convert the list -> tuple.
        self.cube_color_palette = self.prominent_color_palette
        for side, bgr in self.cube_color_palette.items():
            self.cube_color_palette[side] = tuple(bgr)

    def get_prominent_color(self, bgr):
        for color_name, color_bgr in self.cube_color_palette.items():
            if tuple([int(c) for c in bgr]) == color_bgr:
                return self.prominent_color_palette[color_name]
        return COLOR_PLACEHOLDER

    @staticmethod
    def get_dominant_color(roi):
        pixels = np.float32(roi.reshape(-1, 3))

        n_colors = 1
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant = palette[np.argmax(counts)]
        return tuple(dominant)

    def get_closest_color(self, bgr):
        lab = bgr2lab(bgr)
        distances = []
        for color_name, color_bgr in self.cube_color_palette.items():
            distances.append({
                'color_name': color_name,
                'color_bgr': color_bgr,
                'distance': ciede2000(lab, bgr2lab(color_bgr))  # ciede2000 calculates the distance to a given color
            })
        closest = min(distances, key=lambda item: item['distance'])
        return closest

    def convert_bgr_to_color_initial(self, bgr):
        notations = {
            'green': 'g',
            'white': 'w',
            'blue': 'b',
            'red': 'r',
            'orange': 'o',
            'yellow': 'y'
        }
        color_name = self.get_closest_color(bgr)['color_name']
        return notations[color_name]

    def set_cube_color_pallete(self, palette):
        for side, bgr in palette.items():
            self.cube_color_palette[side] = tuple([int(c) for c in bgr])

    def convert_name_to_bgr(self, name):
        return self.prominent_color_palette[name]


color_detector = ColorDetection()
