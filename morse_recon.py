import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image

def construct_morse_graph(contours):
    morse_graph = defaultdict(list)
    for i, contour_i in enumerate(contours):
        for j, contour_j in enumerate(contours):
            if i != j:
                if is_neighbor(contour_i, contour_j):
                    morse_graph[i].append(j)
    return morse_graph

def is_neighbor(contour_i, contour_j, threshold=10):
    # Check if two contours are neighbors based on distance threshold
    distance = cv2.matchShapes(contour_i, contour_j, cv2.CONTOURS_MATCH_I1, 0)
    return distance < threshold

def complete_binary_map(binary_map, morse_graph, contours):
    for i, contour in enumerate(contours):
        if len(morse_graph[i]) == 0:
            # No neighbors, complete the contour
            cv2.drawContours(binary_map, [contour], -1, 255, -1)
        else:
            # Find the average of neighbors and connect to the contour
            avg_neighbor = np.mean([contours[j] for j in morse_graph[i]], axis=0).astype(int)
            cv2.drawContours(binary_map, [avg_neighbor], -1, 255, -1)
    return binary_map

# Load the binary map
binary_map = cv2.imread('merged256_new.tif', cv2.IMREAD_GRAYSCALE)

# Find contours in the binary map
contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Construct Morse graph
morse_graph = construct_morse_graph(contours)

# Complete the binary map
completed_map = complete_binary_map(binary_map.copy(), morse_graph, contours)

# Display the completed map
img=Image.fromarray(completed_map)
img.save('Comp_merge256.tif')
print('Reconstruction Successful')