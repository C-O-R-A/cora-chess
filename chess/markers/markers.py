import cv2
import numpy as np

# Define the aruco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

number_of_markers = 4
marker_size = 200

for i in range(0,number_of_markers,1):
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, i+1, marker_size)
    cv2.imwrite(f"markers/marker_{i}.png", marker_image)

