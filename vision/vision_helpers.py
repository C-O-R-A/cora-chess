
"""
Vision helpers for detecting markers and board moves
"""

import os
import chess
import cv2
import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from helpers import board_info

def detect_markers(im, verbose=False, show=False) -> dict:
    """
    Detects markers in an image of a chess board.
    Returns a list of detected marker positions.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary)
    corners, ids, _ = detector.detectMarkers(im)

    if verbose:
        print("Detected marker corners:", corners)
        print("Detected marker IDs:", ids)

    marker_poses = {}

    for i in range(len(corners)):
        marker_id = ids[i][0]
        marker_poses[f"marker_{marker_id}"] = corners[i]
        if show:
            # Draw the marker border
            cv2.polylines(im, [corners[i].astype(int)], True, (0, 255, 0), 2)

            # Label each corner
            corner_labels = [
                "TL",
                "TR",
                "BR",
                "BL",
            ]  # top-left, top-right, bottom-right, bottom-left
            for j, pt in enumerate(corners[i][0]):
                pt = tuple(pt.astype(int))
                cv2.circle(im, pt, 5, (0, 0, 255), -1)  # draw small circle
                cv2.putText(
                    im,
                    f"{corner_labels[j]}",
                    (pt[0] + 5, pt[1] - 5),  # offset text a bit
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

    if show:
        im_markers = cv2.aruco.drawDetectedMarkers(im.copy(), corners, ids)
        cv2.imshow("Detected Markers", im_markers)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return marker_poses

def perspective_correction(im, show=False) -> ndarray:
    """
    Perspective correction using detected markers to align the image of the chess board.
    """
    markers = detect_markers(im, show=show)
    size = (1000, 1000)

    corners_dst = np.array(
        [[0, 0], [size[0] - 1, 0], [size[0] - 1, size[1] - 1], [0, size[1] - 1]],
        dtype=np.float32,
    )

    corners = np.array(
        [
            markers["marker_1"][0][3],
            markers["marker_2"][0][2],
            markers["marker_3"][0][1],
            markers["marker_4"][0][0],
        ],
        dtype=np.float32,
    )

    flat_im = cv2.warpPerspective(
        im, cv2.getPerspectiveTransform(corners, corners_dst), size
    )

    if show:
        cv2.imshow("Corrected", flat_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return flat_im

def board_size_from_checkers(im, verbose=False, show=False) -> tuple:
    """
    Determines the size of the chess board in terms of number of squares.
    This is done by using a checkerboard pattern detection algorithm to identify
    the corners of the squares on the chessboard, and then counting the number
    of squares along each dimension.
    """
    # Use OpenCV's checkerboard detection
    ret, corners = cv2.findChessboardCorners(im, (3, 7), None)

    # display corners for debugging
    if ret:
        im_corners = cv2.drawChessboardCorners(im.copy(), (3, 7), corners, ret)

        if show:
            cv2.imshow("Checkerboard Corners", im_corners)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Calculate square size
        square_size_x = np.linalg.norm(corners[0] - corners[1])

        if verbose:
            print(corners)
            print(f"Estimated square size: {square_size_x:.1f} pixels")
            print(f"Estimated board size: {board_size:.1f} pixels")

        board_size = 8 * square_size_x

        return board_size, square_size_x
    else:
        print("Checkerboard corners not detected.")
        return None

def crop_img(im, size) -> ndarray:
    """
    Crops the image to just the board area by
    finding the bounding box of the chessboard
    in the flattened image.
    """
    cropped_img = im[
        int(0.5 * im.shape[0] - size // 2) : int(
            0.5 * im.shape[0] + size // 2
        ),
        int(0.5 * im.shape[1] - size // 2) : int(
            0.5 * im.shape[1] + size // 2
        ),
    ]

    return cropped_img

def noise_scores(im, square_size, verbose=False, plot=False, folder=False) -> ndarray:
    """
    generate noise score for each
    square by looking at
    the variance of pixel values
    """
    if len(im.shape) == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    board_dict = board_info(square_size, verbose=verbose)
    noise_scores = []

    for square, info in board_dict.items():
        x_l = int(info["x_lower"])
        x_u = int(info["x_upper"])
        y_l = int(info["y_lower"])
        y_u = int(info["y_upper"])

        # Add inset to squares as edge misalignments may produce false positives
        inset = 5

        square_img = im[y_l+inset:y_u-inset, x_l+inset:x_u-inset]
        noise_score = np.var(square_img)
        board_dict[square]["noise_score"] = noise_score
        noise_scores.append(noise_score)


    # Determine median noise scores 
    # for each type of piece occlusion
    X = np.array(noise_scores).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

    labels = kmeans.labels_
    centers = sorted(kmeans.cluster_centers_.flatten())
    noise_threshold_black = centers[0]
    noise_threshold_white = (centers[0] + centers[1]) / 2


    for square in chess.SQUARE_NAMES:
        board_dict[square]["occupied"] = (
            board_dict[square]["noise_score"] > (noise_threshold_white if (board_dict[square]["color"] == 'white') else noise_threshold_black)
        )

    if plot:
        # Histogram plot
        plt.figure(figsize=(6,4))
        plt.hist(noise_scores, bins=20)
        plt.title("Noise Score Histogram")
        plt.xlabel("Variance")
        plt.ylabel("Frequency")
        if folder:
            plt.savefig(os.path.join(folder, 'Noise_score_hist.png'))

        # K-Means plot
        plt.figure(figsize=(8, 2))
        plt.scatter(noise_scores, np.zeros_like(noise_scores),
                    c=labels, cmap='viridis', s=80)
        plt.scatter(centers, [0]*len(centers),
                    c='red', marker='x', s=200, linewidths=3)
        plt.yticks([])
        plt.xlabel("Noise Score (Variance)")
        plt.title("KMeans Clustering of Square Noise Scores")
        if folder:
            plt.savefig(os.path.join(folder, 'Noise_score_kmeans.png'))

    if verbose:
        print(f"Centers: {centers}")
        print("Noise scores for each square:")
        for square, score in zip(chess.SQUARES, noise_scores):
            print(f"{square}: {score:.2f} \n")
        print("Square dict: \n")
        print(f"{board_dict}")

    
    # Overlay green cast on top of occupied squares
    overlay = im.copy()


    for square, info in board_dict.items():
        x_l = int(info["x_lower"])
        x_u = int(info["x_upper"])
        y_l = int(info["y_lower"])
        y_u = int(info["y_upper"])

        # Color occupied squares
        if info["occupied"]:
            overlay[y_l:y_u, x_l:x_u] = (0, 255, 0)

        # ---- ADD TEXT LABEL ----
        text = info["color"]  # "white" or "black"

        # Position text in center of square
        center_x = int((x_l + x_u) / 2)
        center_y = int((y_l + y_u) / 2)

        cv2.putText(
            overlay,
            text,
            (center_x - 20, center_y + 5),  # adjust offset if needed
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,  # font scale
            (0, 0, 255),  # red text
            1,
            cv2.LINE_AA,
        )

    # Blend overlay with original image
    alpha = 0.3 
    overlayed = cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0, im)
    return board_dict, overlayed

