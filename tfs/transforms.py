from sre_constants import SUCCESS, FAILURE
from time import time
import chess
import numpy as np
import codi


def TF(pose):
    pos_vec = pose[0]
    rot_vec = pose[1]

    # Creat a 4x4 transformation matrix from position and rotation vectors
    TF = np.zeros((4, 4))
    C_x = np.cos(rot_vec[0])  # Cos x
    C_y = np.cos(rot_vec[1])  # Cos y
    C_z = np.cos(rot_vec[2])  # Cos z
    S_x = np.sin(rot_vec[0])  # Sin x
    S_y = np.sin(rot_vec[1])  # Sin y
    S_z = np.sin(rot_vec[2])  # Sin z

    TF = np.array(
        [
            [C_y * C_z, -C_y * S_z, S_y, pos_vec[0]],
            [
                C_x * S_z + C_z * S_x * S_y,
                C_x * C_z - S_x * S_y * S_z,
                -C_y * S_x,
                pos_vec[1],
            ],
            [
                S_x * S_z - C_x * C_z * S_y,
                C_z * S_x + C_x * S_y * S_z,
                C_y * C_x,
                pos_vec[2],
            ],
            [0, 0, 0, 1],
        ]
    )

    return TF


def get_vec_from_TF(TF):
    pos = TF[0:3, 3]
    rot = np.zeros(3)
    rot[0] = np.arctan2(TF[2, 1], TF[2, 2])  # rx
    rot[1] = np.arctan2(-TF[2, 0], np.sqrt(TF[2, 1] ** 2 + TF[2, 2] ** 2))  # ry
    rot[2] = np.arctan2(TF[1, 0], TF[0, 0])  # rz
    return pos, rot


def sq_to_board_coord(square):
    coord = [square // 8, square % 8, 0]  # x,y,z relative to board origin
    return coord


async def move_to_square(
    target_sq, approach_pose, board, robot_client, print_status=False
):
    # convert move to board coordinates
    sq_coords = sq_to_board_coord(target_sq)
    pos = sq_coords * board.cell_size
    rot = [
        0,
        0,
        0,
    ]  # board (and target square) rotation sq_r_b is zero (rotation wrt the body fixed frame is zero)
    piece_pose = np.array([pos, rot])

    # Compute square to board transformation matrix
    square_T_board = TF(pos, rot)

    # get board transformation matrix relative to robot base
    board_T_world = board.transformation_matrix

    # Compute square to world transformation matrix
    square_T_world = np.matmul(square_T_board, board_T_world)

    # Move robot to the target pose at standby height
    target_pos, target_rot = get_vec_from_TF(square_T_world)
    target = np.concatenate((target_pos, target_rot))
    await robot_client.send_command(
        target, space="TS", rt=False, interface_type="position", verbose=False
    )

    # Block until move is complete
    while robot_client.move_status == "moving":

        if print_status:
            print(robot_client.move_status)

        time.sleep(0.1)

        if robot_client.move_status == "at_target":
            return SUCCESS

        elif robot_client.move_status == "error":
            return FAILURE
