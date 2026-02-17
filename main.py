import asyncio
import signal
from enum import Enum, auto
import time
import logging
import numpy as np
import chess
from sre_constants import SUCCESS, FAILURE

from codi import CoraClient
from chess.Agent import Agent 
from Board import board
import tfs.transforms as tf

logger = logging.getLogger("robot_chess")
logging.basicConfig(level=logging.INFO)

# TODO: Update to most recent Cora API changes

class CoraChess:
    def __init__(self):
        self.handlers = {}
        self.robot_client = CoraClient()
        self.board = board()
        self.agent = Agent(model=None)  # Load trained model

    ###################
    ### FSM methods ###
    ###################

    #############################
    ### Robot control methods ###
    #############################

    async def pick_and_place(self, move):
        from_sq = board.move.from_square
        to_sq =  move.to_square
        approach_pose = np.array([0.0, 0.0, 0.3],
                                 [0.0, 0.0, 0.0])

        ## approach above piece at height of 30cm
        await tf.move_to_square(
            from_sq, approach_pose, self.board, 
            self.robot_client, print_status=False
        )

        ## move down to height of 5cm
        ready_pose = approach_pose.copy()
        ready_pose[2] = 0.05
        await tf.move_to_square(
            from_sq, 0.05, self.board, 
            self.robot_client, print_status=False
        )

        ## close gripper
        await self.robot_client.send_command(
            np.array(0, 0, 0, 0, 0, 0),
            space="TS",
            rt=False,
            interface_type="velocity",
            print=False,
            gripper_command=np.array(1.0),
        )

        ## move up
        await tf.move_to_square(
            from_sq, 0.3, self.board, self.robot_client, print_status=False
        )

        ## approach above target square at standby height
        await tf.move_to_square(
            to_sq, 0.3, self.board, self.robot_client, print_status=False
        )

        ## move down
        await tf.move_to_square(
            to_sq, 0.05, self.board, self.robot_client, print_status=False
        )

        ## open gripper
        await self.robot_client.send_command(
            np.array([0, 0, 0, 0, 0, 0]),
            space="TS",
            rt=False,
            interface_type="velocity",
            print=False,
            gripper_command=np.array(0.0),
        )

        ## move to standby pose
        result = await self.robot_client.send_command(
            self.standby_pose,
            space="TS",
            rt=False,
            interface_type="position",
            print=True,
        )
        if result == SUCCESS:
            return SUCCESS
        else:
            return FAILURE
