# ...existing code...
import asyncio
import signal
from enum import Enum, auto
import time
import logging
import numpy as np
import chess 
from sre_constants import SUCCESS, FAILURE

import codi
from codi import CoraClient, GuiClient
from chess.chess_cora import Agent, Board
import tfs.transforms as tf

logger = logging.getLogger("robot_chess")
logging.basicConfig(level=logging.INFO)


class State(Enum):
    IDLE = auto()
    WAIT_START = auto()
    SCAN_BOARD = auto()
    PLAN_MOVE = auto()
    EXECUTE_MOVE = auto()
    VERIFY_MOVE = auto()
    WAIT_OPPONENT = auto()
    ERROR = auto()
    SHUTDOWN = auto()


class CaraChess:
    def __init__(self):
        self.handlers = {}
        self.robot_client = CoraClient()
        self.board = Board()
        self.agent = Agent(model=None)  # Load trained model 
        self.standby_pose = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0])  # Example standby pose

    
    ### FSM methods ###
    

    ### Non-FSM methods ###

    async def pick_and_place(self, move):
        from_sq, to_sq = move.from_square, move.to_square

        ## approach above piece at height of 30cm
        await tf.move_to_square(from_sq, 0.3, self.board, self.robot_client, print_status=False)

        ## move down to height of 5cm
        await tf.move_to_square(from_sq, 0.05, self.board, self.robot_client, print_status=False)

        ## close gripper
        await self.robot_client.send_command(np.array(0,0,0,0,0,0), space='TS', rt=False, interface_type='velocity', print=False, gripper_command=np.array(1.0))

        ## move up
        await tf.move_to_square(from_sq, 0.3, self.board, self.robot_client, print_status=False)

        ## approach above target square at standby height
        await tf.move_to_square(to_sq, 0.3, self.board, self.robot_client, print_status=False)

        ## move down
        await tf.move_to_square(to_sq, 0.05, self.board, self.robot_client, print_status=False)

        ## open gripper
        await self.robot_client.send_command(np.array([0,0,0,0,0,0]), space='TS', rt=False, interface_type='velocity', print=False, gripper_command=np.array(0.0))

        ## move to standby pose
        result = await self.robot_client.send_command(self.standby_pose, space='TS', rt=False, interface_type='position',print=True)
        if result == SUCCESS:
            return SUCCESS
        else:
            return FAILURE
        