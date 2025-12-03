import codi
from codi import ColossusClient, GuiClient
import numpy
import torch


client = ColossusClient(
    host="192.168.0.10",
    video_port=8001,
    command_port=8002,
    states_port=8003
)
