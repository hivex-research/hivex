import uuid
from typing import Tuple, List, Mapping
from collections import defaultdict
import uuid

# RLlib
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# ML-Agents
from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage

HumanCommand = Mapping[str, str]


class HumanCommandChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6b3-784f4387d1f7"))
        self.command = ""

    def on_message_received(self, msg: IncomingMessage):
        text = msg.read_string()
        print(f"Received message from Unity: {text}")
        self.command = text

    def send_string(self, message: str):
        msg = OutgoingMessage()
        msg.write_string(message)
        self.queue_message_to_send(msg)

    def get_and_reset_commands(self) -> str:
        """
        Returns the current commands, and resets the internal storage of the commands.

        :return:
        """
        s = self.command
        self.command = ""
        return s
