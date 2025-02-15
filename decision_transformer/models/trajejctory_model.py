import numpy as np
import tensorflow as tf


class TrajectoryModel(tf.keras.Model):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    # def call(self, states, actions, rewards, masks=None, attention_mask=None):
    #     # "masked" tokens or unspecified inputs can be passed in as None
    #     return None, None, None

    # def get_action(self, states, actions, rewards, **kwargs):
    #     # these will come as tensors on the correct device
    #     return torch.zeros_like(actions[-1])
