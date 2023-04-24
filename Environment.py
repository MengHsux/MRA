import sys
if sys.version_info >= (3,0):
    from queue import Queue
else:
    from Queue import Queue

import numpy as np
import scipy.misc as misc

import time

from Config import Config
from GameManager import GameManager

class Environment:
    def __init__(self):
        self.game = GameManager(Config.ATARI_GAME, display=Config.PLAY_MODE)
        # self.game = GameManager(Config.ATARI_GAME, display=True)
        self.nb_frames = Config.STACKED_FRAMES
        self.frame_q = Queue(maxsize=self.nb_frames)
        self.previous_state = None
        self.current_state = None
        self.total_reward = 0

        self.reset()

    # @staticmethod
    # def _rgb2gray(rgb):
    #     return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    @staticmethod
    def _preprocess(image):
        # image = Environment._rgb2gray(image)
        # image = misc.imresize(image, [Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH], 'bilinear')
        image = image.astype(np.float32) / 255
        return image

    def _get_current_state(self):
        if not self.frame_q.full():
            return None  # frame queue is not full yet.
        x_ = np.array(self.frame_q.queue)
        x_ = np.transpose(x_, [1, 2, 0])  # move channels
        return x_

    def _update_frame_q(self, frame):
        if self.frame_q.full():
            self.frame_q.get()
        image = Environment._preprocess(frame)
        self.frame_q.put(image)

    def get_num_actions(self):
        return 3

    def reset(self):
        self.total_reward = 0
        self.frame_q.queue.clear()
        self._update_frame_q(self.game.reset())
        self.previous_state = self.current_state = None

    def step(self, action, hit_ball_flag, ball_x1, ball_y1):
        observation, reward, reward_hit_ball, reward_big_angle, done, hit_ball_flag, ball_x1, ball_y1 = self.game.step(action, hit_ball_flag, ball_x1, ball_y1)

        # time.sleep( 0.3 )

        self.total_reward += reward
        self._update_frame_q(observation)

        self.previous_state = self.current_state
        self.current_state = self._get_current_state()
        return reward, reward_hit_ball, reward_big_angle, done, hit_ball_flag, ball_x1, ball_y1
