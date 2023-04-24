# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import cv2
import math
from ale_python_interface import ALEInterface

class GameManager:
    def __init__(self, game_name, display):
        self.game_name = game_name
        self.display = display

        self.ale = ALEInterface()
        self.ale.setInt(b'random_seed', 123)
        self.ale.setInt(b'frame_skip', 6)
        if self.display:
            self.ale.setBool(b'display_screen', True)
        self.ale.loadROM(b'pong.bin')
        # self.env = gym.make(game_name)
        self.reset()

    def reset(self):
        # observation = self.env.reset()
        self.hit_ball_flag = 0
        self.ball_x1 = 0
        self.ball_y1 = 0
        self.ale.reset_game()
        action = 0
        self.ale.act(action)
        (screen_width,screen_height) = self.ale.getScreenDims()
        screen_data_gray = np.zeros(screen_width*screen_height,dtype=np.uint8)
        self.ale.getScreenGrayscale(screen_data_gray)
        observation = np.reshape(screen_data_gray, (210,160))
        observation = observation[:][34:194]
        ret, observation = cv2.threshold(observation,100,255,cv2.THRESH_BINARY)
        # observation = np.reshape(observation, (160, 160))
        return observation

    def step(self, action, hit_ball_flag_old, ball_x1, ball_y1):
        # self._update_display()
        # observation, reward, done, info = self.env.step(action)
        action_set = [0,3,4]
        action = action_set[action]
        reward = self.ale.act(action)
        # self.ale.act(action)
        done = self.ale.game_over()

        if reward != 0:
            self.hit_ball_flag = 0

        (screen_width,screen_height) = self.ale.getScreenDims()
        screen_data_gray = np.zeros(screen_width*screen_height,dtype=np.uint8)
        self.ale.getScreenGrayscale(screen_data_gray)
        observation = np.reshape(screen_data_gray, (210,160))
        observation = observation[:][34:194]
        ret, observation = cv2.threshold(observation,100,255,cv2.THRESH_BINARY)

        reward_hit_ball = 0
        reward_big_angle = 0
        if reward == -1:
            reward_hit_ball = -1
        if hit_ball_flag_old == 0 and np.sum(observation[:,135:140]) != 0:
            self.hit_ball_flag = 1
        if hit_ball_flag_old == 1 and np.sum(observation[:,135:140]) != 0:
            self.ball_x1 = np.argwhere(observation[:,135:140] != 0)[0,1] + 135
            self.ball_y1 = np.argwhere(observation[:,135:140] != 0)[0,0]
        if hit_ball_flag_old == 1 and np.sum(observation[:,124:134]) != 0:
            self.hit_ball_flag = 0
            reward_hit_ball = 1             ## hit ball

            self.ball_x2 = np.argwhere(observation[:,124:134] != 0)[0,1] + 124
            self.ball_y2 = np.argwhere(observation[:,124:134] != 0)[0,0]
            Hypotenuse = np.sqrt(np.square(self.ball_x2 - ball_x1) + np.square(self.ball_y2 - ball_y1))
            side = ball_x1 - self.ball_x2
            angle = math.acos(side / Hypotenuse) / math.pi * 180
            if angle < 20:
                reward_big_angle = -1       #big angle
            else:
                reward_big_angle = 1

        return observation, reward, reward_hit_ball, reward_big_angle, done, self.hit_ball_flag, self.ball_x1, self.ball_y1
