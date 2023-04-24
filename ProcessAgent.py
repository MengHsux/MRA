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

from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time
import csv
import random

from Config import Config
from Environment import Environment
from Experience import Experience


class ProcessAgent(Process):
    def __init__(self, id, prediction_q, training_q, QL_training_q, QL_training_q_size, episode_log_q, lock):
        super(ProcessAgent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.QL_training_q = QL_training_q
        self.QL_training_q_size = QL_training_q_size
        self.training_q = training_q
        self.episode_log_q = episode_log_q
        self.lock = lock

        self.env = Environment()
        self.num_actions = self.env.get_num_actions()
        self.actions = np.arange(self.num_actions)
        self.weight = np.arange(3)
        self.gate = np.zeros(3).astype(np.float32)

        self.discount_factor = Config.DISCOUNT
        # one frame at a time
        self.wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)

    def _accumulate_rewards(self, experiences, discount_factor, terminal_reward, terminal_reward_hit,
                            terminal_reward_big):
        reward_sum = terminal_reward
        reward_sum_hit = terminal_reward_hit
        reward_sum_big = terminal_reward_big
        for t in reversed(range(0, len(experiences) - 1)):
            r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.REWARD_MAX)
            reward_sum = discount_factor * reward_sum + r
            experiences[t].reward = reward_sum

            r_hit = np.clip(experiences[t].reward_hit_ball, Config.REWARD_MIN, Config.REWARD_MAX)
            r_big = np.clip(experiences[t].reward_big_angle, Config.REWARD_MIN, Config.REWARD_MAX)
            reward_sum_hit = discount_factor * reward_sum_hit + r_hit
            reward_sum_big = discount_factor * reward_sum_big + r_big

            experiences[t].reward_hit_ball = (reward_sum_hit - (0.6122365479)) / 1.4315440129
            experiences[t].reward_big_angle = (reward_sum_big - (0.1634638122)) / 0.2670975099
        return experiences[:-1]

    def convert_data(self, experiences):
        pis = [exp.prediction for exp in experiences]
        values = [exp.value for exp in experiences]
        # assign now values to exp.value
        for idx, value in enumerate(values):
            experiences[idx].value = value
        if experiences[-1].done:
            terminal_v = 0
        else:
            terminal_v = experiences[-1].value
            exps = experiences[:len(experiences) - 1]
        ## compute rho
        ratios = [pis[id][experiences[id].action] / experiences[id].miu for id in range(len(experiences))]

        rho_ = [min(Config.RHO, rho) for rho in ratios]
        # rho_ = [1. for rho in ratios]
        c_ = [min(Config.C, rho) for rho in ratios]
        # c_ = [1. for rho in ratios]

        delta_vs = []
        for idx, exp in enumerate(experiences):
            if idx == len(experiences) - 1:
                delta_vs.append(rho_[idx] * (exp.reward + Config.DISCOUNT * terminal_v - exp.value))
            else:
                delta_vs.append(rho_[idx] * (exp.reward + Config.DISCOUNT * experiences[idx + 1].value - exp.value))

        vs = []
        for idx in reversed(range(0, len(experiences))):
            if idx == len(experiences) - 1:
                v = experiences[idx].value + delta_vs[idx]
                experiences[idx].reward_sum = experiences[idx].reward + Config.DISCOUNT * terminal_v
            else:
                v = experiences[idx].value + delta_vs[idx] + Config.DISCOUNT * c_[idx] * (
                        vs[-1] - experiences[idx + 1].value)
                experiences[idx].reward_sum = experiences[idx].reward + Config.DISCOUNT * vs[-1]
            vs.append(v)
            experiences[idx].v = v
            experiences[idx].rho = rho_[idx]

        x_ = np.array([exp.state for exp in experiences])
        r_ = np.array([exp.reward for exp in experiences])
        a_ = np.eye(self.num_actions)[np.array([exp.action for exp in experiences])].astype(np.float32)
        gate_ = np.eye(3)[np.array([exp.gate for exp in experiences])].astype(np.float32)
        A_hit = np.array([exp.reward_hit_ball for exp in experiences])
        A_big = np.array([exp.reward_big_angle for exp in experiences])
        v = np.array([exp.value for exp in experiences])
        rho = np.array([exp.rho for exp in experiences])
        return x_, r_, a_, gate_, A_hit, A_big, v, rho

    def predict(self, state):
        # put the state in the prediction q
        self.prediction_q.put((self.id, state))
        # wait for the prediction to come back
        p, p1, p2, gate_p, v, v_hit, v_big = self.wait_q.get()
        return p, p1, p2, gate_p, v, v_hit, v_big

    def select_gate(self, gate_p):
        if Config.PLAY_MODE:
            # action = np.random.choice(self.actions, p=prediction)
            gate = np.argmax(gate_p)
        else:
            gate = np.random.choice(self.weight, p=gate_p)
        return gate

    def select_action(self, prediction):
        if Config.PLAY_MODE:
            # action = np.random.choice(self.actions, p=prediction)
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.actions, p=prediction)
        return action

    def run_episode(self):
        self.env.reset()
        done = False
        experiences = []

        time_count = 0
        reward_sum = 0.0

        hit_ball_flag = 0
        ball_x1 = 0
        ball_y1 = 0

        while not done:
            # very first few frames
            if self.env.current_state is None:
                self.env.step(0, hit_ball_flag, ball_x1, ball_y1)  # 0 == NOOP
                continue

            prediction, prediction1, prediction2, gate_p, value, v_hit, v_big = self.predict(self.env.current_state)
            action = self.select_action(prediction)
            action1 = self.select_action(prediction1)
            action2 = self.select_action(prediction2)

            gate = self.select_gate(gate_p)
            if gate == 0:
                action = action
            if gate == 1:
                action = action1
            if gate == 2:
                action = action2

            reward, reward_hit_ball, reward_big_angle, done, hit_ball_flag, ball_x1, ball_y1 = self.env.step(action, hit_ball_flag, ball_x1, ball_y1)
            reward_sum += reward

            exp = Experience(self.env.previous_state, action, prediction, prediction1, prediction2, gate, gate_p, reward, reward_hit_ball, reward_big_angle, v_hit, v_big, self.env.current_state, done, value)
            experiences.append(exp)

            if done or time_count == Config.TIME_MAX:
                terminal_reward = 0 if done else value

                terminal_reward_hit = 0 if done else v_hit
                terminal_reward_big = 0 if done else v_big

                updated_exps = self._accumulate_rewards(experiences, self.discount_factor, terminal_reward, terminal_reward_hit, terminal_reward_big)
                x_, r_, a_, gate_, A_hit, A_big, v, rho = self.convert_data(updated_exps)
                yield experiences, x_, r_, a_, v, rho, gate_, A_hit, A_big, reward_sum

                # reset the tmax count
                time_count = 0
                # keep the last experience for the next batch
                experiences = [experiences[-1]]
                reward_sum = 0.0

            time_count += 1

    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        while self.exit_flag.value == 0:
            total_reward = 0
            total_length = 0
            for experiences, x_, r_, a_, v, rho, gate_, A_hit, A_big, reward_sum in self.run_episode():
                total_reward += reward_sum
                total_length += len(a_) + 1  # +1 for last frame that we drop
                self.training_q.put((x_, r_, a_, v, rho, gate_, A_hit, A_big))
                # Also add experiences to the QL replay buffer
                for exp in experiences:
                    self.QL_training_q.append(exp)
                    # Sleeping helps with updating the counter
                    time.sleep(0.0001)
                    # Use locks to keep an accurate counter for the buffer size
                    with self.lock:
                        self.QL_training_q_size.value += 1
            self.episode_log_q.put((datetime.now(), total_reward, total_length))