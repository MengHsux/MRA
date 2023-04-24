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

from multiprocessing import Queue, Value, Lock
from multiprocessing.managers import SyncManager

import time
from collections import deque

from Config import Config
from Environment import Environment
from NetworkVP_fusion_gate import NetworkVP_fusion_gate
from NetworkVP_hit_ball import NetworkVP_hit_ball
from NetworkVP_big_angle import NetworkVP_big_angle
from ProcessAgent import ProcessAgent
from ProcessStats import ProcessStats
from ThreadDynamicAdjustment import ThreadDynamicAdjustment
from ThreadPredictor import ThreadPredictor
from ThreadTrainer import ThreadTrainer
from ThreadQL_Trainer import ThreadQL_Trainer

class Server:
    def __init__(self):
        self.stats = ProcessStats()

        self.training_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.prediction_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        # for the PG update step.
        self.v_prediction_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        # for the QL update step.
        self.Q_value_prediction_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)

        self.model = NetworkVP_fusion_gate(Config.DEVICE, Config.NETWORK_NAME, Environment().get_num_actions())
        self.model_hit_ball = NetworkVP_hit_ball(Config.DEVICE, Config.NETWORK_NAME_HIT, Environment().get_num_actions())
        self.model_big_angle = NetworkVP_big_angle(Config.DEVICE, Config.NETWORK_NAME_BIG, Environment().get_num_actions())
        self.model_hit_ball.load()
        self.model_big_angle.load()
        if Config.LOAD_CHECKPOINT:
            self.stats.episode_count.value = self.model.load()

        self.training_step = 0
        self.frame_counter = 0

        self.agents = []
        self.predictors = []
        self.trainers = []
        self.QL_trainers = []
        self.dynamic_adjustment = ThreadDynamicAdjustment(self)

    def add_agent(self, QL_training_q, QL_training_q_size, lock):
        self.agents.append(
            ProcessAgent(len(self.agents), self.prediction_q, self.training_q, QL_training_q, QL_training_q_size,
                         self.stats.episode_log_q, lock))
        self.agents[-1].start()

    def remove_agent(self):
        self.agents[-1].exit_flag.value = True
        self.agents[-1].join()
        self.agents.pop()

    def add_predictor(self):
        self.predictors.append(ThreadPredictor(self, len(self.predictors)))
        self.predictors[-1].start()

    def remove_predictor(self):
        self.predictors[-1].exit_flag = True
        self.predictors[-1].join()
        self.predictors.pop()

    def add_trainer(self):
        self.trainers.append(ThreadTrainer(self, len(self.trainers)))
        self.trainers[-1].start()

    def remove_trainer(self):
        self.trainers[-1].exit_flag = True
        self.trainers[-1].join()
        self.trainers.pop()

    def add_QL_trainer(self, QL_training_q, QL_training_q_size):
        self.QL_trainers.append(ThreadQL_Trainer(self, len(self.QL_trainers), QL_training_q,
                                                 QL_training_q_size))
        self.QL_trainers[-1].start()

    def remove_QL_trainer(self):
        self.QL_trainers[-1].exit_flag = True
        self.QL_trainers[-1].join()
        self.QL_trainers.pop()

    def train_model(self, x_, r_, a_, v_, rho_, gate_, A_hit_, A_big_, trainer_id):
        self.model.train(x_, r_, a_, v_, rho_, gate_, A_hit_, A_big_, trainer_id)
        self.training_step += 1
        self.frame_counter += x_.shape[0]

        self.stats.training_count.value += 1
        self.dynamic_adjustment.temporal_training_count += 1

        # if Config.TENSORBOARD and self.stats.training_count.value % Config.TENSORBOARD_UPDATE_FREQUENCY == 0:
        #     self.model.log(x_, a_)

    def save_model(self):
        self.model.save(self.stats.episode_count.value)

    def main(self):
        self.stats.start()
        # self.dynamic_adjustment.start()

        # Make deque multiprocessing friendly by using SyncManager
        SyncManager.register('deque', deque)
        m = SyncManager()
        m.start()
        QL_training_q = m.deque(maxlen=Config.MAX_BUFFER_SIZE)
        # Initialize lock to keep an accurate counter for the buffer size
        lock = Lock()
        QL_training_q_size = Value('i', 0)
        # Since ThreadDynamicAdjustment class is gone, we must add all processes
        # and agents manually
        for _ in range(Config.AGENTS):
            self.add_agent(QL_training_q, QL_training_q_size, lock)
        for _ in range(Config.PREDICTORS):
            self.add_predictor()
        for _ in range(Config.PG_TRAINERS):
            self.add_trainer()
        for _ in range(Config.QL_TRAINERS):
            self.add_QL_trainer(QL_training_q, QL_training_q_size)

        if Config.PLAY_MODE:
            for trainer in self.trainers:
                trainer.enabled = False

        learning_rate_multiplier = (
                                       Config.LEARNING_RATE_END - Config.LEARNING_RATE_START) / Config.ANNEALING_EPISODE_COUNT
        beta_multiplier = (Config.BETA_END - Config.BETA_START) / Config.ANNEALING_EPISODE_COUNT

        while self.stats.episode_count.value < Config.EPISODES:
            step = min(self.stats.episode_count.value, Config.ANNEALING_EPISODE_COUNT - 1)
            self.model.learning_rate = Config.LEARNING_RATE_START + learning_rate_multiplier * step
            self.model.beta = Config.BETA_START + beta_multiplier * step

            # Saving is async - even if we start saving at a given episode, we may save the model at a later episode
            if Config.SAVE_MODELS and self.stats.should_save_model.value > 0:
                self.save_model()
                self.stats.should_save_model.value = 0

            time.sleep(0.01)

        # Shut down everything
        m.shutdown()
        self.dynamic_adjustment.exit_flag = True
        while self.agents:
            self.remove_agent()
        while self.predictors:
            self.remove_predictor()
        while self.trainers:
            self.remove_trainer()
