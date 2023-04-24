from threading import Thread
import numpy as np
from multiprocessing import Queue
import random

from Config import Config
# This is a new class for my PGQ: algorithm not found in the original GA3C
class ThreadQL_Trainer(Thread):
    def __init__(self, server, id, QL_training_q, QL_training_q_size):
        super(ThreadQL_Trainer, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.QL_training_q = QL_training_q
        self.QL_training_q_size = QL_training_q_size
        self.Q_value_wait_q1 = Queue(maxsize=1)
        self.Q_value_wait_q2 = Queue(maxsize=1)
        self.exit_flag = False

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
        a_ = np.eye(self.server.model.num_actions)[np.array([exp.action for exp in experiences])].astype(np.float32)
        gate_ = np.eye(3)[np.array([exp.gate for exp in experiences])].astype(np.float32)
        A_hit = np.array([exp.reward_hit_ball for exp in experiences])
        A_big = np.array([exp.reward_big_angle for exp in experiences])
        v = np.array([exp.value for exp in experiences])
        rho = np.array([exp.rho for exp in experiences])
        curs = np.array([exp.cur_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])
        return x_, r_, a_, gate_, A_hit, A_big, v, rho, curs, dones

    def run(self):
        while not self.exit_flag:
            if Config.TRAIN_MODELS:
                # Don't do QL update step until there's a minimum number of experiences
                # in the QL replay buffer
                if self.QL_training_q_size.value < Config.MIN_BUFFER_SIZE or self.server.model.get_global_step() % 20 != 0:
                    continue
                # Randomly sample a small batch of experiences
                experiences = random.sample(self.QL_training_q._getvalue(), Config.QL_BATCH_SIZE)
                prevs, rewards, actions, gate_, A_hit, A_big, v, rho, curs, dones = self.convert_data(experiences)
                # These are the QL baselines
                self.server.Q_value_prediction_q.put((self.id, prevs))
                prev_Q1 = np.sum(self.Q_value_wait_q1.get() * actions, axis=1)
                prev_Q2 = np.sum(self.Q_value_wait_q2.get() * actions, axis=1)
                # These are the maximum Q-values for the current state
                self.server.Q_value_prediction_q.put((self.id, curs))
                cur_Q1 = np.max(self.Q_value_wait_q1.get(), axis=1)
                cur_Q2 = np.max(self.Q_value_wait_q2.get(), axis=1)
                advantages1 = np.zeros(cur_Q1.shape[0])
                advantages2 = np.zeros(cur_Q2.shape[0])

                gate__ = gate_
                A_hit__ = A_hit
                A_big__ = A_big
                v__ = v
                rho__ = rho

                # We calculate QL advantages here
                for i in range(cur_Q1.shape[0]):
                    advantages1[i] = A_hit__[i] + Config.DISCOUNT * cur_Q1[i] - prev_Q1[i] if dones[i] is False else \
                    A_hit__[i]
                # We calculate QL advantages here
                for i in range(cur_Q2.shape[0]):
                    advantages2[i] = A_big__[i] + Config.DISCOUNT * cur_Q2[i] - prev_Q2[i] if dones[i] is False else \
                    A_big__[i]
                self.server.train_model(prevs, rewards, actions, v__, rho__, gate__, advantages1, advantages2, self.id)
