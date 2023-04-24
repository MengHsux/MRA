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

from threading import Thread

import numpy as np

from Config import Config


class ThreadPredictor(Thread):
    def __init__(self, server, id):
        super(ThreadPredictor, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False

    def run(self):
        ids = np.zeros(Config.PREDICTION_BATCH_SIZE, dtype=np.uint16)
        states = np.zeros(
            (Config.PREDICTION_BATCH_SIZE, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES),
            dtype=np.float32)

        while not self.exit_flag:
            ids[0], states[0] = self.server.prediction_q.get()

            size = 1
            while size < Config.PREDICTION_BATCH_SIZE and not self.server.prediction_q.empty():
                ids[size], states[size] = self.server.prediction_q.get()
                size += 1

            batch = states[:size]
            # p, v = self.server.model.predict_p_and_v(batch)
            p = self.server.model.predict_p(batch)
            p1 = self.server.model.predict_p1(batch)
            p2 = self.server.model.predict_p2(batch)

            gate_p = self.server.model.predict_gate_p(batch)

            v = self.server.model.predict_v(batch)

            v_hit = self.server.model_hit_ball.predict_v(batch)
            v_big = self.server.model_big_angle.predict_v(batch)

            for i in range(size):
                if ids[i] < len(self.server.agents):
                    self.server.agents[ids[i]].wait_q.put((p[i], p1[i], p2[i], gate_p[i], v[i], v_hit[i], v_big[i]))
                    # self.server.agents[ids[i]].wait_q.put((p[i], v[i]))

            # These are for calculating QL advantages
            Q_value_size = 0
            while Q_value_size < Config.PREDICTION_BATCH_SIZE and not self.server.Q_value_prediction_q.empty():
                Q_value_id, Q_value_states = self.server.Q_value_prediction_q.get()
                Q1 = self.server.model.predict_Q_value1(Q_value_states)
                Q2 = self.server.model.predict_Q_value2(Q_value_states)

                if Q_value_id < len(self.server.QL_trainers):
                    self.server.QL_trainers[Q_value_id].Q_value_wait_q1.put(Q1)
                    self.server.QL_trainers[Q_value_id].Q_value_wait_q2.put(Q2)
                Q_value_size += 1
            # These are for calculating PG advantages
            v_size = 0
            while v_size < Config.PREDICTION_BATCH_SIZE and not self.server.v_prediction_q.empty():
                v_id, v_states = self.server.v_prediction_q.get()
                v1 = self.server.model.predict_v1(v_states)
                v2 = self.server.model.predict_v2(v_states)

                if v_id < len(self.server.trainers):
                    self.server.trainers[v_id].v_wait_q1.put(v1)
                    self.server.trainers[v_id].v_wait_q2.put(v2)
                v_size += 1