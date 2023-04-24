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
from multiprocessing import Queue
from Config import Config


class ThreadTrainer(Thread):
    def __init__(self, server, id):
        super(ThreadTrainer, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False
        self.v_wait_q1 = Queue(maxsize=1)
        self.v_wait_q2 = Queue(maxsize=1)

    def run(self):
        while not self.exit_flag:
            batch_size = 0
            while batch_size <= Config.TRAINING_MIN_BATCH_SIZE:
                x_, r_, a_, v_, rho_, gate_, A_hit_, A_big_ = self.server.training_q.get()
                if batch_size == 0:
                    x__ = x_; r__ = r_; a__ = a_; gate__ = gate_; A_hit__ = A_hit_; A_big__ = A_big_
                    v__ = v_; rho__ = rho_
                    # x__ = x_; r__ = r_; a__ = a_
                else:
                    x__ = np.concatenate((x__, x_))
                    r__ = np.concatenate((r__, r_))
                    a__ = np.concatenate((a__, a_))
                    gate__ = np.concatenate((gate__, gate_))
                    A_hit__ = np.concatenate((A_hit__, A_hit_))
                    A_big__ = np.concatenate((A_big__, A_big_))
                    v__ = np.concatenate((v__, v_))
                    rho__ = np.concatenate((rho__, rho_))
                batch_size += x_.shape[0]

                self.server.v_prediction_q.put((self.id, x__))
                baselines1 = self.v_wait_q1.get()
                baselines2 = self.v_wait_q2.get()
                # Subtract by the PG baselines to get PG advantages
                A_hit__ = A_hit__ - baselines1
                # Subtract by the PG baselines to get PG advantages
                A_big__ = A_big__ - baselines2
                
            if Config.TRAIN_MODELS:
                self.server.train_model(x__, r__, a__, v__, rho__, gate__, A_hit__, A_big__, self.id)
                # self.server.train_model(x__, r__, a__, self.id)
