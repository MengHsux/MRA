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

import os
import re
import numpy as np
import tensorflow as tf

from Config import Config


class NetworkVP_fusion_gate:
    def __init__(self, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

        self.img_width = Config.IMAGE_WIDTH
        self.img_height = Config.IMAGE_HEIGHT
        self.img_channels = Config.STACKED_FRAMES

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                if Config.TENSORBOARD: self._create_tensor_board()
                if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                    vars = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)
                

    def _create_graph(self):
        self.x = tf.placeholder(
            tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='X')
        self.y_r = tf.placeholder(tf.float32, [None], name='Yr')
        
        self.A_hit = tf.placeholder(tf.float32, [None], name='Ahit')
        self.A_big = tf.placeholder(tf.float32, [None], name='Abig')
        self.rho = tf.placeholder(tf.float32, [None], name='rho')
        self.v = tf.placeholder(tf.float32, [None], name='value')

        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        self.global_step = tf.Variable(0, trainable=False, name='step')

        # As implemented in A3C paper
        self.m1 = self.maxpool_layer(self.x, 'maxpool1')
        self.n1 = self.conv2d_layer(self.m1, 8, 16, 'conv11', strides=[1, 4, 4, 1])
        # self.n1 = self.conv2d_layer(self.x, 8, 16, 'conv11', strides=[1, 4, 4, 1])
        self.n2 = self.conv2d_layer(self.n1, 4, 32, 'conv12', strides=[1, 2, 2, 1])
        self.action_index = tf.placeholder(tf.float32, [None, self.num_actions])
        self.gate_index = tf.placeholder(tf.float32, [None, 3])
        _input = self.n2

        flatten_input_shape = _input.get_shape()
        nb_elements = flatten_input_shape[1] * flatten_input_shape[2] * flatten_input_shape[3]

        self.flat = tf.reshape(_input, shape=[-1, nb_elements._value])
        self.d1 = self.dense_layer(self.flat, 256, 'dense1')

        self.logits_v = tf.squeeze(self.dense_layer(self.d1, 1, 'logits_v', func=None), axis=[1])
        self.cost_v_gate = 0.5 * tf.reduce_sum(tf.square(self.y_r - self.logits_v), axis=0)

        self.logits_v1 = tf.squeeze(self.dense_layer(self.d1, 1, 'logits_v1', func=None), axis=[1])
        self.logits_v2 = tf.squeeze(self.dense_layer(self.d1, 1, 'logits_v2', func=None), axis=[1])
        self.cost_v_avg = 0.5 * tf.reduce_sum(
            tf.square((self.A_hit + self.A_big) / 2 - (self.logits_v1 + self.logits_v2) / 2), axis=0)
        self.cost_v_1 = 0.5 * tf.reduce_sum(tf.square(self.A_hit - self.logits_v1), axis=0)
        self.cost_v_2 = 0.5 * tf.reduce_sum(tf.square(self.A_big - self.logits_v2), axis=0)

        self.logits_gate_p = self.dense_layer(self.d1, 3, 'logits_gate_p', func=None)
        self.softmax_gate_p = tf.nn.softmax(self.logits_gate_p)
        self.selected_gate_prob = tf.reduce_sum(self.softmax_gate_p * self.gate_index, axis=1)
        self.cost_gate_p_1 = tf.log(tf.maximum(self.selected_gate_prob, self.log_epsilon)) \
                    * (self.y_r - tf.stop_gradient(self.logits_v))
        self.cost_gate_p_2 = -1 * self.var_beta * \
                    tf.reduce_sum(tf.log(tf.maximum(self.softmax_gate_p, self.log_epsilon)) *
                                    self.softmax_gate_p, axis=1)
        self.cost_gate_p_1_agg = tf.reduce_sum(self.cost_gate_p_1, axis=0)
        self.cost_gate_p_2_agg = tf.reduce_sum(self.cost_gate_p_2, axis=0)
        self.cost_gate_p = -(self.cost_gate_p_1_agg + self.cost_gate_p_2_agg)

        self.logits_p = self.dense_layer(self.d1, self.num_actions, 'logits_p', func=None)
        self.softmax_p = (tf.nn.softmax(self.logits_p) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)
        self.selected_action_prob = tf.reduce_sum(self.softmax_p * self.action_index, axis=1)
        self.cost_p_1 = self.rho * tf.log(tf.maximum(self.selected_action_prob, self.log_epsilon)) \
                         * ((self.A_hit + self.A_big) / 2 - tf.stop_gradient((self.logits_v1 + self.logits_v2) / 2))  ### average
        self.cost_p_2 = -1 * self.var_beta * \
                    tf.reduce_sum(tf.log(tf.maximum(self.softmax_p, self.log_epsilon)) *
                                    self.softmax_p, axis=1)
        # use a mask and entropy since we pad bactches of size < TIME_MAX
        mask = tf.reduce_max(self.action_index, axis=1)
        self.cost_p_1_agg = tf.reduce_sum(self.cost_p_1 * mask, axis=0)
        self.cost_p_2_agg = tf.reduce_sum(self.cost_p_2 * mask, axis=0)
        self.cost_p = -(self.cost_p_1_agg + self.cost_p_2_agg)

        log_policy = tf.log(tf.clip_by_value(self.softmax_p, Config.SMALL_VALUE, 1.0))
        entropy = - tf.reduce_sum(self.softmax_p * log_policy)

        self.logits_p1 = self.dense_layer(self.d1, self.num_actions, 'logits_p1', func=None)
        self.softmax_p1 = (tf.nn.softmax(self.logits_p1) + Config.MIN_POLICY) / (
                1.0 + Config.MIN_POLICY * self.num_actions)
        self.selected_action_prob1 = tf.reduce_sum(self.softmax_p1 * self.action_index, axis=1)
        self.cost_p_11 = tf.log(tf.maximum(self.selected_action_prob1, self.log_epsilon)) \
                         * (self.A_hit - tf.stop_gradient(self.logits_v1))  ### average
        self.cost_p_21 = -1 * self.var_beta * \
                         tf.reduce_sum(tf.log(tf.maximum(self.softmax_p1, self.log_epsilon)) *
                                       self.softmax_p1, axis=1)
        self.cost_p_1_agg1 = tf.reduce_sum(self.cost_p_11, axis=0)
        self.cost_p_2_agg1 = tf.reduce_sum(self.cost_p_21, axis=0)
        self.cost_p1 = -(self.cost_p_1_agg1 + self.cost_p_2_agg1)

        log_policy1 = tf.log(tf.clip_by_value(self.softmax_p1, Config.SMALL_VALUE, 1.0))
        entropy1 = - tf.reduce_sum(self.softmax_p1 * log_policy1)

        # The state-action values
        self.Q_values1 = self.var_beta * (tf.log(tf.maximum(self.softmax_p1, self.log_epsilon))) + \
                         tf.expand_dims(self.cost_p_21, 1) + tf.expand_dims(self.logits_v1, 1)

        self.logits_p2 = self.dense_layer(self.d1, self.num_actions, 'logits_p2', func=None)
        self.softmax_p2 = (tf.nn.softmax(self.logits_p2) + Config.MIN_POLICY) / (
                1.0 + Config.MIN_POLICY * self.num_actions)
        self.selected_action_prob2 = tf.reduce_sum(self.softmax_p2 * self.action_index, axis=1)
        self.cost_p_12 = tf.log(tf.maximum(self.selected_action_prob2, self.log_epsilon)) \
                         * (self.A_big - tf.stop_gradient(self.logits_v2))  ### average
        self.cost_p_22 = -1 * self.var_beta * \
                         tf.reduce_sum(tf.log(tf.maximum(self.softmax_p2, self.log_epsilon)) *
                                       self.softmax_p2, axis=1)
        self.cost_p_1_agg2 = tf.reduce_sum(self.cost_p_12, axis=0)
        self.cost_p_2_agg2 = tf.reduce_sum(self.cost_p_22, axis=0)
        self.cost_p2 = -(self.cost_p_1_agg2 + self.cost_p_2_agg2)

        log_policy2 = tf.log(tf.clip_by_value(self.softmax_p2, Config.SMALL_VALUE, 1.0))
        entropy2 = - tf.reduce_sum(self.softmax_p2 * log_policy2)

        # The state-action values
        self.Q_values2 = self.var_beta * (tf.log(tf.maximum(self.softmax_p2, self.log_epsilon))) + \
                         tf.expand_dims(self.cost_p_22, 1) + tf.expand_dims(self.logits_v2, 1)

        self.cost_all = self.cost_p + self.cost_p1 + self.cost_p2 + self.cost_gate_p + self.cost_v_gate + self.cost_v_avg + self.cost_v_1 + self.cost_v_2 - entropy * 0.01 - entropy1 * 0.01 - entropy2 * 0.01
        # self.cost_all = self.cost_p + self.cost_v
        self.opt = tf.train.RMSPropOptimizer(
            learning_rate=self.var_learning_rate,
            decay=Config.RMSPROP_DECAY,
            momentum=Config.RMSPROP_MOMENTUM,
            epsilon=Config.RMSPROP_EPSILON)

        self.train_op = self.opt.minimize(self.cost_all, global_step=self.global_step)


    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("Pcost_advantage", self.cost_p_1_agg))
        summaries.append(tf.summary.scalar("Pcost_entropy", self.cost_p_2_agg))
        summaries.append(tf.summary.scalar("Pcost", self.cost_p))
        summaries.append(tf.summary.scalar("Vcost", self.cost_v))
        summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        summaries.append(tf.summary.scalar("Beta", self.var_beta))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        summaries.append(tf.summary.histogram("activation_n1", self.n1))
        summaries.append(tf.summary.histogram("activation_n2", self.n2))
        summaries.append(tf.summary.histogram("activation_d2", self.d1))
        summaries.append(tf.summary.histogram("activation_p", self.softmax_p))

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def maxpool_layer(self, input, name):
        with tf.variable_scope(name):
            output = tf.nn.max_pool(input, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")
        return output

    def dense_layer(self, input, out_dim, name, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.matmul(input, w) + b
            if func is not None:
                output = func(output)

        return output

    def conv2d_layer(self, input, filter_size, out_dim, name, strides, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w',
                                shape=[filter_size, filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.nn.conv2d(input, w, strides=strides, padding='SAME') + b
            if func is not None:
                output = func(output)

        return output

    def __get_base_feed_dict(self):
        return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate}

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict_single(self, x):
        return self.predict_p(x[None, :])[0]

    def predict_p(self, x):
        prediction = self.sess.run(self.softmax_p, feed_dict={self.x: x})
        return prediction

    def predict_p1(self, x):
        prediction1 = self.sess.run(self.softmax_p1, feed_dict={self.x: x})
        return prediction1

    def predict_p2(self, x):
        prediction2 = self.sess.run(self.softmax_p2, feed_dict={self.x: x})
        return prediction2

    def predict_Q_value1(self, x):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x})
        prediction = self.sess.run(self.Q_values1, feed_dict=feed_dict)
        return prediction

    def predict_Q_value2(self, x):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x})
        prediction = self.sess.run(self.Q_values2, feed_dict=feed_dict)
        return prediction

    def predict_gate_p(self, x):
        gate_p = self.sess.run(self.softmax_gate_p, feed_dict={self.x: x})
        return gate_p
        
    def predict_v(self, x):
        v = self.sess.run(self.logits_v, feed_dict={self.x: x})
        return v

    def predict_v1(self, x):
        prediction = self.sess.run(self.logits_v1, feed_dict={self.x: x})
        return prediction

    def predict_v2(self, x):
        prediction = self.sess.run(self.logits_v2, feed_dict={self.x: x})
        return prediction
    
    def predict_p_and_v(self, x):
        return self.sess.run([self.softmax_p, self.logits_v], feed_dict={self.x: x})
    
    def train(self, x, y_r, a, v, rho, gate, A_hit, A_big, trainer_id):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.y_r: y_r, self.action_index: a, self.v:v, self.rho:rho, self.gate_index: gate, self.A_hit: A_hit, self.A_big: A_big})
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def _checkpoint_filename(self, episode):
        return 'checkpoints/%s_%08d' % (self.model_name, episode)
    
    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[2])

    def save(self, episode):
        self.saver.save(self.sess, self._checkpoint_filename(episode))

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)
        return self._get_episode_from_filename(filename)
       
    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))
