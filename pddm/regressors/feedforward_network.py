# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf
import numpy as np


def feedforward_network(inputStates, inputSize, outputSize, num_fc_layers,
                        depth_fc_layers, tf_datatype, scope):

    with tf.variable_scope(str(scope)):

        #concat K entries together [bs x K x sa] --> [bs x ksa]
        inputState = tf.layers.flatten(inputStates)

        #vars
        intermediate_size = depth_fc_layers
        reuse = False
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=False, seed=None, dtype=tf_datatype)
        fc = tf.contrib.layers.fully_connected
        max_logvar = tf.Variable(np.ones([1, outputSize])/2., trainable=True, dtype=tf.float32, name="max_log_var")
        min_logvar = tf.Variable(-np.ones([1, outputSize])*10., trainable=True, dtype=tf.float32, name="min_log_var")

        # make hidden layers
        for i in range(num_fc_layers):
            if i==0:
                fc_i = fc(
                    inputState,
                    num_outputs=intermediate_size,
                    activation_fn=None,
                    weights_initializer=initializer,
                    biases_initializer=initializer,
                    reuse=reuse,
                    trainable=True)
            else:
                fc_i = fc(
                    h_i,
                    num_outputs=intermediate_size,
                    activation_fn=None,
                    weights_initializer=initializer,
                    biases_initializer=initializer,
                    reuse=reuse,
                    trainable=True)
            h_i = tf.nn.relu(fc_i)

        # make output layer
        z = fc(
            h_i,
            num_outputs=(outputSize - 1) * 2 + 1,
            activation_fn=None,
            weights_initializer=initializer,
            biases_initializer=initializer,
            reuse=reuse,
            trainable=True)
        half = (outputSize - 1)/2
        mean, logvar, catastrophe_prob = z[:, :half], z[:, half:-1], z[:, -1:]
        #out, catastrophe_prob = z[:, :outputSize - 1], z[:, outputSize - 1:]
        logvar = max_logvar - tf.nn.softplus(max_logvar - logvar)
        logvar = min_logvar + tf.nn.softplus(logvar - min_logvar)
    return mean, logvar, max_logvar, min_logvar, catastrophe_prob
    #return z #out, catastrophe_prob
