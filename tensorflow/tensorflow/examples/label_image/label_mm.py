# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import sys
import os.path
import glob
import numpy as np
import tensorflow as tf
import gc
import json
import matplotlib.pyplot as plt

from tensorflow.python.platform import gfile
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=bool)
    parser.add_argument('--summaries_dir',
                        type=str,
                        default='/mnt/storage/home/bf15007/Manta_Inception/tensorflow/retrain_logs',
                        help='Where to save summary logs for TensorBoard.'
                        )
    args = parser.parse_args()

    def top_k_graph(final_results, ground_truth, evaluated_images):
        with tf.Session() as sess:

            accuracy = tf.placeholder(tf.float32)
            top_k_summary = tf.summary.scalar('Top_K', accuracy)

            top_k_writer = tf.summary.FileWriter(args.summaries_dir + '/mm_top_k'.format(sess.graph))
            sess.run(tf.global_variables_initializer())
            for k in range(1, 100):
                prediction = 0
                for results, truth in zip(final_results, ground_truth):
                    for j in range(0, k):
                        if results[j] == truth:
                            prediction += 1
                accuracy_str = sess.run(top_k_summary, {accuracy: prediction / evaluated_images})
                top_k_writer.add_summary(accuracy_str, k)
                top_k_writer.flush()
                print('top {} is {}% correct from {} images'.format(k, (prediction / evaluated_images)*100, evaluated_images))

    if args.graph:
        pass
    with open("mm_results.json") as f:
        data = json.load(f)

        ground_truth = []
        final_results = []
        for i in range(0, 198):
            id = str(data[i]["individualId"])
            ground_truth.append(id)
            top_k = []
            for j in range(0, 99):
                top_k.append(str(data[i]["scores"]["ids"][j]))
            final_results.append(top_k)
        final_results = np.reshape(final_results, [198, 99])
        #print(ground_truth)
        #print(final_results[0])
        top_k_graph(final_results, ground_truth, 198)
