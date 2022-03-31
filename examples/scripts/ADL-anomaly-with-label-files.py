#!/usr/bin/env python
# coding: utf-8

############################################################################
# Copyright 2022 Hewlett Packard Enterprise Development LP
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
############################################################################


# HPE AI Observability: Anomaly and Drift detection Library (ADL)
# Anomaly Detection [Inference environment]

# This notebook demonstrates how to use ADL API to perform anomaly detection.
# The data is fed into the anomaly detector using files in a folder in this
# standalone environment. In real world applications the stream data can be fed
# into the anomaly detector using data pipeline
#
# Dataset used in this notebook as an example: The NIH Chest X-ray dataset
# consists of chest x-ray images provided by the NIH Clinical Center and is
# available through the NIH download site:
# https://nihcc.app.box.com/v/ChestXray-NIHCC
#
# We have used COVID19 X-Ray images from IEEE8023/Covid Chest X-Ray Dataset:
# https://github.com/ieee8023/covid-chestxray-dataset
#
# The initial model as well as ADL is trained with NIH images. Covid images are
# introduced as anomalies.

# Import Anomaly Detection feature from HPE ADL
from hpeai.adl.features import AnomalyDetection

import time
import pandas as pd
import os

# Specify Inputs to Anomaly Detection API
baseline = '/home/ai/adl/baselines/hpe_adl_baseline.zip'
source_data = '/home/ai/adl/source_data/'
label_file = '/home/ai/adl/labels/images.csv'
output_data = '/home/ai/adl/output_data'

batch_size = 10

# Initialize Anomaly Detection module
ad = AnomalyDetection(baseline=baseline, source_data=source_data,
                      label_file=label_file, output_data=output_data)


# Start Anomaly Detection
# Method will read the image files from source_data, batch them and invoke
# anomaly detection. The detection output and anomolous images are stored in
# output_data
#
# In real world, applications can pass source data using pipeline to anomaly
# detection module. The output can be sent to a pipeline on anomaly detection
ad.start(batch_size=batch_size)

# Note: To view detected anomalous images at runtime, use the alert callback
# functionality as described later in this notebook.
# Wait till the anomaly detection task gets completed using the sample snippet
# below because the anomaly detection runs asynchronously:
while(ad.get_running_task_status()):
    time.sleep(1)


# Display detected anomalies
# Results are logged in anomaly_output.csv
# Anomaly column indicates YES (detected) NO (not detected)
pd.set_option("max_rows", None)
anomaly_output = pd.read_csv(os.path.join(output_data, 'anomalies',
                                          'anomaly_output.csv'))
print(anomaly_output)


# Display detected Anomalous images
# You can choose to display images and charts for anomalous images leveraging
# the code snippet below:
'''
from PIL import Image
import matplotlib.pyplot as plt
import math

def display_images(folder_path):
    show_images(folder_path)
    pass


def show_images(folder_path, image_types=['jpeg', 'jpg', 'png']):
    fig = plt.figure(figsize=(50, 20))

    for root, _, files in os.walk(folder_path):
        file_paths = []
        for img_type in image_types:
            file_paths += get_images_by_type(root, files, img_type)

        num_files = len(file_paths)
        num_rows, num_cols = get_number_rows_and_columns(num_files)

        for i in range(num_files):
            sub = fig.add_subplot(num_rows, num_cols, i + 1)
            with open(file_paths[i], 'rb') as f:
                image = Image.open(f)
                sub.imshow(image)
    plt.show()


def display_cluster_images(folder_path):
    show_images(folder_path, image_types=['png'])
    pass


def get_number_rows_and_columns(num_files):
    num_cols = int(math.sqrt(num_files))
    num_rows = num_files // num_cols

    if num_rows * num_cols < num_files:
        num_rows += 1

    return num_rows, num_cols


def get_images_by_type(root, files, image_type='jpeg'):
    file_paths = []
    for file in files:
        file_path = os.path.join(root, file)

        # Get the filename only from the initial file path.
        filename = os.path.basename(file_path)

        # Use splitext() to get filename and extension separately.
        (_, ext) = os.path.splitext(filename)

        if ext == '.' + image_type:
            file_paths.append(file_path)
    return file_paths


# get_ipython().run_line_magic('matplotlib', 'inline')
display_images(os.path.join(output_data, 'anomalies', 'images'))


# The section below demonstrates APIs to get additional information for
# verification and debugging.
# Display Cluster plots for each batch.
display_cluster_images(os.path.join(output_data, 'clusters'))
'''


# Optionally define a callback function to monitor results
# For integrating anomaly detection into ML pipeline, use this function to
# define the callback action

# If status_log = None, Baseline derivation completed without any failure.
def user_callback(status_log=None):
    if status_log is None:
        print("Anomaly Detection Completed")
    else:
        print('USER CALLBACK: ', status_log)


# User registers a callback function
ad.alert(user_callback)


# Stop Anomaly Detection
# To stop the anomaly detection while it is in progress, User can invoke stop()
# function.
ad.stop()
