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

# Drift Detection [Inference environment]

# This notebook demonstrates how to use ADL API to perform drift detection.
# The data is fed into the drift detector using a files in a folder in this
# standalone environment. In real world applications the stream data can be fed
# into the drift detector using data pipeline
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
# introduced to demonstrate drift

# Import Drift Detection feature from HPE ADL
from hpeai.adl.features import DriftDetection

import time
import os
import pandas as pd

# Specify Inputs to Drift Detection API
# You can specify data in chunks for drift detection on a daily basis as an
# example
baseline = '/home/ai/adl/baselines/hpe_adl_baseline.zip'
source_data = '/home/ai/adl/source_data/'
output_data = '/home/ai/adl/output_data'
batch_size = 50

# Example: For drift detection on a daily basis across 6 days
days_data = ["day1", "day2", "day3", "day4", "day5", "day6"]

# Initialize Drift Detection module
dd = DriftDetection(baseline=baseline, output_data=output_data)

# Start Drift Detection
# Loop over the days to demonstrate streaming environment and drift over a
# period. Method will read the image files from source_data, batch them and
# invoke drift detection. The detection output and drifted images are stored in
# output_data.
#
# In real world, applications can pass source data using pipeline to drift
# detection module. The output can be sent to a pipeline on drift detection

# Note: To view drift detection results at runtime, use the alert callback
# functionality as described later in this notebook.
# Wait till the drift detection task gets completed using the sample snippet
# below because the drift detection runs asynchronously:
for day in days_data:
    source_folder = os.path.join(source_data, day)
    dd.start(batch_size=batch_size, source_data=source_folder)

    while(dd.get_task_running_status()):
        time.sleep(1)
    time.sleep(10)


# Display Absolute drift
# Absolute drift is the total drift against the baseline across batches
# Drift detected column indicates YES (detected) NO (not detected)
absolute_drift_output = pd.read_csv(os.path.join(output_data,
                                                 'absolute_drift_results.csv'))
print(absolute_drift_output)


# Display Relative drift
# Relative drift is the drift between successive batches of data
relative_drift_output = pd.read_csv(os.path.join(output_data,
                                                 'relative_drift_results.csv'))
print(relative_drift_output)


# Display Drifted images
# You can choose to display images and charts for anomalous images leveraging
# the code snippet below:
'''
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Range1d
from bokeh.io import output_notebook
from PIL import Image
import matplotlib.pyplot as plt
import math


def draw_drift_chart(data_repr_file_path) -> None:
    df = pd.read_csv(data_repr_file_path)
    x_range = range(0, len(df["Baseline"].to_list()))
    TOOLTIPS = [("Time:", "@z"), ("Loss:", "@y")]
    tools = ["pan", "box_zoom", "reset", "hover", "save"]
    p = figure(title="Data Representation - Images",
               x_axis_label='Images Index',
               y_axis_label='Representation value',
               tooltips=TOOLTIPS, tools=tools)
    source1 = ColumnDataSource(data=dict(y=df["Baseline"], z=x_range))
    source2 = ColumnDataSource(data=dict(y=df["Batch"], z=x_range))
    p.xaxis.major_label_orientation = 1
    p.y_range = Range1d(0, 1)
    p.line(x="z", y="y", color="red", line_width=2, source=source1)
    p.scatter(x="z", y="y", color="green", line_width=2, source=source2)
    output_notebook()
    show(p)


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


display_images(os.path.join(output_data, 'anomalies', 'images'))

# Display Absolute drift chart on a per batch basis.
draw_drift_chart("BaselinevsBatch.csv")
'''


# User can optionally define a callback function as below to track progress
# If status_log = None, Baseline derivation completed without any failure.
def user_callback(status_log=None):
    if status_log is None:
        print("Drift Detection Completed")
    else:
        print(status_log)


# User registers a callback function
dd.alert(user_callback)

# Stop Drift Detection
# To stop the drift detection while it is in progress, User can invoke stop()
# function.
dd.stop()
