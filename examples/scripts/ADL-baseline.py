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

# Baseline Derivation [Training environment]

# This notebook contains the user application to  generate a baseline package
# comprising reference data and a trained model (currently an autoencoder)
# using ADL Python API. The user supplies the following information:
# - Path of the source data images
# - Path for storing baseline
#
# Dataset used in this notebook as an example: The NIH Chest X-ray dataset
# consists of chest x-ray images provided by the NIH Clinical Center and is
# available through the NIH download site:
# https://nihcc.app.box.com/v/ChestXray-NIHCC

# Import Baseline Derivation feature from HPE ADL
from hpeai.adl.features import BaselineDerivation

import time

# Provide Inputs to Baseline Derivation API

# Folder containing source images.
source_data = "/home/ai/adl/source_data/"

# Output folder to store the baseline
output_data = "/home/ai/adl/baselines_output"

# Initialize Baseline Derivation Model
# - uses the default auto encoder from ADL
bd = BaselineDerivation(source_data=source_data, output_data=output_data)

# Start Baseline derivation
# Train the Auto Encoder
# Specify epoch count and batch size.
bd.start(epochs=15, batch_size=200)

# Wait till the baseline derivation task gets completed using the sample
# snippet below because the baseline derivation runs asynchronously:
while bd.get_running_task_status():
    time.sleep(1)

# Save derived baseline package [consists of trained AutoEncoder model and
# baseline reference data]
# Run the save API only after completing training
bd.save('hpe_adl_baseline.zip')

# Optionally define a callback function to monitor results
# For integrating baseline derivation into ML pipeline, use this function to
# define the callback action


# If status_log = None, Baseline derivation completed without any failure.
def user_callback(status_log=None):
    if status_log is None:
        print("Baseline Derivation Completed")
    else:
        print(status_log)


# Register the callback function
bd.alert(user_callback)

# Stop Baseline Derivation
# To stop the baseline derivation while it is in progress, User can invoke
# stop() function.
bd.stop()
