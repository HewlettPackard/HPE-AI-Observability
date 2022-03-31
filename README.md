<h3 align="center">HPE AI Observability</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
![Python version](https://img.shields.io/badge/python-3.8-blue.svg)

</div>

---

<p align="center"> HPE AI Observability comprises an Anomaly and Drift detection library (ADL) which is used for timely detection of anomalies and drift (AD) at production environments by monitoring data and Machine Learning model predictions in AI environments.
  <br>
</p>

## 📝 Table of Contents

- [HPE AI Observability](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Getting in touch](#getting_in_touch)

## 🧐 HPE AI Observability<a name = "about"></a>
<p align="justify">
In Artifical Intelligence (AI) environments, an enormous volume of data is created for real time analytics and deep insights.  Data scientists and engineers  build ML models with labeled data,  arrive at  reasonably generalized models, and deploy the best performing model in production. 
</p>  
<p align="justify">
However, in production AI environments, data streams are not typically stationery and are different from training data. A drift can happen on the input streaming data, or a concept shift can happen altering the relationship between the input and output to the  model. Further, there may be anomalies / outliers in the data that can be a noise or unexpected data points. Data drift and anomalies  influence models predictions leading to model performance decay. 
</p>
<p align="justify">
  Consistent accuracy of ML model catering to data shift dynamics is an important requirement to support increased adoption of ML models. A degrading performance in models due to drift or anomalies can lead to erroneous predictions and will result in negative impact on business outcomes. Hence it is important to continuously evaluate model performance related to data and/or concept drift. 
</p>
<p align="justify">
HPE AI Observability supports timely detection of anomalies and drift at production environments by monitoring data and ML model predictions.This information can be used by customer to enable automated re-training of the ML model on the newly seen data. This ensures consistent quality of model performance by enabling continuous learning.
</p>

**This is a preview version and an ongoing effort. We plan to update this repository progressively.**

**This version of HPE AI Observability is supported on Linux 64-Bit (x86) OS. This version is currently tested with Ubuntu 20.04.**

## 🏁 Getting Started <a name = "getting_started"></a>

HPE AI Observability software kit is distributed as a Python library with APIs exposed to derive baselines and perform Anomaly / Drift detection. Currently this is supported for Image Classification use cases. 
<p align="justify">
Image datasets that have been used for verifying ADL functionalities are: NIH Clinical Center and CIFAR10. These can be downloaded from the following links:

- [NIH Chest X-ray](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) [Note: This link has CIFAR10 python version]
</p>

ADL supports the following functionality:
- Baseline derivation: User generates a baseline package which contains referential datas and a trained ADL detection model at the training environment, that is deployed at inference for anomaly and drift detection.
- Anomaly Detection: User runs anomaly detection at production to detect anomalies in incoming data and obtain alerts.
- Data Drift Detection: User runs drift detection at production to detect data drift and obtain alerts.

The following instructions will help you setup ADL in your local environment. See the installation notes to deploy the bits.

### Prerequisites

Minimum python **3.8** version is required.

### Deployment and Usage

1. Create an HPE Passport account to access [My HPE Software Center (MSC)](https://myenterpriselicense.hpe.com/cwp-ui/product-details/HPE-AI-Observability/0.1.0/sw_free).

2. Download the **HPE_AI_Observability_for_Linux_x64_Q2V41-11032.tar.gz** file.

3. Download the signature file **HPE_AI_Observability_for_Linux_x64_Q2V41-11032.tar.gz.sig**. Verify the file using the link https://myenterpriselicense.hpe.com/cwp-ui/free-software/HPLinuxCodeSigning 

4. Untar **HPE_AI_Observability_for_Linux_x64_Q2V41-11032.tar.gz** as shown below:

```bash
tar -xf HPE_AI_Observability_for_Linux_x64_Q2V41-11032.tar.gz
```

5. Install the HPE AI Observability wheel file:

```python
pip install hpeaiobservability-0.1.0-cp38-cp38-manylinux2014_x86_64.whl
```

This will install HPE AI Observability and all the required dependencies.

Note:
It is **recommended** to create a separate python virtual environment. Using your existing python environment may lead to version conflicts on packages installed. 

**As one approach, find below the steps for setting up a Python virtual environment using Anaconda Platform:**

Install [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) on an [Ubuntu 20.4.2](https://ubuntu.com/download/server) LTS system.

Execute the below commands to create and activate python environment for ADL.
```bash
# Create new environment with Python 3.8
conda create -n adl python=3.8 -y

# Activate the ADL conda environment
conda activate adl
```
### Uninstall
```bash
python -m pip uninstall hpeaiobservability -y
```


## 🎈 Usage: An overview of HPE AI Observability usage is explained below. <a name="usage"></a>
### For API Specification, refer [Wiki](https://github.com/HewlettPackard/HPE-AI-Observability/wiki)

### Configuring the ADL autoencoder architecture.
ADL employs a convolutional autoencoder for anomaly and drift detection features. You can choose to alter the architecture of the autoencoder by specifying parameters in this [ADL autoencoder specification file](https://github.com/HewlettPackard/HPE-AI-Observability/blob/main/examples/notebooks/cae_spec_v1.yaml). Exercise caution while altering this file as it may impact ADL functional performance. At this [link](https://github.com/HewlettPackard/HPE-AI-Observability/blob/main/examples/notebooks/) the following specification files are available for use:
1. Default: [Sample Specification File for NIH Chest X-Ray Dataset](https://github.com/HewlettPackard/HPE-AI-Observability/blob/main/examples/notebooks/cae_spec_v1.yaml) 
2. [Sample Specification File for CIFAR10 Dataset](https://github.com/HewlettPackard/HPE-AI-Observability/blob/main/examples/notebooks/cae_spec_v1_cifar10.yaml)

Note: The datasets used in the example notebooks are experimental only, for demonstrating the ADL functionality.

The default values of the attributes corresponding to the dimensions of an image dataset in the cae_spec_v1.yaml are:
```bash
shape_params:   
    dim1: 128     [This represents Height i.e. image dimension 1]
    dim2: 128     [This represents Width i.e. image dimension 2]
    channels: 1   [This represents Depth; 1 for Grayscale and 3 for RGB]
```

To add the shape of the image for the dataset you are using, modify the  attributes in the yaml file:
```bash
shape_params:
    dim1: <your image dimension 1>
    dim2: <your image dimension 2>
    channels: <1 for Grayscale and 3 for RGB>
```
**Overwrite the cae_spec_v1.yaml file with your altered specification file for it to take effect.**

### Baseline Derivation at Training environment:
<p align="justify">
A baseline package is required to be generated that creates a zip file comprising reference data and a trained convolutional autoenocoder model using ADL Python API. The autoencoder and the reference data enable anomaly and drift detection at inference.
</p>

Refer this [example notebook](https://github.com/HewlettPackard/HPE-AI-Observability/blob/main/examples/notebooks/ADL-baseline.ipynb) for this functionality.

### Anomaly Detection at Inference environment:
<p align="justify">
At inference, production image data is fed to ADL for anomaly detection. Currently, this is through files in a folder as per the API specification. Users can develop their applications using the ADL API for feeding in streaming data.
</p>

Refer this [example notebook](https://github.com/HewlettPackard/HPE-AI-Observability/blob/main/examples/notebooks/ADL-anomaly.ipynb) for this functionality.

### Drift Detection at Inference environment:
<p align="justify">
At inference, production image data is fed to ADL for data drift detection. Currently, this is through files in a folder as per the API specification. Users can develop their applications using the ADL API for feeding in streaming data.

ADL detects the following:
  
- Absolute drift: The total drift against the baseline across multiple batches of data.
- Relative Drift: The drift between successive batches of data.
- Departure: A data shift that is close to the baseline
</p>

Refer this [example notebook](https://github.com/HewlettPackard/HPE-AI-Observability/blob/main/examples/notebooks/ADL-drift.ipynb) for this functionality.

## Getting in touch <a name = "getting_in_touch"></a>
Feedback and questions are appreciated. You can use the issue tracker to report bugs on GitHub.

or

Join the Slack channel [hpe-ai-observability](https://hpe-external.slack.com/archives/C037QD18YMT) to communicate with us.

## Contributing
  Refer to [Contributing](CONTRIBUTING.md) for more information.

## License
  The distribution of HPE AI Observability in this repository is for non-commercial and experimental use under this [license](LICENSE.md). 
  
  See [ATTRIBUTIONS](ATTRIBUTIONS.md) for terms and conditions for using the datasets included in this repository.

© Copyright 2022 Hewlett Packard Enterprise Development LP
