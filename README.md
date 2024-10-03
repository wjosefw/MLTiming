# **MLTiming: A Machine-Learning Framework for Radiation Time Pick-Up**

![Example Image](https://github.com/wjosefw/Signal-processing-with-Neural-networks/blob/main/scheme_2%20(1).png)

**MLTiming** is a user-friendly machine-learning tool designed to train individual gamma radiation detectors for precise time-stamping of incoming radiation signals. This innovative framework eliminates the need for auxiliary or reference detectors during the training process, simplifying calibration.

**Repository Developer:**  
José Avellaneda, PhD Candidate, Nuclear Physics Group, Complutense University of Madrid.  
For inquiries, contact: [javellan@ucm.es](mailto:javellan@ucm.es)

---

## **Overview**

**MLTiming** leverages the signals from a specific detector to independently produce precise time-stamps. The system is trained using pairs of radiation pulses and their corresponding delayed copies, separated by a known time interval. This machine-learning approach optimizes the model to reproduce the time difference accurately, facilitating precise detector calibration without external reference devices.

---

## **How It Works**

### Step 0: Prepare Data
- **Input**: Waveforms should be converted to a numpy array with shape `(Nev, Nt)`, where `Nev` is the number of detected events and `Nt` is the number of time points in the waveform.
  
### Step 1: Crop the Waveforms
- **Cropping**: For this study, the waveforms are cropped between 10 ns and 14.8 ns of the coincidence window. Future applications should modify the cropping window according to the specifics of the detection setup (e.g., crystal, electronics, event energy, etc.).

### Step 2: Create Virtual Coincidences
- **Delay Pairs**: Use the `create_and_delay_pulse_pair` function, which requires the size of the time step and the maximum number of delay steps. This function creates pairs of pulses with known time delays for model training.
### Step 3: Define Model
- **Select Inputs**: We provide a range of functions and models that utilize waveforms as data inputs or compute features to optimize input size. For feature extraction, please use the `momentos` function, which requires the virtual coincidence array and the desired number of features to calculate.
- **Choose a Model**: For initial testing, three distinct network architectures are available: a Multi-Layer Perceptron (MLP), a Convolutional Neural Network (CNN), and a KAN model. You can find these architectures in the `models.py` file.
### Step 4: Train Models
- **Execute Training Loop**: Leverage the predefined training loops to train the networks. This process will yield results for both the validation and testing sets at each epoch, allowing you to monitor the model's performance over time.
---

## **Key Features**

- **Independent Calibration**: Train individual detectors without the need for auxiliary or reference detectors, simplifying the calibration process.
- **Fast and Accurate**: Achieve high-speed processing and precise time-stamping for real-world gamma radiation detection.
- **Explainable AI**: Provides transparency into the machine-learning model’s decision-making process, enhancing interpretability for researchers.

---

## **Benefits**

With **MLTiming**, researchers and practitioners can efficiently calibrate gamma radiation detectors, ensuring peak performance in real-world scenarios. The key benefits include:

- **Simplified Workflow**: No need for additional equipment or complex calibration setups.
- **Scalability**: Adaptable to different types of detectors, detection setups, and radiation environments.
- **Versatility**: Applicable across a range of detection systems, enhancing performance in both laboratory and field settings.

---

## **Installation and Setup**

1. Clone this repository:
   ```bash
   git clone https://github.com/wjosefw/Signal-processing-with-Neural-networks.git
2. After activating the virtual environment, you can install specific package requirements as follows: 
    ```bash
    pip install -r requirements.txt
