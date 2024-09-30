# **MLTiming: Machine-Learning framework for radiation time pick-up**

![Example_image](https://github.com/wjosefw/Signal-processing-with-Neural-networks/blob/main/scheme_2%20(1).png)

An easy-to-use, explainable machine learning tool to train individual gamma radiation detectors to give accurate time-stamps to incoming radiation signals. In our framework we utilize the signals of a given detector to train it to return accurate time-stamps without the need of auxiliary or reference detectors to aid the training process. We achieve this by using as training data pairs of pulses and their copies delayed by a known factor, the ML tool is then optimized to be able to reproduce the known time difference between each pulse and its delayed copy. In other words, with MLTiming you can calibrate detectors independently in a fast and reliable fashion enabeling the fast processing and timestamping of realistic influxes of gamma radiation.
