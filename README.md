# Benchmark tasks and datasets

This repository contains different benchmark tasks and datasets along with a minimal code to generate, import and plot the same data.

## 1. Duffing oscillator response analysis (DORA) prediction task:

The DORA task is to generate the response analysis of a forced Duffing oscillator using a minimal training dataset. This task tests the generalization capabilities of the machine-learning model by extrapolating the system response in unseen parameter regimes i.e amplitude of external periodic forcing in this case. The best working model should be able to qualitatively capture the system response for instance, the exact number of cycles when the system is in a limit-cycle regime or chaotic trajectories for the amplitude values shifting the system in a chaotic regime by training the model only on a limited datasets.

<p align="center">
<img src="https://github.com/maneesh51/Benchmark-Tasks/blob/bb41fa278823815ca984b40db618be6f6e0459e3/DORA_3.png">
</p>

### 1.1 Description of files
The training (DORA_Train.csv) and testing (DORA_Test.csv) data for the DORA task can be generated and saved by running the DORA_generator.py file from the command (or Anaconda) prompt. User can give the evaluation time (default 250) and choose to plot (0:No, 1:Yes(Default)) the data using the command:

```python DORA_generator.py -time 250 -plots 1```

The training and testing data for the DORA task can also be generated, loaded and plotted with the DORA.py or DORA.ipynb files.

### 1.2 Description of data
The Train and Test data files have 4 columns in total. The first two columns consist of time evolution of the Duffing oscillator's position and velocity given by: $q1(t)$ and $q2(t)$. 3rd column contains the time evolution of external periodic forcing and 4th column has its amplitude. The train set contains data for two external forcing amplitudes, $f\in[0.46,0.49]$ and the test set consists that for a total of five forcing amplitudes, $f\in[0.2,0.35,0.48,0.58,0.75]$. 
