# Benchmark tasks and datasets

This repository contains different benchmark tasks and datasets along with a minimal code to generate, import and plot the same data.

## 1. Duffing oscillator response analysis (DORA) prediction task:

The DORA task is to generate the response analysis of a forced Duffing oscillator using a minimal training dataset. This task tests the generalization capabilities of the machine-learning model by extrapolating the system response in unseen parameter regimes i.e frequency in this case. The best working model should be able to qualitatively capture the system response for instance, of the exact number of cycles when the system is in a limit-cycle regime or chaotic trajectories for the frequencies shifting the system in a chaotic regime.  

<p align="center">
<img src="https://github.com/maneesh51/Benchmark-Tasks/blob/bb41fa278823815ca984b40db618be6f6e0459e3/DORA_3.png">
</p>

### 1.1 Description of files
The training (DORA_Train.csv) and testing (DORA_Test.csv) data for the DORA task can be generated and saved by running the DORA_generator.py file from the command (or Anaconda) prompt. User can give the evaluation time (default 250) and choose to plot (0:No, 1:Yes(Default)) the data using the command:

```python DORA_generator.py -time 250 -plots 1```

The training and testing data for the DORA task can also be generated, loaded and plotted with the DORA.py or DORA.ipynb files.

### 1.2 Description of data
The Train and Test data files have 4 columns in total. The first two columns consist of time evolution of the Duffing oscillator's position and velocity given by: $q1(t)$ and $q2(t)$. 3rd column contains the time evolution of external periodic forcing and 4th column has it amplitude ($f(amp)$).
