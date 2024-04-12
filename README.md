# Benchmark tasks and datasets

This repository contains different benchmark tasks and datasets along with a minimal code to generate, import and plot the same data.

## 1. Duffing oscillator response analysis (DORA) prediction task:

The DORA task is to predict the response analysis of a forced Duffing oscillator using a minimal training dataset. This task tests the generalization capabilities of the machine-learning model by extrapolating the system response in unseen parameter regimes i.e amplitude of external periodic forcing in this case. The best working model should be able to qualitatively capture the system response for instance, the exact number of cycles when the system is in a limit-cycle regime or chaotic trajectories for the amplitude values shifting the system in a chaotic regime by training the model only on limited datasets.

<p align="center">
<img src="https://github.com/maneesh51/Benchmark-Tasks/blob/bb41fa278823815ca984b40db618be6f6e0459e3/DORA_3.png">
</p>

### 1.1 Data Loading
 

### 1.2 Description of files
The training (DORA_Train.csv) and testing (DORA_Test.csv) data for the DORA task can be generated, plotted and saved by running the DORA_generator.py file from the command (or Anaconda) prompt. User can give the evaluation time (default 250) and choose to plot (0:No, 1:Yes(Default)) the data using the command:

```python DORA_generator.py -time 250 -plots 1```

The training and testing data for the DORA task can also be generated, loaded and plotted with the DORA.ipynb file. Alternatively, the training and testing data for the DORA task can be simply loaded and plotted with the ReadData.py file.


### 1.3 Description of data
The Train and Test data files have 5 columns in total. The first column represents the time, 2nd and 3rd columns consist of the time evolution of the Duffing oscillator's position and velocity given by: $q1(t)$ and $q2(t)$ respectively. 4th column contains the time evolution of external periodic forcing and 5th column has its amplitude. The train set contains data for two external forcing amplitudes, $f\in[0.46,0.49]$ and the test set consists for a total of five forcing amplitudes, $f\in[0.2,0.35,0.48,0.58,0.75]$. 

### 1.4 Model Evaluation
The success of the prediction model will depend on the extrapolation of system behavior outside the external forcing used for training. The system response characteristics for the external forcing is quantified in terms of amplitude and mean of the $q1^{2}(t)$, which can be obtained using a provided function `Signal_Characteristic`. The prediction performance can be quantified in terms of average and maximum vibration amplitude Mean Squared Error (MSE) in the steady-state time ($t*=20s$):

a) Response Amplitude Error = MSE[ Max(  $q1_{prediction}^{2}(t>t*)$  ), Max( $q1_{original}^{2}(t>t*))$ ]
                  
b) Response Mean Error = MSE[ Mean( $q1_{prediction}^{2}(t>t*)$ ), Mean( $q1_{original}^{2}(t>t*))$ ]
