# Ensemble-Deep-Reinforcement-Learning-in-Wind-Speed-Prediction

### 1. Introduction
In this project, we applied three types of supervised deep learning models to predict wind speed in north Taiwan, and then we applied Deep Reinforcement Learning(DRL) to optimize predictions. 
### 2. Highlight
* Compared to the traditional ensemble model, we applied DRL to optimize the prediction
* We get R2 score over 0.97 in several stations.
* We decomposed wind speed signals by Empirical wavelet transform.
### 3. FlowChart
The figure is the flowchart of the project.
Procedure: 
#### 3-1. I change the time frequency of datasets from hourly to daily dataset, and then I replace missing values with an interpolate function. 
#### 3-2. I use Empirical Wavelet Transform (EWT) to decompose data. I made two experiments for testing the performance in the once decomposition and twice decomposition (Based on the decomposed signal from the first decomposing).  Both methods will decompose 100 sub-signals for each feature. 
#### 3-3. After decomposing the signal, I split the whole dataset into training, validation, and testing sets (0.64:0.16:0.2) 
#### 3-4. I set each sample to contain N days of data. It means the model can get past N days' information to predict the next day's wind speed.
#### 3-5. Training, validating, and testing by simple supervised models: I use three basic supervised DL models: 
* 1D-CNN-GRU model (1 conv1d layers(kernel_size =2); 1 avgpooling layers,1 gru layer; 3 dense layers)
* LSTM model(2 lstm layers; 3 dense layers) 
* 1D-CNN-LSTM model(1 conv1d layers(kernel_size =2); 1 avgpooling layers,1 lstm layer; 3 dense layers)
I use mse as the loss function and adam as the optimizer, learning rate = 0.0005, epochs = 40. It should be noted that I use the R2 score(sklearn.metrics.r2_score) to evaluate model performance, therefore, the below results are R2 score.

#### 3-6. Training, validating, and testing ensemble DRL:  The DRL agent is for optimizing prediction based on simple models’ prediction. It distributes weight to each prediction of supervised models. I get input data from three trained models, and then I train, validate, and test. Before training the DRL agent, I apply min-max normalization to normalize the prediction of supervised models. 

### 4. Main Packages
* Gym
* Stable baselines3
* Tensorflow
* Pandas
* Numpy
* scikit-learn
* ewtpy
### 5. Results
I use two hourly datasets (Place: Keelung and Banqiao) as examples. 
### 6. Conclusion 
In this project, I applied DRL in wind speed prediction. Compared to the most common use of DRL, we input a dataset to control observation in each iteration, not producing observation by the environment. The observation of the environment is the output of supervised DL models. For this reason, I should design classes to make codes clear and smooth, not just use packages to finish the project. Moreover, this is my first time using wavelet transform to decompose wind speed signals. Compared to my past experience, datasets have several features for prediction. I seldom meet a situation in which I need to use a decomposed method. Therefore, I knew decomposing methods are very powerful if we only have a single data signal.
Overall, the experience is interesting to me.
