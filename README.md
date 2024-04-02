# GLinear: A Linear Predictor for Multivariate Time Series Forecasting
This repository is the official Pytorch implementation of GLinear Predictor: "Paper Link".

## Description

The architecture of GLinear predictor is composed of two fully-connected layers of same input size having a Gaussian Error Linear Unit (GeLU) nonlinearity in-between them. Different configurations of input and layers sizes are tested to lead to this final architecture. 

<p align="center">
  <img src="/Extra/Glinear.png" alt="GitHub Logo" width="200"/>
</p>

The **Reversible Instance Normalization (RevIN)**  is applied to the input and output layer of GLinear model. The normalization layer transforms the original data distribution into a mean-centred distribution, where the distribution discrepancy between different instances is reduced. This normalized data is then applied as new input to the used predictor and then final output is denormalized at last step to provide the final prediction.

1. RevIN Normalization:
2. Linear transformation:
3. GELU activation:
4. Second Linear transformation:
5. RevIN Denormalization:




## Acknowledgment
This code is built on the code base of LTSF-Linear Predictors (**NLinear**, **DLinear**). We would like to thank the following GitHub repository for their valuable code bases, datasets and detailed descriptions:

[https://github.com/cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)

Required Environment and libraries can be installed using instructions given in above link. 

Furthermore, we would also like to thank the pytorch implementation of **RLinear** predictor used for comparative analysis, available at:
[https://github.com/plumprc/RTSF/blob/main/models/RLinear.py](https://github.com/plumprc/RTSF/blob/main/models/RLinear.py)

## Training the GLinear and Other Predictors:

First, create the conda environment and install the required tools and libraries, as follows:

```
conda create -n GLinear python=3.6.9
conda activate GLinear
pip install -r requirements.txt
```

### Datasets
All processed datasets are in the ./dataset directory

### Use Following Command to Train Linear Predictors for Varying LookBack Windows: 
```bash scripts/EXP-LookBackWindow_\&_LongForecasting/Linear_LookBackWindow.sh```

### Use Following Command to Train Linear Predictors for Varying Prediction Lengths: 
```bash scripts/EXP-LookBackWindow_\&_LongForecasting/Linear_LongForecasting.sh```

### Results
All the results will be shown in ./logs directory

## Citing
If you use or find this code repository useful, consider citing it as follows:
```
S.Tahir.H.Rizvi and N. Kanwal
```





