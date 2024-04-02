# GLinear: A Linear Predictor for Multivariate Time Series Forecasting
This repository is the official Pytorch implementation of GLinear Predictor: "Paper Link".

## Description
This code is built on the code base of LTSF-Linear Predictors. We would like to thanks the following GitHub repo for its valuable code bases, datasets and detailed description:

[https://github.com/cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)

Required Environment can be installed using instructions given in above link. 

Furthermore, the pytorch implementaion of RLinear predictor used for comparative analysis is avaible at:
[https://github.com/plumprc/RTSF/blob/main/models/RLinear.py](https://github.com/plumprc/RTSF/blob/main/models/RLinear.py)

## Training the GLinear and Other Predictors:

### Use Following Command to Train Linear Predictors for Varying LookBack Windows: 
```bash scripts/EXP-LookBackWindow_\&_LongForecasting/Linear_LookBackWindow.sh```

### Use Following Command to Train Linear Predictors for Varying Prediction Lengths: 
```bash scripts/EXP-LookBackWindow_\&_LongForecasting/Linear_LongForecasting.sh```

