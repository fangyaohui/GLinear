# GLinear: A Linear Predictor for Multivariate Time Series Forecasting
This repository is the official Pytorch implementation of GLinear Predictor: "Paper Link".

## Description
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

###Dataset
All processed datasets are in the ./dataset directory

### Use Following Command to Train Linear Predictors for Varying LookBack Windows: 
```bash scripts/EXP-LookBackWindow_\&_LongForecasting/Linear_LookBackWindow.sh```

### Use Following Command to Train Linear Predictors for Varying Prediction Lengths: 
```bash scripts/EXP-LookBackWindow_\&_LongForecasting/Linear_LongForecasting.sh```






Training Example
In scripts/ , we provide the model implementation Dlinear/Autoformer/Informer/Transformer
In FEDformer/scripts/, we provide the FEDformer implementation
In Pyraformer/scripts/, we provide the Pyraformer implementation
For example:

To train the LTSF-Linear on Exchange-Rate dataset, you can use the script scripts/EXP-LongForecasting/Linear/exchange_rate.sh:

sh scripts/EXP-LongForecasting/Linear/exchange_rate.sh
It will start to train DLinear by default, the results will be shown in logs/LongForecasting. You can specify the name of the model in the script. (Linear, DLinear, NLinear)

All scripts about using LTSF-Linear on long forecasting task is in scripts/EXP-LongForecasting/Linear/, you can run them in a similar way. The default look-back window in scripts is 336, LTSF-Linear generally achieves better results with longer look-back window as dicussed in the paper.

Scripts about look-back window size and long forecasting of FEDformer and Pyraformer are in FEDformer/scripts and Pyraformer/scripts, respectively. To run them, you need to first cd FEDformer or cd Pyraformer. Then, you can use sh to run them in a similar way. Logs will be stored in logs/.

Each experiment in scripts/EXP-LongForecasting/Linear/ takes 5min-20min. For other Transformer scripts, since we put all related experiments in one script file, directly running them will take 8 hours per day. You can keep the experiments you are interested in and comment on the others.
