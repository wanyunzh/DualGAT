# DualGAT Model Repository

This repository contains the code and resources for running the DualGAT model on three different datasets.

## Files Overview

- **DualGAT_nas.py**: Code for running the DualGAT model on the NASDAQ 100 dataset.
- **DualGAT_sp500.py**: Code for running the DualGAT model on the S&P 500 dataset.
- **DualGAT_stocknet.py**: Code for running the DualGAT model on the StockNet dataset.

### Output Files

The following output files (valid_pred_ic.csv,valid_label.csv,pred.csv,label.csv) are generated from running our Temporal Pre-training Model:

The code for our Temporal Pre-training Model can be found at [quantbench GitHub Repository](https://github.com/SaizhuoWang/quantbench) by specifying the model parameter as `multi_scale_rnn`.

### Expert Mining

- **expert_mining.py**: This script provides the expert prediction results used as pseudo labels in our study.

### Stocktwits Database

- **Stocktwits_demo.zip**: This zip file contains a subset of our Stocktwits database. It includes examples of tweet messages and the information they contain, demonstrating the format used in our database.

