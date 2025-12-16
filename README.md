
# Transformer Temporal Embeddings for Volatility Forecasting

This project investigates the efficacy of Transformer-based architectures for forecasting realized volatility in financial time series. A core challenge in applying Transformers to continuous time series data is effectively encoding temporal information. This repository provides a modular framework to experiment with and compare various temporal embedding strategies, moving beyond standard positional encodings used in NLP.

Key features include:

* **Multi-Embedding Support**: Compare purely relative mechanisms (**ALiBi**), continuous time representations (**Time2Vec**, **CTLPE**), and standard absolute encodings (**Sinusoidal**, **Learned**).
* **Residual Forecasting**: The model is designed to predict the *innovation* (change) in volatility relative to a naive baseline (the previous period's volatility), stabilizing training and improving convergence.
* **End-to-End Pipeline**: Includes tools for data ingestion, preprocessing (log returns & realized vol calculation), training, and evaluation.

In order to run this project, make sure you have wrds account (or can use `download_ohlcv.py` for yfinance). Also download the fi-2010 dataset from `https://drive.google.com/drive/folders/1l_YpPE3x_-FgTfOB9AyXlKuR5SLcopmk?usp=sharing`, and put the downloaded data in folder (or whatever path you like as long as you put it in the args).
