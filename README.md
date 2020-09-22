# LSTM Stock Predictor

![deep-learning.jpg](Images/deep-learning.jpg)

Due to the volatility of cryptocurrency speculation, investors will often try to incorporate sentiment from social media and news articles to help guide their trading strategies. One such indicator is the [Crypto Fear and Greed Index (FNG)](https://alternative.me/crypto/fear-and-greed-index/) which attempts to use a variety of data sources to produce a daily FNG value for cryptocurrency. I have built and evaluated deep learning models using both the FNG values and simple closing prices to determine if the FNG indicator provides a better signal for cryptocurrencies than the normal closing price data.

In oder to do so, I have used deep learning recurrent neural networks to model bitcoin closing prices. One model uses the FNG indicators to predict the closing price while the second model uses a window of closing prices to predict the nth closing price.

I have done the following:

1. [Prepared the data for training and testing](#prepare-the-data-for-training-and-testing)
2. [Built and trained custom LSTM RNNs](#build-and-train-custom-lstm-rnns)
3. [Evaluated the performance of each model](#evaluate-the-performance-of-each-model)
4. [Comparitive Analysis of model and Findings](#comparitive-analysis-and-Findings)
- - -

### Files

[Closing Prices Notebook](Codes/lstm_stock_predictor_closing.ipynb)

[FNG Notebook](Codes/lstm_stock_predictor_fng.ipynb)

- - -

## Instructions


### Prepare the data for training and testing

Created Jupyter Notebooks for each RNN. 

For the Fear and Greed model, I have used the FNG values to try and predict the closing price. A function is provided in the notebook to help with this.

For the closing price model, I have used previous closing prices to try and predict the next closing price. A function is provided in the notebook to help with this.

Each model is built on 70% of the data for training and 30% of the data for testing. For training, useD at least 10 estimators for both models.

Applied a MinMaxScaler to the X and y values to scale the data for the model.

Finally, reshaped the X_train and X_test values to fit the model's requirement of samples, time steps, and features. (*example:* `X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))`)


### Build and train custom LSTM RNNs

In each Jupyter Notebooks, created the same custom LSTM RNN architecture. In one notebook, I have fited the data using the FNG values while in the other, the data was fit using only closing prices.

I have used the same parameters and training steps for each model. This is necessary to compare each model accurately.

I have experimented with the model architecture and parameters to see which provides the best results, but have used the same architecture and parameters when comparing each model.


### Evaluate the performance of each model

Finally, used the testing data to evaluate each model and compare the performance.


### Comparitive Analysis and Findings

I have compared both the above LSTM models by varying the window size and number of units. Based on which I have tried to answer the following questions:

   ![Comparitive Analysis](Images/Comparitive_Analysis.png)

1. Which model has a lower loss?

    LSTM RNN models which uses a window of closing prices to predict the nth closing price has lower losses across all the number_units compared to the model that uses Bitcoin fear and greed index values to predict the nth day closing price as observed in the table above. 

2. Which model tracks the actual values better over time?

    Deep Learning Model which uses a window of closing prices to predict the 5th closing price tracks the actual values better than the model which uses FNG indicators to predict the closing price as shown in the respective graphs below.

   ![LSTM Model with Closing Prices](Images/Plot_CP_4_10.png) ![LSTM Model with FNG](Images/Plot_FNG_4_10.png)

3. Which window size works best for the model? 

    I have narrowed down on LSTM model with window_size=4 and number_units = 10 for both the models as the losses are the least compared across other variations as showcased in the table above. <i> Note that, starter code output was based on window_size=10 and number_units = 30 </i>

- - -

### Resources

[Keras Sequential Model Guide](https://keras.io/getting-started/sequential-model-guide/)

[Illustrated Guide to LSTMs](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)

[Stanford's RNN Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

- - -