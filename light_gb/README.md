

# Predicting BankNifty's Next-Minute Movement: A LightGBM-Based Approach

## Objective

This project explores the feasibility of predicting the **next one-minute price direction** of the BankNifty index, a major benchmark composed of leading Indian banking stocks. Given the highly volatile and noisy nature of minute-level financial data, the goal is not to achieve perfect foresight but to test whether any **predictive signal** can be extracted using machine learning — specifically, **LightGBM**, a fast and scalable gradient boosting framework.

---

## Model Choice: Why LightGBM?

LightGBM (Light Gradient Boosting Machine) was selected due to its:

- **High efficiency** on large datasets
    
- **Native handling of categorical and numerical features**
    
- **Strong performance** on classification tasks involving tabular data
    

### Conceptual Intuition Behind LightGBM

LightGBM builds an ensemble of decision trees in sequence. Each new tree is trained to **minimize the residual errors** of the previous ensemble. Over time, this leads to a robust model that focuses more on the difficult-to-predict cases.

Key characteristics:

- **Boosting-based learning:** Later trees focus on correcting earlier errors.
    
- **Leaf-wise tree growth:** Unlike level-wise growth (used by some algorithms), LightGBM grows trees by expanding the leaf with the highest loss reduction — leading to faster convergence.
    
- **Histogram-based splits:** Improves speed and reduces memory usage, making it well-suited for high-frequency financial data.
    

---

## Data Overview

The dataset comprises **10 years of one-minute OHLC (Open, High, Low, Close)** data for BankNifty. This includes every market minute over the trading hours from 9:15 AM to 3:30 PM on all trading days.

### Why Minute-Level Data is Challenging

- **High noise-to-signal ratio:** Microstructure noise and random fluctuations dominate at this scale.
    
- **Short-term unpredictability:** Institutional players and algorithms react faster than models can adapt.
    
- **Non-stationarity:** Market behavior varies across time-of-day, day-of-week, and macroeconomic conditions.
    

---

## Feature Engineering

To give the model predictive power, we engineered a rich set of features that capture price behavior, momentum, and market context:

### Technical Indicators

- **Relative Strength Index (RSI)**
    
- **Exponential and Simple Moving Averages (EMA, SMA)**
    
- **MACD and Signal Line**
    
- **Bollinger Bands**
    
- **Average True Range (ATR)**
    

### Lag-Based Features

- Returns and price differences at various lags (1-min, 2-min, …, 10-min)
    
- Rolling standard deviations and means to measure short-term volatility
    

### Time-Based Features

- Minute of the hour
    
- Hour of the day
    
- Day of the week
    

These features aim to capture recurring patterns, volatility spikes, and intraday behavioral shifts.

---

## Prediction Target

The model is tasked with a **binary classification** problem:

- **Target = 1** if the close price in the next minute is higher than the current minute
    
- **Target = 0** otherwise
    

This formulation simplifies the problem into predicting short-term direction rather than magnitude, aligning with the goal of directional trading strategies.

---

## Model Performance

We evaluated the model using:

- **Accuracy:** Proportion of correct predictions
    
- **AUC (Area Under the ROC Curve):** Ability to distinguish upward from downward moves
    
- **F1 Score:** Harmonic mean of precision and recall, especially relevant in imbalanced cases
    

### Baseline Model (Default Hyperparameters)

- `n_estimators`: 100
    
- `learning_rate`: 0.1
    
- `num_leaves`: 31
    
- **Accuracy:** ~50.1%
    
- **AUC Score:** ~0.504
    

This performance is effectively **equivalent to random guessing**. Unsurprising, given the difficulty of the task and the need for tuned hyperparameters.

---

## Optimized Model (via Optuna Hyperparameter Tuning)

We used **Optuna**, a state-of-the-art Bayesian optimization library, to search for the most effective hyperparameter combination.

### Best Parameters Found

- `n_estimators`: 500
    
- `learning_rate`: 0.0198
    
- `num_leaves`: 82
    
- `max_depth`: 11
    
- `min_child_samples`: 69
    
- `subsample`: 0.7596
    
- `colsample_bytree`: 0.7088
    
- `reg_alpha`: 0.3272
    
- `reg_lambda`: 0.7116
    

### Post-Tuning Results

- **Test Accuracy:** 51.74%
    
- **AUC Score:** 0.5244
    
- **F1 Score (for class "UP"):** 0.38





![download](https://github.com/user-attachments/assets/edff194d-e63a-4bd5-8fe5-04bff23811d5)
![download](https://github.com/user-attachments/assets/2fcef008-a636-4892-b9c2-8842a319e68c)
![download](https://github.com/user-attachments/assets/36be9b2c-6b19-4b47-89d3-88ba9542d149)


While the improvement appears modest, the **increase over chance-level performance is statistically meaningful** given the volume of test samples.
**Interpreting the Performance:**

*   **A Glimmer of Skill:** The good news is that after careful tuning, our model is no longer just randomly guessing. An accuracy of 51.7% and an AUC of 0.5244 mean it has found a *very slight* statistical edge. It's demonstrably better than a coin flip. The visual plot, showing around 54.5% correct guesses in that segment, reinforces this.
*   **Still a Tough Challenge:** However, this edge is very small. In the high-stakes, fast-paced world of 1-minute market movements, an accuracy of ~52-54% is unlikely to be consistently profitable after considering trading costs (like brokerage fees and the tiny price differences when you buy/sell).
*   **What the "F1-Score" Tells Us:** The low F1-score for "UP" predictions (0.38) means that while the model might be right a bit more than half the time *when it predicts UP*, it actually *misses* a large number of the times the market truly goes up. It's not very good at *catching* all the upward opportunities.

**In Conclusion:**

Our journey into predicting Banknifty's next-minute moves has been a fascinating one! By carefully tuning our LightGBM model, we've managed to coax out a performance that's *just a hair* better than random chance. It's a testament to how incredibly difficult it is to predict these high-frequency markets.

While our model isn't a crystal ball ready for real-world trading based on these results, it's a great learning experience and shows the power of machine learning in trying to find subtle patterns in complex data. Further improvements might come from entirely new types of data (like news sentiment or order book information), looking at longer time horizons, or developing even more sophisticated models.
