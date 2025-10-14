
Regression tasks predict continuous values. The following metrics evaluate the accuracy of predicted values against true values:

| Metric | Purpose | Use Case |
|--------|---------|----------|
| **Mean Absolute Error (MAE)** <br> \( \displaystyle \frac{1}{N} \sum_{i=1}^N \vert y_i - \hat{y}_i \vert \) | Measures average absolute difference between predictions and true values | Robust to outliers, interpretable as average error |
| **Mean Squared Error (MSE)** <br> \( \displaystyle \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 \) | Measures average squared difference between predictions and true values | Sensitive to outliers, commonly used in neural network loss functions |
| **Root Mean Squared Error (RMSE)** <br> \( \displaystyle \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2} \) | Square root of MSE, providing error in same units as target | Preferred for interpretable error magnitude, widely used in forecasting |
| **Mean Absolute Percentage Error (MAPE)** <br> \( \displaystyle \frac{1}{N} \sum_{i=1}^N \left \vert \frac{y_i - \hat{y}_i}{y_i} \right \vert \cdot 100 \) | Measures average percentage error relative to true values | Useful when relative errors matter (e.g., financial predictions), but sensitive to zero or near-zero true values |
| **$R^2$ (Coefficient of Determination)** <br> \( \displaystyle 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2} \) | Measures proportion of variance in dependent variable explained by model | Indicates model fit, with values closer to 1 indicating better fit |
| **Adjusted $R^2$** <br> \( \displaystyle 1 - \left( \frac{(1 - R^2)(N - 1)}{N - k - 1} \right) \) | Adjusts R² for number of predictors, penalizing overly complex models | Useful when comparing models with different numbers of features |
| **Median Absolute Error ($\text{MedAE}$)** <br> \( \displaystyle \text{median}(\vert y_1 - \hat{y}_1 \vert, \dots, \vert y_N - \hat{y}_N \vert) \) | Measures median of absolute differences, highly robust to outliers | Preferred in datasets with extreme values or non-Gaussian errors |
<!-- | Huber Loss | Combines MSE and MAE, less sensitive to outliers than MSE | \( \displaystyle L_\delta(y_i, \hat{y}_i) = \begin{cases} \frac{1}{2}(y_i - \hat{y}_i)^2 & \text{if } \|y_i - \hat{y}_i\| \leq \delta \\ \delta \|y_i - \hat{y}_i\| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases} \) | Used in robust regression tasks, often as a loss function in neural networks |
| **Explained Variance Score** <br> \( \displaystyle 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)} \) | Measures proportion of variance explained by the model, similar to R² | Indicates how well the model captures variability in the data | -->
