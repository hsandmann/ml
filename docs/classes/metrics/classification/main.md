<style>
table.confusion-matrix {
    border-collapse: collapse;
    margin: 10px 0;
}
table.confusion-matrix td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: center;
    width: 50px;
    height: 50px;
}
table.confusion-matrix tr:nth-child(even){background-color: #f2f2f2;}
table.confusion-matrix tr:hover {background-color: #ddd;}
table.confusion-matrix th {
    padding-top: 12px;
    padding-bottom: 12px;
    text-align: center;
    background-color: #4CAF50;
    color: white;
}
</style>

Below is a detailed list of metrics commonly used to evaluate the accuracy and performance of classification and regression models in machine learning, including neural networks. The metrics are categorized based on their applicability to classification or regression tasks, with explanations of their purpose and mathematical formulations where relevant.


## Classification Metrics

Classification tasks involve predicting discrete class labels. The following metrics assess the accuracy and effectiveness of such models:

| Metric | Purpose | Formula | Use Case |
|--------|---------|:-------:|----------|
| Accuracy | Measures the proportion of correct predictions across all classes | \( \displaystyle \frac{TP + TN}{TP + TN + FP + FN} \) | Suitable for balanced datasets but misleading for imbalanced ones |
| Precision | Evaluates the proportion of positive predictions that are actually correct | \( \displaystyle \frac{TP}{TP + FP} \) | Important when false positives are costly (e.g., spam detection) |
| Recall (Sensitivity) | Assesses the proportion of actual positives correctly identified | \( \displaystyle \frac{TP}{TP + FN} \) | Critical when false negatives are costly (e.g., disease detection) |
| F1-Score | Harmonic mean of precision and recall, balancing both metrics | \( \displaystyle 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \) | Useful for imbalanced datasets where both precision and recall matter |
| AUC-ROC | Measures the model’s ability to distinguish between classes across all thresholds | Area under the curve plotting True Positive Rate (Recall) vs. False Positive Rate \( \displaystyle \left( \frac{FP}{FP + TN} \right) \) | Effective for binary classification and assessing model robustness |
| AUC-PR | Focuses on precision and recall trade-off, especially for imbalanced datasets | Area under the curve plotting Precision vs. Recall | Preferred when positive class is rare (e.g., fraud detection) |
| Confusion Matrix | Provides a tabular summary of prediction outcomes (TP, TN, FP, FN) | <table class="confusion-matrix"><tr><td>TP</td><td>TN</td></tr><tr><td>FP</td><td>FN</td></tr></table> | Offers detailed insights into class-specific performance, especially for multi-class problems |
| Hamming Loss | Calculates the fraction of incorrect labels to the total number of labels | \( \displaystyle \frac{1}{N} \sum_{i=1}^N \frac{1}{L} \sum_{j=1}^L \mathbf{1}(y_{ij} \neq \hat{y}_{ij}) \) | Suitable for multi-label classification tasks |
| Balanced Accuracy | Average of recall obtained on each class, useful for imbalanced datasets | \( \displaystyle \frac{1}{C} \sum_{i=1}^C \frac{TP_i}{TP_i + FN_i} \) | Effective for multi-class problems with class imbalance |

<!-- | Log Loss | Penalizes incorrect predictions based on predicted probabilities | \( \displaystyle -\frac{1}{N} \sum_{i=1}^N [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)] \) | Common in probabilistic classifiers like neural networks with softmax outputs | -->


Specific better ROC curves:
![ROC Curve Example](https://upload.wikimedia.org/wikipedia/commons/1/13/ROC_curve.svg)