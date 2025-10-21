
Naive Bayes Classifier is a family of probabilistic machine learning algorithms used primarily for classification tasks. It's based on Bayes' Theorem, which describes the probability of an event based on prior knowledge of conditions that might be related to the event.

The "naive" part comes from the strong assumption that the features (predictors) in the dataset are independent of each other given the class label. This simplifies calculations significantly, making the algorithm efficient even for large datasets.

In essence, Naive Bayes calculates the probability that a given instance belongs to a particular class and assigns it to the class with the highest posterior probability. It's particularly popular in text classification because it handles high-dimensional data well (e.g., word counts in documents).

Key variants include:

- **Gaussian Naive Bayes**: Assumes features follow a normal distribution (for continuous data).
- **Multinomial Naive Bayes**: Suited for discrete data, like word frequencies in text.
- **Bernoulli Naive Bayes**: For binary/boolean features, like word presence/absence.

Naive Bayes is derived from Bayes' Theorem:

\[ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} \]

In classification terms:

- Let \( C \) be the class label.
- Let \( X = (x_1, x_2, \dots, x_n) \) be the feature vector.

We want the posterior probability \( P(C|X) \), and we classify \( X \) to the class \( C_k \) that maximizes this:

\[ P(C_k|X) = \frac{P(X|C_k) \cdot P(C_k)}{P(X)} \]

Since \( P(X) \) is constant for all classes, we can ignore it and focus on maximizing \( P(X|C_k) \cdot P(C_k) \).

The naive assumption: Features are conditionally independent given the class, so:

\[ P(X|C_k) = P(x_1|C_k) \cdot P(x_2|C_k) \cdot \dots \cdot P(x_n|C_k) = \prod_{i=1}^n P(x_i|C_k) \]

- **Prior \( P(C_k) \)**: Probability of class \( C_k \) in the training data (e.g., fraction of samples in that class).
- **Likelihood \( P(x_i|C_k) \)**: Depends on the variant:
    - For Multinomial: \( P(x_i|C_k) = \frac{N_{ki} + \alpha}{N_k + \alpha \cdot V} \) (with Laplace smoothing, where \( \alpha \) is the smoothing parameter, \( N_{ki} \) is count of feature i in class k, \( N_k \) is total counts in class k, V is vocabulary size).
    - For Gaussian: \( P(x_i|C_k) = \frac{1}{\sqrt{2\pi\sigma_{ki}^2}} \exp\left( -\frac{(x_i - \mu_{ki})^2}{2\sigma_{ki}^2} \right) \) (mean \( \mu \) and variance \( \sigma^2 \) estimated from training data).

To avoid zero probabilities (when a feature doesn't appear in a class), we use additive smoothing (e.g., Laplace with \( \alpha = 1 \)).

The final prediction is:

\[ \hat{C} = \arg\max_{C_k} P(C_k) \prod_{i=1}^n P(x_i|C_k) \]

(Often computed in log space to avoid underflow: \( \log P(C_k) + \sum_{i=1}^n \log P(x_i|C_k) \).)

## Numerical Simulation

Let's walk through a simple example using Multinomial Naive Bayes for text classification. Suppose we classify fruits based on features: "sweet" (count), "crunchy" (count), "red" (count). We have this training data:

| Fruit   | Sweet | Crunchy | Red | Class    |
|---------|-------|---------|-----|----------|
| Apple1 | 1     | 2       | 1   | Apple   |
| Apple2 | 0     | 1       | 2   | Apple   |
| Orange1| 3     | 0       | 0   | Orange  |
| Orange2| 2     | 1       | 0   | Orange  |

Classes: 2 Apples, 2 Oranges. So, priors: P(Apple) = 2/4 = 0.5, P(Orange) = 0.5.

Total counts per class:

- Apple: Sweet=1+0=1, Crunchy=2+1=3, Red=1+2=3. Total words in Apple: 1+3+3=7.
- Orange: Sweet=3+2=5, Crunchy=0+1=1, Red=0+0=0. Total words in Orange: 5+1+0=6.

Vocabulary size V=3 (sweet, crunchy, red). Use Laplace smoothing (\( \alpha=1 \)).

Likelihoods:

- P(sweet|Apple) = (1+1)/(7+3) = 2/10 = 0.2
- P(crunchy|Apple) = (3+1)/10 = 4/10 = 0.4
- P(red|Apple) = (3+1)/10 = 4/10 = 0.4
- P(sweet|Orange) = (5+1)/(6+3) = 6/9 ≈ 0.667
- P(crunchy|Orange) = (1+1)/9 = 2/9 ≈ 0.222
- P(red|Orange) = (0+1)/9 = 1/9 ≈ 0.111

Now, classify a new fruit: Sweet=2, Crunchy=1, Red=0.

Posterior for Apple: 

\[
\begin{align*}
& P(\text{Apple}) * P(\text{sweet}|\text{Apple})^2 * P(\text{crunchy}|\text{Apple})^1 * P(\text{red}|\text{Apple})^0 \\
= & 0.5 * (0.2)^2 * 0.4 * 1 \\
= & 0.5 * 0.04 * 0.4 \\
= & 0.008
\end{align*}
\]

Posterior for Orange:

\[
\begin{align*}
& P(\text{Orange}) * P(\text{sweet}|\text{Orange})^2 * P(\text{crunchy}|\text{Orange})^1 * P(\text{red}|\text{Orange})^0 \\
= & 0.5 * (0.667)^2 * 0.222 * 1 \\
≈ & 0.5 * 0.445 * 0.222 \\
≈ & 0.5 * 0.099 \\
≈ & 0.0495
\end{align*}
\]

Higher for Orange, so classify as Orange.

!!! info "Note"

    The calculations above can be done in log space to prevent numerical underflow:

    Using logs:
    
    \[
    \begin{align*}
    log(\text{Apple}) & = log(0.5) + 2*log(0.2) + log(0.4) \\
    & ≈ -0.693 + 2*(-1.609) + (-0.916) \\
    & ≈ -0.693 -3.218 -0.916 \\
    & = -4.827
    \end{align*}
    \]

    \[
    \begin{align*}
    log(\text{Orange}) & ≈ -0.693 + 2*(-0.405) + (-1.507) \\
    & ≈ -0.693 -0.81 -1.507 \\
    & ≈ -3.01
    \end{align*}
    \]

    \( exp(-3.01) > exp(-4.827) \), so Orange wins.

## Implementation

=== "Result"

    ```python exec="1" html="1"
    --8<-- "docs/classes/naive_bayes/naive-bayes-sklearn.py"
    ```

=== "Code"

    ```python exec="0"
    --8<-- "docs/classes/naive_bayes/naive-bayes-sklearn.py"
    ```

## Applications

Naive Bayes is widely used due to its simplicity and speed:

- **Spam Detection**: Classify emails as spam/ham based on word frequencies (e.g., Gmail's early filters).
- **Sentiment Analysis**: Determine if reviews are positive/negative by analyzing text features.
- **Document Classification**: Categorize news articles into topics like sports, politics.
- **Medical Diagnosis**: Predict diseases from symptoms (assuming independence).
- **Recommendation Systems**: Basic user preference prediction.
- **Real-Time Prediction**: Fraud detection in finance, where quick decisions are needed on high-dimensional data.

It's especially effective in natural language processing (NLP) with bag-of-words models.

## Pros and Cons

### Pros

<div class="grid cards" markdown>

-   __**Simple and Fast**__

    ---

    Easy to implement, trains quickly (O(n) time), and predicts in constant time relative to data size.

-   __**Handles High Dimensions**__

    ---

    Performs well with many features, like in text data (curse of dimensionality resistant).

-   __**Scalable**__

    ---

    Can be applied to large datasets and updated easily with new data.

-   __**Interpretable**__

    ---

    Probabilistic nature allows understanding of feature contributions to predictions.

-   __**Good with Small Datasets**__

    ---

    Requires less training data than complex models.

-   __**Probabilistic Output**__

    ---

    Provides probability estimates, not just labels.

-   __**Robust to Irrelevant Features**__

    ---

    The independence assumption helps ignore noise.

</div>

### Cons

<div class="grid cards" markdown>


-   __**Independence Assumption**__

    ---

    Rarely holds in real data (e.g., words like "machine" and "learning" are correlated), leading to suboptimal performance.

-   __**Zero Probability Problem**__

    ---

    If a feature-class combo is missing in training, probability becomes zero (mitigated by smoothing, but not perfect).

-   __**Poor for Continuous Data Without Proper Variant**__

    ---

    Multinomial works for counts, but Gaussian assumes normality, which may not fit.

-   __**Biased Estimates**__

    ---

    Posterior probabilities are often inaccurate, though classification can still be good.

-   __**Sensitive to Data Representation**__

    ---
    
    In text, needs careful preprocessing (e.g., stemming, stop words).

</div>

## Additional

### Play Golf

A classic example used to illustrate Naive Bayes is predicting whether to play golf based on weather conditions. The dataset includes features like Outlook (Sunny, Overcast, Rain), Temperature (Hot, Mild, Cool), Humidity (High, Normal), and Windy (True, False), along with the target variable Play (Yes, No).

| | Outlook  | Temperature | Humidity | Windy | Play |
|:-:|----------|-------------|----------|-------|------|
| 1 | Sunny    | Hot         | High     | False | No   |
| 2 | Sunny    | Hot         | High     | True  | No   |
| 3 | Overcast | Hot         | High     | False | Yes  |
| 4 | Rain     | Mild        | High     | False | Yes  |
| 5 | Rain     | Cool        | Normal   | False | Yes  |
| 6 | Rain     | Cool        | Normal   | True  | No   |
| 7 | Overcast | Cool        | Normal   | True  | Yes  |
| 8 | Sunny    | Mild        | High     | False | No   |
| 9 | Sunny    | Cool        | Normal   | False | Yes  |
| 10| Rain     | Mild        | Normal   | False | Yes  |
| 11| Sunny    | Mild        | Normal   | True  | Yes  |
| 12| Overcast | Mild        | High     | True  | Yes  |
| 13| Overcast | Hot         | Normal   | False | Yes  |
| 14| Rain     | Mild        | High     | True  | No   |

Using this dataset, we can calculate the probabilities needed to predict whether to play golf given specific weather conditions using the Naive Bayes approach.

#### The goal

The objective is to predict whether a person will play golf on a new, unseen day based on its weather conditions, such as:

Outlook = Sunny, Temperature = Cool, Humidity = High, and Windy = True.

To make this prediction, we will calculate the posterior probabilities for both classes (Play = Yes and Play = No) using the Naive Bayes formula and then compare them.

#### Calculations steps

1. **Calculate prior probabilities**:

    - **P(Play = Yes)**: 9 out of 14 days were "Yes."

        \( \displaystyle P(\text{Play} = \text{Yes}) = \frac{\text{Number of Yes}}{\text{Total}} = \frac{9}{14} = 0.643 \)

    - **P(Play = No)**: 5 out of 14 days were "No."

        \( \displaystyle P(\text{Play} = \text{No}) = \frac{\text{Number of No}}{\text{Total}} = \frac{5}{14} = 0.357 \)

2. **Calculate likelihood probabilities**:

    For each feature given the class, we calculate the likelihoods.

    

    | Feature | P(Feature=Value\|Play=Yes) | P(Feature=Value\|Play=No) |
    |---------|----------------------------|---------------------------|
    | **Outlook=Sunny** | \( \displaystyle P(\text{Sunny}\|\text{Yes}) = \frac{2}{9} \) | \( \displaystyle P(\text{Sunny}\|\text{No}) = \frac{3}{5} \) |
    | **Temperature=Cool** | \( \displaystyle P(\text{Cool}\|\text{Yes}) = \frac{3}{9} \) | \( \displaystyle P(\text{Cool}\|\text{No}) = \frac{1}{5} \) |
    | **Humidity=High** | \( \displaystyle P(\text{High}\|\text{Yes}) = \frac{3}{9} \) | \( \displaystyle P(\text{High}\|\text{No}) = \frac{4}{5} \) |
    | **Windy=True** | \( \displaystyle P(\text{True}\|\text{Yes}) = \frac{3}{9} \) | \( \displaystyle P(\text{True}\|\text{No}) = \frac{3}{5} \) |


3. **Calculate posterior probabilities (Bayes' theorem)**:

    Using the naive assumption that all features are independent, apply the theorem to find the posterior probability for each class. The simplified formula used is:

    \[ P(C|X) \propto P(C) \cdot \prod_{i} P(x_i|C) \]

    - **P(Yes | Sunny, Cool, High, True)**:

        \(\begin{align*}
        = & P(\text{Yes}) \cdot P(\text{Sunny}|\text{Yes}) \cdot P(\text{Cool}|\text{Yes}) \cdot P(\text{High}|\text{Yes}) \cdot P(\text{True}|\text{Yes}) \\
        = & \frac{9}{14} \cdot \frac{2}{9} \cdot \frac{3}{9} \cdot \frac{3}{9} \cdot \frac{3}{9} \\
        \approx & 0.0053
        \end{align*}\)

    - **P(No | Sunny, Cool, High, True)**:

        \(\begin{align*}
        = & P(\text{No}) \cdot P(\text{Sunny}|\text{No}) \cdot P(\text{Cool}|\text{No}) \cdot P(\text{High}|\text{No}) \cdot P(\text{True}|\text{No}) \\
        = & \frac{5}{14} \cdot \frac{3}{5} \cdot \frac{1}{5} \cdot \frac{4}{5} \cdot \frac{3}{5} \\
        \approx & 0.02056
        \end{align*}\)

4. **Make the prediction**:

    Since \( P(\text{No} | \text{Sunny, Cool, High, True}) \approx 0.02056 \) is greater than \( P(\text{Yes} | \text{Sunny, Cool, High, True}) \approx 0.0053 \), we predict that the person will **not** play golf on that day.

<iframe width="100%" height="470" src="https://www.youtube.com/embed/CPqOCI0ahss" title="Naïve Bayes Classifier -  Fun and Easy Machine Learning" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>