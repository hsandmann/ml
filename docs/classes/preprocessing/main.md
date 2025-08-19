Data preprocessing is a critical phase in the development of neural network models, ensuring that raw data is transformed into a suitable format for effective training and inference. This text explores both basic and advanced preprocessing techniques, drawing from established methodologies in machine learning and deep learning. Basic techniques focus on cleaning and normalizing data to handle inconsistencies and scale issues, while advanced methods address complex challenges such as data scarcity, imbalance, and high dimensionality. The discussion highlights their relevance to neural networks, including convolutional neural networks (CNNs), recurrent neural networks (RNNs), and transformers, with emphasis on improving model convergence, generalization, and performance.

Neural networks, as powerful function approximators, are **highly sensitive to the quality and format of input data**. Poorly prepared data can lead to slow convergence, overfitting, or suboptimal accuracy. Preprocessing mitigates these issues by addressing noise, inconsistencies, and structural mismatches in datasets. It encompasses a series of steps that transform raw data into a form that aligns with the assumptions and requirements of neural architectures. For instance, in supervised learning tasks, preprocessing ensures features are scaled appropriately to prevent gradient issues during backpropagation. This text delineates basic techniques, which are foundational and widely applicable, and advanced techniques, which are more specialized and often domain-specific, such as for image, text, or time-series data.

## Typical Preprocessing Tasks

| Task | Description |
|------|-------------|
| **Text Cleaning** | Remove unwanted characters, stop words, and perform stemming/lemmatization. |
| **Normalization** | Standardize text formats, such as date and currency formats. |
| **Tokenization** | Split text into words or subwords for easier analysis. |
| **Feature Extraction** | Convert text into numerical features using techniques like TF-IDF or word embeddings. |
| **Data Augmentation** | Generate synthetic data to increase dataset size and diversity. |

A typical dataset for machine learning tasks might include columns of different data types, such as numerical, categorical, and text, eg.:

```python exec="on" html="0"
--8<-- "docs/classes/preprocessing/titanic-original.py"
```
/// caption
Sample rows from the Titanic dataset
///


## Data Cleaning

Data cleaning involves identifying and rectifying errors, inconsistencies, and missing values in the dataset. Missing values, common in real-world data, can be handled by imputation methods such as mean, median, or mode substitution, or by removing affected rows/columns if the loss is minimal. For example, in pandas, this can be implemented as `df.fillna(df.mean())` for mean imputation. Outliers, which may skew neural network training, are detected using statistical methods like z-scores or interquartile ranges and can be winsorized or removed. Noise reduction, such as smoothing time-series data with moving averages, is also essential, particularly for RNNs where temporal dependencies are critical. Inconsistent data, like varying formats in text (e.g., dates), requires standardization to ensure uniformity. Overall, data cleaning enhances data quality, reducing the risk of misleading patterns during neural network optimization.

=== "Result"

    ```python exec="on" html="0"
    --8<-- "docs/classes/preprocessing/titanic-fill-nan.py"
    ```
    
=== "Code"

    ```python
    --8<-- "docs/classes/preprocessing/titanic-fill-nan.py"
    ```

## Encoding Categorical Variables

Categorical data, non-numeric by nature, must be converted for neural network input. One-hot encoding creates binary vectors for each category, e.g., transforming colors ```['red', 'blue', 'green']``` into ```[[1,0,0], [0,1,0], [0,0,1]]```. This avoids ordinal assumptions but increases dimensionality, which can be mitigated by embedding layers in neural networks for high-cardinality features. Label encoding assigns integers (e.g., 0 for "red", 1 for "blue"), suitable for ordinal categories but risky for nominal ones due to implied ordering. For text data in NLP tasks with transformers, tokenization and subword encoding (e.g., WordPiece) are basic steps to map words to integer IDs.

=== "Result"

    ```python exec="on" html="0"
    --8<-- "docs/classes/preprocessing/titanic-preprocessing.py"
    ```

=== "Code"

    ```python
    --8<-- "docs/classes/preprocessing/titanic-preprocessing.py"
    ```


## Normalization and Standardization

Normalization scales features to a bounded range, typically $[0, 1]$, using min-max scaling:

$$
x' = \displaystyle \frac{x - \min(x)}{\max(x) - \min(x)}
$$

This is crucial for neural networks employing sigmoid or tanh activations, as it prevents saturation.

Standardization, or z-score normalization, transforms data to have a mean of $0$ and standard deviation of $1$:

$$
x' = \frac{x - \mu}{\sigma},
$$

where \(\mu\) is the mean and \(\sigma\) the standard deviation. It is preferred for networks with ReLU activations or when data distributions are Gaussian-like, aiding faster gradient descent convergence. In practice, libraries like scikit-learn provide `MinMaxScaler` and `StandardScaler` for these operations. These techniques are especially vital in multilayer perceptrons (MLPs) and CNNs, where feature scales can dominate loss landscapes.

Below is an example of how to apply normalization and standardization using pandas, based on the [NASDAQ Apple stock price dataset](https://ranaroussi.github.io/yfinance/){target='_blank'}:

=== "Result"

    ```python exec="on" html="0"
    --8<-- "docs/classes/preprocessing/invest-preprocessing.py"
    ```
    
=== "Original"

    ```python exec="on" html="0"
    --8<-- "docs/classes/preprocessing/invest-original.py"
    ```

=== "Code"

    ```python
    --8<-- "docs/classes/preprocessing/invest-preprocessing.py"
    ```


## Feature Scaling

Feature scaling overlaps with normalization but specifically addresses disparate scales across features. Beyond min-max and z-score, logarithmic scaling (\( x' = \log(x + 1) \)) handles skewed distributions, common in financial data for neural forecasting models. Scaling ensures equal contribution of features during weight updates in stochastic gradient descent (SGD).


## Data Augmentation

Data augmentation artificially expands datasets to combat overfitting, particularly in CNNs for image classification. Basic operations include flipping, rotation (e.g., by 90° or random angles), and cropping, while advanced methods involve adding noise (Gaussian or salt-and-pepper) or color jittering. For text data in RNNs or transformers, techniques like synonym replacement, random insertion/deletion, or back-translation (translating to another language and back) generate variations while preserving semantics. In time-series for LSTMs, window slicing or synthetic minority over-sampling technique (SMOTE)[^8] variants create augmented sequences. Generative models like GANs (Generative Adversarial Networks) represent cutting-edge augmentation, producing realistic synthetic samples. These methods improve generalization by exposing models to diverse inputs.

## Handling Imbalanced Data

Imbalanced datasets, where classes are unevenly represented, bias neural networks toward majority classes. Advanced resampling includes oversampling minorities (e.g., SMOTE, which interpolates new instances) or undersampling majorities. Class weighting assigns higher penalties to minority misclassifications in the loss function, e.g., weighted cross-entropy. Ensemble methods, like balanced random forests integrated with neural embeddings, or focal loss in object detection CNNs, further address this. For sequential data, temporal resampling ensures balanced windows.

## Feature Engineering and Selection

Feature engineering crafts new features from existing ones, such as polynomial terms or interactions (e.g., \( x_1 \times x_2 \)) to capture non-linearities before neural input. Selection techniques like mutual information or recursive feature elimination reduce irrelevant features, alleviating the curse of dimensionality in high-dimensional data for autoencoders or dense networks. Embedded methods, like L1 regularization in neural training, perform selection during optimization.

## Dimensionality Reduction

Techniques like Principal Component Analysis (PCA) project data onto lower-dimensional spaces while preserving variance:

$$
X' = X \cdot W
$$

where \(W\) are principal components. Autoencoders, a neural-based approach, learn compressed representations through encoder-decoder architectures. t-SNE or UMAP are used for visualization but less for preprocessing due to non-linearity. These are vital for CNNs on high-resolution images or transformers on long sequences to reduce computational load.

PCA is widely used for dimensionality reduction[^5], while t-SNE[^6] and UMAP[^7] are popular for visualizing high-dimensional data in 2D or 3D spaces.

Basically, PCA identifies orthogonal axes (principal components) capturing maximum variance, enabling efficient data representation. Autoencoders, trained to reconstruct inputs, learn compact latent spaces, useful for denoising or anomaly detection.

!!! info "PCA Steps[^5]"

    **1. Standardize the data:**

    $$
    X' = \frac{X - μ}{σ}
    $$

    **2. Compute the covariance matrix:**

    $$
    C = \frac{1}{n} * (X'ᵀ * X')
    $$

    **3. Calculate eigenvalues and eigenvectors:**

    $$
    \text{eigvals}, \text{eigvecs} = \text{np.linalg.eig}(C)
    $$

    **4. Sort eigenvectors by eigenvalues in descending order.**

    **5. Select top \(k\) eigenvectors to form a new feature space**

    $$
    Y = X' * W
    $$

    where \(W\) is the matrix of selected eigenvectors.

A example of PCA applied to the Iris dataset:

```python
--8<-- "docs/classes/preprocessing/iris-pca.py"
```

Now, the same example using scikit-learn is shown below:

```python
--8<-- "docs/classes/preprocessing/iris-pca-sklearn.py"
```

Eigenfaces, a PCA variant, is used in face recognition tasks to reduce image dimensions while retaining essential features[^4]. In NLP, techniques like Latent Semantic Analysis (LSA) apply SVD (Singular Value Decomposition) to reduce term-document matrices, enhancing transformer efficiency.

## Domain-Specific Advanced Techniques

For time-series in RNNs, techniques include Fast Fourier Transform (FFT) for frequency domain conversion or segmentation into fixed windows. In text preprocessing for sentiment analysis, advanced steps encompass negation handling (e.g., marking "not good" as "not_pos"), intensification (e.g., "very good" as "strong_pos"), and POS tagging to retain sentiment-bearing words. For images in CNNs, advanced signal processing like wavelet transforms or conversion to spectrograms enhances fault diagnosis applications.




[^1]: [Scikit-learn - Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html){target='_blank'}

[^2]: [TensorFlow - Data Augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation){target='_blank'}

[^3]: [AutoML - Automated Machine Learning](https://arxiv.org/abs/1708.02002){target='_blank'}

[^4]: [Face Recognition with OpenCV](https://docs.opencv.org/4.x/da/d60/tutorial_face_main.html){target='_blank'}

[^5]: [PCA - Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis){target='_blank'}

[^6]: [Principal Component Analysis (PCA) from Scratch](https://bagheri365.github.io/blog/Principal-Component-Analysis-from-Scratch/){target='_blank'}

[^7]: [t-SNE - t-distributed Stochastic Neighbor Embedding](https://distill.pub/2016/misread-tsne/){target='_blank'}

[^8]: [SMOTE - Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813){target='_blank'}

[^9]: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002){target='_blank'}

[^10]: [Word Embeddings - Word2Vec, GloVe, FastText](https://nlp.stanford.edu/projects/glove/){target='_blank'}
