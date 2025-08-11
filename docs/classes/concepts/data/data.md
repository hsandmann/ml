All the concepts of Machine Learning are based on data. The quality and quantity of available data are fundamental to the success of any machine learning model. In this context, it is important to understand how data is structured, processed, and used to train models.

## Nature of Data

Data can be thought of as a collection of features or attributes that describe a particular phenomenon or object. In machine learning, data is often represented as a matrix, where each row corresponds to an example and each column corresponds to a feature. This representation is known as the **feature space**.

A feature is a measurable property or characteristic of the phenomenon being studied. Features can be **numerical** (e.g., height, weight) or **categorical** (e.g., color, type). The set of features used to describe the data is known as the **feature set** or **feature vector**. Features are used to describe the data and can be used as input to machine learning models. **Each feature is a dimension of the feature space, and the dataset is represented as a point in this space.**

Features can be numerical or categorical:

- **Numerical features** are those that can take continuous values, such as height or weight;
- **Categorical features** are those that take discrete values, such as color or type.

Depending on the type of machine learning algorithm, features may be treated differently. For example, some algorithms work better with numerical features, while others are more suited for categorical features. In this context, it is necessary to convert categorical features into a format that algorithms can understand, such as using one-hot encoding[^1] or label encoding[^2].

Additionally, numerical features, such as height or weight, are often normalized to ensure that all features contribute equally to the model. Normalization is a preprocessing technique that adjusts the values of features to a common scale, typically between 0 and 1 or -1 and 1. A common approach to normalization is min-max scaling, which transforms the data by subtracting the minimum value and dividing by the range (maximum - minimum). This ensures that all features are on the same scale and can improve the performance of many machine learning algorithms.

## Datasets

Data is often stored in datasets, which are structured collections of data that can be easily accessed, managed, and updated. Datasets can be relational (e.g., SQL databases) or non-relational (e.g., NoSQL databases). Relational datasets store data in tables with predefined schemas, while non-relational datasets allow for more flexible data structures. Some common types of datasets used in machine learning include:

- [**UCI Machine Learning Repository**](https://archive.ics.uci.edu/ml/index.php){:target="_blank"}: a collection of datasets for machine learning tasks, including classification, regression, and clustering.
- [**Kaggle Datasets**](https://www.kaggle.com/datasets){:target="_blank"}: a platform that offers a wide variety of datasets for different machine learning tasks, from text classification to image recognition.
- [**OpenML**](https://www.openml.org/){:target="_blank"}: a collaborative platform for sharing and organizing machine learning datasets and experiments.
- [**Google Dataset Search**](https://datasetsearch.research.google.com/){:target="_blank"}: a search engine for datasets across the web, allowing users to find datasets for various machine learning tasks.
- [**AWS Open Data Registry**](https://registry.opendata.aws/){:target="_blank"}: a collection of publicly available datasets hosted on Amazon Web Services, covering a wide range of domains, including climate, healthcare, and transportation.
- [**Data.gov**](https://www.data.gov/){:target="_blank"}: a repository of datasets provided by the U.S. government, covering various topics such as agriculture, health, and energy.
- [**FiveThirtyEight Data**](https://data.fivethirtyeight.com/){:target="_blank"}: a collection of datasets used in articles by FiveThirtyEight, covering topics such as politics, sports, and economics.
- [**Awesome Public Datasets**](https://github.com/awesomedata/awesome-public-datasets){:target="_blank"}: a curated list of high-quality public datasets for various domains.
- [**The World Bank Open Data**](https://data.worldbank.org/){:target="_blank"}: a collection of global development data, including economic, social, and environmental indicators.
- [**IMDB Datasets**](https://www.imdb.com/interfaces/){:target="_blank"}: a collection of datasets related to movies, TV shows, and actors, useful for natural language processing and recommendation systems.
- [**Yelp Open Dataset**](https://www.yelp.com/dataset){:target="_blank"}: a dataset containing business reviews, user data, and check-ins, useful for sentiment analysis and recommendation systems.

## Data Quality

Data quality is a critical aspect of machine learning, as the performance of models heavily depends on the quality of the data used for training. Poor quality data can lead to inaccurate predictions and unreliable models. Common issues with data quality include:

- **Missing data**: values that are not available for some variables;
- **Duplicate data**: records that appear more than once in the dataset;
- **Noisy data**: values that are inconsistent or incorrect;
- **Imbalanced data**: when one class is much more frequent than another, which can lead to a biased model.
- **Inconsistent data**: when the data does not follow a consistent pattern or format, making it difficult to analyze and train the model.
- **Irrelevant data**: variables that do not contribute to the machine learning task and may harm the model's performance.

To address these issues, it is common to perform a data cleaning and preprocessing process, which may include:

- **Removing missing data**: excluding records with missing values or imputing values based on other observations.
- **Removing duplicates**: identifying and removing duplicate records.
- **Handling noisy data**: applying smoothing or filtering techniques to reduce noise in the data.
- **Balancing classes**: techniques such as undersampling or oversampling - **data augmentation**[^6] - to deal with imbalanced classes.
- **Normalization**: adjusting the values of variables to a common scale, ensuring that all variables contribute equally to the model.
- **Transforming variables**: applying techniques such as logarithm, square root, or Box-Cox to transform non-linear variables into linear ones.
- **Encoding categorical variables**: converting categorical variables into a format that algorithms can understand, such as using one-hot encoding or label encoding.

Additionally, it is important to consider the order of the data, especially in time series problems, where the sequence of the data is crucial for analysis and modeling.

## Data Volume and Balance

The volume and balance of data are also important factors to consider in machine learning. Data volume refers to the amount of data available for training and testing machine learning models. The larger the volume of data, the more information the model can learn, which usually results in better performance. However, it is also important to consider the quality of the data, as noisy or irrelevant data can harm the model's performance.

Additionally, it is important to consider class balancing, especially in classification problems. Class balancing refers to the equitable distribution of classes in the dataset. **If one class is much more frequent than another, this can lead to a biased model**, which tends to predict the majority class. To address this issue, techniques such as undersampling or oversampling can be used to balance the classes. Undersampling involves removing records from the majority class, while oversampling involves duplicating records from the minority class or generating synthetic data.

For supervised learning models, it is essential to have a labeled dataset, where each example has an input (features) and an output (label). This allows the model to learn to map the inputs to the correct outputs.

Furthermore, the data can be classified into three main categories:

| Set | Description |
|--------------------|-----------|
| **Train** | Used to train the model, allowing it to learn the patterns and relationships between features and labels. |
| **Test** | Used to tune the model's hyperparameters and prevent overfitting, ensuring it generalizes well to new examples. |
| **Validation** | Used to evaluate the model's performance on unseen data, ensuring it generalizes well to new examples. |

---

## Some Examples of Datasets

### **Salmon vs Seabass**

A fictional dataset about salmon and seabass, where each record is labeled as "salmon" or "seabass". The goal is to better understand how the data can be used to differentiate between the two species. In this context, the features may include, for example: size and brightness[^5].

#### Problem

Imagine you have a fish sorting machine. Every day, fishing boats dump tons of fish onto a conveyor belt, and the goal of the machine is to separate the fish, classifying them as "salmon" or "seabass" based on their characteristics.

The conveyor belt has sensors that measure the size and brightness of the fish. Based on these measurements, the machine must decide whether the fish is a salmon or a seabass.

$$
\mathbf{x} = \begin{bmatrix}
x_1 \\
x_2 \\
\end{bmatrix}
$$

where \(x_1\) is the size of the fish and \(x_2\) is the brightness of the fish. The machine must learn to classify the fish based on these characteristics, using a function \(f\) that maps the input features to the output class: salmon or seabass.


#### Sample Data

To better understand the data, a sample of fish was taken, where each fish is described by its size and brightness characteristics. The table below presents a sample of the collected data:

| Size (cm) | Brightness (0-10) | Species |
|:--:|:--:|:--:|
| 60 | 6 | salmon |
| 45 | 5 | seabass |
| 78 | 7 | salmon |
| 90 | 5.2 | salmon |
| 71 | 9 | salmon |
| 80 | 3 | seabass |
| 64 | 6 | salmon |
| 58 | 2 | seabass |
| 63 | 6.8 | seabass |
| 50 | 4 | seabass |

When plotting the data, we can visualize the size and brightness of each fish in a two-dimensional space. Each fish is represented by a point in this space, where the x-axis represents the size and the y-axis represents the brightness. The points are colored according to their species: salmon or seabass.


```python exec="1" html="1"
--8<-- "docs/classes/concepts/data/salmon_vs_seabass_1.py"
```
/// caption
Sample data of salmon and seabass, where each fish is described by its size and brightness characteristics. The points are colored according to their species: salmon (blue) or seabass (orange). For 1-dimensional data, the points are plotted along the x-axis, representing the size and brightness of the fish.
///


```python exec="1" html="1"
--8<-- "docs/classes/concepts/data/salmon_vs_seabass_2.py"
```

/// caption
Sample data of salmon and seabass, where each fish is described by its size and brightness characteristics. The points are colored according to their species: salmon (blue) or seabass (orange). The x-axis represents the size of the fish, while the y-axis represents its brightness.
///

The machine must learn to draw a line that separates the two classes, salmon and seabass, based on the size and brightness characteristics. This line is called a **decision boundary**. So that, as soon as a new fish is placed on the conveyor belt, the machine can decide whether it is a salmon or a seabass based on its size and brightness characteristics - as shown in the figure on the right.

In general, in the context of classification, the machine must learn to draw **decision boundaries** in a multidimensional feature space. Allowing, when a new example is presented, the machine to decide which class it belongs to based on the characteristics of the example.

!!! warning "Attention"

    The decision boundary is not always linear. In some cases, the data may be distributed in a way that requires a non-linear decision boundary to separate the classes effectively. In such cases, more complex models, such as neural networks or support vector machines with kernels, may be needed to find an appropriate separation.

### **Iris Dataset**

[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris): the Iris Dataset is a classic dataset used for classification tasks in machine learning. It was introduced by Sir Ronald A. Fisher in 1936[^3] and has since become one of the most widely used datasets in the field[^4].

The Iris Dataset is a classic and **real** dataset used for flower classification. It contains 150 samples of three different species of Iris flowers (Iris setosa, Iris versicolor, and Iris virginica), with four features: petal and sepal length and width.

![](iris_dataset.png)

The dataset is widely used to demonstrate machine learning algorithms, especially for classification tasks. It is simple enough to be easily understood, but also presents interesting challenges for more complex models.

A sample of the Iris dataset is presented in the table below:

| sepal length<br>(cm) | sepal width<br>(cm) | petal length<br>(cm) | petal width<br>(cm) | class   |
|:--:|:--:|:--:|:--:|----|
| 5.7     | 3.0     | 4.2     | 1.2     | versicolor |
| 5.7     | 2.9     | 4.2     | 1.3     | versicolor |
| 6.2     | 2.9     | 4.3     | 1.3     | versicolor |
| 5.1     | 3.5     | 1.4     | 0.2     | setosa  |
| 4.9     | 3.0     | 1.4     | 0.2     | setosa  |
| 4.7     | 3.2     | 1.3     | 0.2     | setosa  |
| 6.7     | 3.0     | 5.2     | 2.3     | virginica |
| 6.3     | 2.5     | 5.0     | 1.9     | virginica |
| 6.5     | 3.0     | 5.2     | 2.0     | virginica |
/// caption
Sample of the Iris dataset, containing features such as sepal length, sepal width, petal length, and petal width, along with the class of the flower. The dataset is widely used for classification tasks in machine learning.
///

Below there is a code snippet that loads the Iris dataset using the `pandas` library and visualizes it using `matplotlib`. The dataset is loaded from a CSV file, and the features are plotted in a scatter plot, with different colors representing the different classes of flowers.

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/classes/concepts/data/iris_data.py"
```

Also, the dataset can be visualized using the `seaborn` library, which provides a high-level interface for drawing attractive statistical graphics:

```python exec="1" html="1"
--8<-- "docs/classes/concepts/data/iris_visualization.py"
```
/// caption
Dataset visualization of the Iris dataset using the `seaborn` library. The scatter plot shows the relationship between the features of the flowers, with different colors representing the different classes. The diagonal plots show the distribution of each feature, allowing for a better understanding of the data.
///

In this visualization, each feature is represented by an axis, and the flowers are plotted in a multidimensional space. The colors represent the different classes of flowers, allowing for the identification of patterns and separations between the classes. Note that for some configurations, such as petal length vs petal width, the classes are well separated, while in others, such as sepal length vs sepal width, the classes overlap.

!!! quote "Real World"

    The Iris dataset is a classic example of a dataset used in machine learning, particularly for classification tasks. It is simple enough to be easily understood, but also presents interesting challenges for more complex models. The dataset is widely used in educational contexts to teach concepts of machine learning and data analysis.

    One can imagine that in more complex problems, such as image recognition or natural language processing, the data can be much more complex and challenging. Not allowing for a clear visualization of the spatial distribution of features. However, the fundamental principles of machine learning remain the same: understanding the data, properly preprocessing it, and choosing the right model for the task.

### **Other Datasets**

Data distribution is a crucial aspect of machine learning, as it directly affects the model's ability to learn and generalize. Usually, the nature of the data can be visualized in scatter plots, histograms, or boxplots, allowing for the identification of patterns, trends, and anomalies in the data - of course, when the data has a low number of dimensions (2 or 3).

Illustrations of some distributions with only two dimensions are presented below:

```python exec="1" html="1"
--8<-- "docs/classes/concepts/data/distributions.py"
```
/// caption
Data distributions in two dimensions in different spatial formats. For each surface, the separation between classes is made based on the characteristics of the data. The distribution of the data can affect the model's ability to learn and generalize.
///

The figure above presents four different data distributions in two dimensions, each with its own spatial characteristics. The separation between classes is made based on the characteristics of the data, and the distribution of the data can affect the model's ability to learn and generalize. In general, the function of a machine learning technique is to find a separation between classes in order to maximize the model's accuracy.


## Summary

Data is the foundation of any machine learning model. The quality, quantity, and nature of the available data are critical to the model's success. It is important to understand how the data is structured, processed, and used to train models, as well as to consider the volume of data and the balance of classes.

In addition, it is essential to perform proper data preprocessing, which may include cleaning, transformation, and normalization, to ensure that models can learn effectively and make accurate predictions.

The great challenge in machine learning is to seek the best separation between classes in order to maximize the model's accuracy. This involves not only the choice of algorithm but also a deep understanding of the data and the relationships between variables.



[^1]: [One-Hot Encoding - Wikipedia](https://en.wikipedia.org/wiki/One-hot){:target="_blank"}

[^2]: [Label Encoding - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html){:target="_blank"}

[^3]: Fisher, R. A.. 1936. Iris. UCI Machine Learning Repository.
[https://doi.org/10.24432/C56C76.](https://doi.org/10.24432/C56C76){:target="_blank"}

[^4]: [Iris Dataset - Wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set){:target="_blank"}

[^5]: Richard O. Duda, Peter E. Hart, and David G. Stork. 2000. [Pattern Classification (2nd Edition)](https://dl.acm.org/doi/book/10.5555/954544){:target="_blank"}. Wiley-Interscience, USA.

[^6]: [Data Augmentation - Wikipedia](https://en.wikipedia.org/wiki/Data_augmentation){:target="_blank"}