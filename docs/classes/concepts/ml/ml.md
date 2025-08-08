Artificial Intelligence (AI) is a broad field that encompasses various approaches and techniques for creating intelligent systems capable of performing tasks that typically require human intelligence. These tasks include reasoning, learning, perception, and decision-making.

AI can be categorized into three main paradigms, each with its own strengths and weaknesses: Symbolic AI, Connectionist AI, and Neuro-Symbolic AI. Each of these paradigms has its own strengths and weaknesses, and they are often used in different contexts depending on the problem being addressed.

## AI Paradigms
| Paradigm          | Description                                                                                   |
|-------------------|-----------------------------------------------------------------------------------------------|
| Symbolic AI       | Focuses on high-level reasoning and knowledge representation using symbols and rules. It excels in tasks requiring logical reasoning, such as theorem proving and expert systems. However, it struggles with perception and learning from raw data. Examples include logic-based systems, expert systems, and knowledge graphs. |
| Connectionist AI  | Based on artificial neural networks (ANNs), it excels in pattern recognition, learning from large datasets, and handling noisy data. It is particularly effective in tasks like image and speech recognition. However, it often lacks interpretability and struggles with reasoning tasks. Examples include convolutional neural networks (CNNs), recurrent neural networks (RNNs), and transformers. |
| Neuro-Symbolic AI | Combines the strengths of both symbolic and connectionist AI, aiming to create systems that can reason about complex problems while also learning from data. It leverages symbolic reasoning capabilities alongside neural networks to enhance interpretability and reasoning abilities. Examples include neuro-symbolic systems that integrate symbolic logic with neural networks, such as knowledge-augmented language models and graph neural networks. |

```python exec="on" html="1"
--8<-- "docs/classes/concepts/ml/relations.py"
```

Neuro-Symbolic AI combines symbolic reasoning with neural networks, leveraging the strengths of both approaches. It aims to create systems that can reason about complex problems while also learning from data.

This approach is particularly useful in tasks that require both high-level reasoning and the ability to learn from raw data, such as natural language understanding and complex decision-making.

There are several approaches to implementing AI. Machine learning (ML) is one of the most common methods, where algorithms learn from data to make predictions or decisions. Neural networks, a subset of ML, are inspired by the structure and function of the human brain and are particularly effective in tasks like image and speech recognition. Deep learning, a more advanced form of neural networks, uses multiple layers of processing to extract complex patterns from large datasets.

```python exec="on" html="1"
--8<-- "docs/classes/concepts/ml/hierarchical.py"
```



Artificial Neural Networks (ANNs), or simply **neural networks**, are computational models inspired by the structure and function of biological neural networks. They consist of interconnected nodes (neurons) that process information in a manner similar to the way neurons in the human brain operate. ANNs are capable of learning from data, making them powerful tools for various tasks such as image recognition, natural language processing, and decision-making.

Neural networks are the backbone of many modern AI applications, enabling machines to learn from experience and improve their performance over time. They are particularly effective in handling complex patterns and large datasets, making them suitable for a wide range of applications, from computer vision to speech recognition.

## Milestones

[timeline left alternate(./docs/classes/concepts/ml/timeline.json)]


## Machine Learning

In the context of AI, machine learning (ML) techniques are used to enable systems to learn from data and improve their performance over time without being explicitly programmed. These techniques allow AI systems to adapt and generalize from examples, making them capable of handling a wide range of tasks, from image recognition to natural language processing.

The techniques are often split into two main categories: **supervised learning** and **unsupervised learning**.

!!! info "Supervised Learning"
    
    **Supervised learning** involves training a model on labeled data, where the input data is paired with the correct output. This allows the model to learn patterns and make predictions based on new, unseen data.
    
    This approach is particularly effective when there is a clear relationship between the input features and the output labels, allowing the model to generalize from the training data to make accurate predictions on new data. Examples include classification tasks (e.g., identifying objects in images) and regression tasks (e.g., predicting house prices based on features).

!!! info "Unsupervised Learning"
    
    **Unsupervised learning**, on the other hand, involves training a model on unlabeled data, where the model must find patterns and relationships within the data without explicit guidance.
    
    This approach is useful for discovering hidden structures in data, such as clusters or groups, without prior knowledge of the labels. It is often used in exploratory data analysis and feature extraction. Examples include clustering tasks (e.g., grouping similar documents) and dimensionality reduction tasks (e.g., reducing the number of features in a dataset while preserving important information).

There are also **semi-supervised learning** techniques, which combine both labeled and unlabeled data to improve model performance. This approach is particularly useful when labeled data is scarce or expensive to obtain, allowing the model to leverage the abundance of unlabeled data to enhance its learning.

Also, there are **reinforcement learning** techniques, where an agent learns to make decisions by interacting with an environment and receiving feedback in the form of rewards or penalties. This approach is particularly effective for tasks that involve sequential decision-making, such as game playing or robotic control.

Machine learning techniques address a wide range of problems, primarily through **classification** and **regression**, which are core supervised learning tasks. Classification involves predicting discrete labels or categories based on input features, while regression focuses on predicting continuous values. These approaches are extensively applied across domains such as image recognition, natural language processing, and time series forecasting. However, machine learning also includes other techniques like clustering, dimensionality reduction, reinforcement learning, and anomaly detection, expanding its applicability to diverse challenges.

Few examples of machine learning techniques include:

| Technique          | Description                                                                                   |
|-------------------|-----------------------------------------------------------------------------------------------|
| Decision Trees    | A tree-like model used for classification and regression tasks, where each internal node represents a feature, each branch represents a decision rule, and each leaf node represents an outcome.
| Random Forest     | An ensemble method that combines multiple decision trees to improve accuracy and reduce overfitting. It works by training multiple decision trees on different subsets of the data and averaging their predictions.
| Support Vector Machines (SVM) | A supervised learning algorithm that finds the optimal hyperplane to separate different classes in the feature space. It is effective for high-dimensional data and can handle both linear and non-linear classification tasks.
| K-Nearest Neighbors (KNN) | A simple algorithm that classifies new instances based on the majority class of their k-nearest neighbors in the feature space. It is a non-parametric method that can be used for both classification and regression tasks. |
| Naive Bayes       | A probabilistic classifier based on Bayes' theorem, assuming independence between features. It is particularly effective for text classification tasks, such as spam detection and sentiment analysis. |
| Linear Regression | A statistical method used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data. It is commonly used for predicting continuous outcomes based on input features. |
| Logistic Regression | A statistical method used for binary classification tasks, where the output is a probability that can be mapped to two classes. It models the relationship between input features and the log-odds of the outcome using a logistic function. |
| K-Means Clustering | An unsupervised learning algorithm that partitions data into k clusters based on feature similarity. It iteratively assigns data points to the nearest cluster centroid and updates the centroids until convergence. |
| Principal Component Analysis (PCA) | A dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving important features. It identifies the principal components that capture the most variance in the data, making it useful for visualization and feature extraction. |
| Gradient Boosting | An ensemble learning technique that builds a series of weak learners (usually decision trees) in a sequential manner, where each new learner corrects the errors of the previous ones. It is effective for both classification and regression tasks and is widely used in machine learning competitions. |

## Neural Networks

Neural networks are a class of machine learning models inspired by the structure and function of the human brain. They consist of interconnected nodes (neurons) organized in layers, where each connection has an associated weight that is adjusted during training. Neural networks are particularly effective for tasks involving complex patterns, such as image and speech recognition.

Neural networks can be categorized into several types, including: 

- **Feedforward Neural Networks (FNNs)**: The simplest type of neural network where information flows in one direction, from input to output, without cycles. They are commonly used for tasks like classification and regression.
- **Convolutional Neural Networks (CNNs)**: Specialized neural networks designed for processing grid-like data, such as images. They use convolutional layers to automatically learn spatial hierarchies of features, making them highly effective for image recognition tasks.
- **Recurrent Neural Networks (RNNs)**: Neural networks designed for sequential data, such as time series or natural language. They have connections that loop back on themselves, allowing them to maintain a memory of previous inputs. This makes them suitable for tasks like language modeling and speech recognition.
- **Transformers**: A type of neural network architecture that uses self-attention mechanisms to process sequences of data. They have revolutionized natural language processing tasks, enabling models like BERT and GPT to achieve state-of-the-art performance in various language understanding tasks.

## Deep Learning

Deep learning is a subset of machine learning that focuses on using deep neural networks with many layers to learn complex representations of data. It has achieved remarkable success in various domains, including computer vision, natural language processing, and speech recognition. Deep learning models are capable of automatically learning hierarchical features from raw data, eliminating the need for manual feature engineering. This has led to significant advancements in AI applications, enabling systems to perform tasks that were previously considered challenging or impossible.

---

## Additional Resources

<iframe width="100%" height="470" src="https://www.youtube.com/embed/21EiKfQYZXc" allowfullscreen></iframe>






[^1]: [Wiki - Neuro-Symbolic AI](https://en.wikipedia.org/wiki/Neuro-symbolic_AI){target='_blank'}

[^2]: 2020, Forbes - [Symbolism Versus Connectionism In AI: Is There A Third Way?](https://www.forbes.com/councils/forbestechcouncil/2020/09/01/symbolism-versus-connectionism-in-ai-is-there-a-third-way/){target='_blank'}

[^3]: Garcez, A.d., Lamb, L.C. Neurosymbolic AI: the 3rd wave. Artif Intell Rev 56, 12387–12406 (2023).
[doi.org/10.1007/s10462-023-10448-w](https://doi.org/10.1007/s10462-023-10448-w){target='_blank'}
[:octicons-download-24:](https://arxiv.org/abs/2012.05876){target='_blank'}

[^4]: **Hodgkin–Huxley model.**
Alan Hodgkin and Andrew Huxley develop a mathematical model of the action potential in neurons, describing how neurons transmit signals through electrical impulses. This model is foundational for understanding neural dynamics and influences the development of artificial neural networks.
*Hodgkin, A. L., Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve.*
[:octicons-book-24:](https://doi.org/10.1113/jphysiol.1952.sp004764){target='_blank'}
[:material-wikipedia:](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model){target='_blank'}
[:octicons-download-24:](https://www.its.caltech.edu/~jkenny/nb250c/papers/hodgkin_52_5.pdf){target='_blank'} [:medal:](https://www.nobelprize.org/prizes/medicine/1963/summary/){target='_blank'}.

[^5]: **Visual Cortex and Monocular Deprivation.**
David H. Hubel and Torsten N. Wiesel conduct pioneering research on the visual cortex of cats, demonstrating how visual experience shapes neural development. Their work on monocular deprivation shows that depriving one eye of visual input during a critical period leads to permanent changes in the visual cortex, highlighting the importance of experience in neural plasticity.
*Hubel, D. H., & Wiesel, T. N. (1963). Effects of monocular deprivation in kittens.*
[:octicons-book-24:](https://doi.org/10.1007/bf00348878){target='_blank'}
[:octicons-download-24:](https://cw.fel.cvut.cz/b241/_media/courses/a6m33ksy/hubel-wiesel-1964-kittens.pdf){target='_blank'}
[:medal:](https://www.nobelprize.org/prizes/medicine/1981/summary/){target='_blank'}.

[^6]: **Neocognitron.** Kunihiko Fukushima develops the Neocognitron, an early convolutional neural network (CNN) model that mimics the hierarchical structure of the visual cortex. This model is a precursor to modern CNNs and demonstrates the potential of hierarchical feature extraction in image recognition tasks.
*Fukushima, K. (1980). Neocognitron: A new algorithm for pattern recognition tolerant of deformations and shifts in position.*
[:octicons-book-24:](https://doi.org/10.1007/BF00344251){target='_blank'}
[:material-wikipedia:](https://en.wikipedia.org/wiki/Neocognitron){target='_blank'}
[:octicons-download-24:](https://www.cs.princeton.edu/courses/archive/spr08/cos598B/Readings/Fukushima1980.pdf){target='_blank'}.

[^7]: **Hopfield Networks.**
John Hopfield introduces Hopfield networks, a type of recurrent neural network that can serve as associative memory systems. These networks are capable of storing and recalling patterns, laying the groundwork for later developments in neural network architectures.
*Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities.*
[:octicons-book-24:](https://doi.org/10.1073/pnas.79.8.2554){target='_blank'}
[:material-wikipedia:](https://en.wikipedia.org/wiki/Hopfield_network){target='_blank'}
[:octicons-download-24:](https://www.dna.caltech.edu/courses/cs191/paperscs191/Hopfield82.pdf){target='_blank'} [:medal:](https://www.nobelprize.org/prizes/physics/2024/summary/){target='_blank'}.

[^8]: **Self-Organizing Maps (SOM).**
Teuvo Kohonen develops Self-Organizing Maps, a type of unsupervised learning algorithm that maps high-dimensional data onto a lower-dimensional grid. SOMs are used for clustering and visualization of complex data, providing insights into the structure of the data.
*Kohonen, T. (1982). Self-organized formation of topologically correct feature maps.*
[:octicons-book-24:](https://doi.org/10.1007/BF00337288){target='_blank'}
[:material-wikipedia:](https://en.wikipedia.org/wiki/Self-organizing_map){target='_blank'}
[:octicons-download-24:](https://tcosmo.github.io/assets/soms/doc/kohonen1982.pdf){target='_blank'}. 

[^9]: **Long Short-Term Memory (LSTM) Networks.**
Sepp Hochreiter and Jürgen Schmidhuber introduce LSTM networks, a type of recurrent neural network designed to learn long-term dependencies in sequential data. This architecture addresses the vanishing gradient problem in RNNs, enabling effective modeling of long-term dependencies in sequential data.
*Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.*
[:octicons-book-24:](https://doi.org/10.1162/neco.1997.9.8.1735){target='_blank'}
[:material-wikipedia:](https://en.wikipedia.org/wiki/Long_short-term_memory){target='_blank'}
[:octicons-download-24:](https://www.bioinf.jku.at/publications/older/2604.pdf){target='_blank'}".

[^10]: **Residual Networks (ResNets).**
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun introduce Residual Networks (ResNets), a deep learning architecture that uses skip connections to allow gradients to flow more easily through deep networks. This architecture enables the training of very deep neural networks, significantly improving performance on image recognition tasks.
*He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for image recognition.*
[:octicons-book-24:](https://doi.org/10.1109/CVPR.2016.90){target='_blank'}
[:material-wikipedia:](https://en.wikipedia.org/wiki/Residual_network){target='_blank'}
[:octicons-download-24:](https://arxiv.org/pdf/1512.03385){target='_blank'}
