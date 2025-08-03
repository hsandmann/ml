Todo os conceitos de Machine Learning são baseados em dados. A qualidade e a quantidade dos dados disponíveis são fundamentais para o sucesso de qualquer modelo de aprendizado de máquina. Neste contexto, é importante entender como os dados são estruturados, processados e utilizados para treinar modelos.

## Natureza dos Dados

Cada característica é uma variável que descreve um aspecto do dado. Por exemplo, em um conjunto de dados sobre flores, as características podem incluir o comprimento e a largura das pétalas e sépalas.

Variáveis são os atributos ou colunas de um conjunto de dados. Elas podem ser categóricas (como "cor" ou "tipo") ou numéricas (como "altura" ou "peso"). As variáveis são usadas para descrever os dados e podem ser usadas como entrada para modelos de aprendizado de máquina. **Cada variável é uma dimensão do espaço de características, e o conjunto de dados é representado como um ponto nesse espaço.**

As variáveis podem ser numéricas ou categóricas:

- **Variáveis numéricas** são aquelas que podem assumir valores contínuos, como altura ou peso;
- **Variáveis categóricas** são aquelas que assumem valores discretos, como cor ou tipo.

Dependendo do tipo de algoritmo de aprendizado de máquina, as variáveis podem ser tratadas de maneiras diferentes. Por exemplo, existem algoritmos que lidam melhor com variáveis numéricas, enquanto outros são mais adequados para variáveis categóricas. Neste contexto, é necessário converter as variáveis categóricas em um formato que os algoritmos possam entender, como usando codificação one-hot[^1] ou label encoding[^2].

Adicionalmente, as variáveis numéricas, como altura ou peso, são frequentemente normalizadas para garantir que todas as variáveis contribuam igualmente para o modelo. A normalização é uma técnica de pré-processamento que ajusta os valores das variáveis para uma escala comum, geralmente entre 0 e 1 ou -1 e 1.

Todo esse processo de preparação dos dados é crucial para garantir que os modelos de aprendizado de máquina possam aprender de maneira eficaz e fazer previsões precisas. Essa é etapa é o **pré-processamento** ou **normalização** dos dados, que envolve a limpeza, transformação e normalização dos dados antes de serem usados para treinar modelos.

## Base de Dados

Existem várias bases de dados disponíveis para treinamento e teste de modelos de aprendizado de máquina. Algumas das mais conhecidas incluem:

- **UCI Machine Learning Repository**: uma coleção de conjuntos de dados para tarefas de aprendizado de máquina, incluindo classificação, regressão e clustering.
- **Iris Dataset**: um conjunto de dados clássico usado para classificação de flores com base em características como comprimento e largura das pétalas e sépalas[^3][^4].
- **MNIST**: um conjunto de dados de imagens de dígitos manuscritos, amplamente utilizado para tarefas de reconhecimento de imagem.
- **CIFAR-10**: um conjunto de dados de imagens de objetos em 10 classes diferentes, usado para tarefas de classificação de imagens.
- **Kaggle Datasets**: uma plataforma que oferece uma ampla variedade de conjuntos de dados para diferentes tarefas de aprendizado de máquina, desde classificação de texto até reconhecimento de imagem.

Problemas comuns em conjuntos de dados incluem:

- **Dados ausentes**: valores que não estão disponíveis para algumas variáveis;
- **Dados duplicados**: registros que aparecem mais de uma vez no conjunto de dados;
- **Dados ruidosos**: valores que são inconsistentes ou incorretos.
- **Dados desbalanceados**: quando uma classe é muito mais frequente do que outra, o que pode levar a um modelo enviesado.
- **Dados inconsistentes**: quando os dados não seguem um padrão ou formato consistente, dificultando a análise e o treinamento do modelo.
- **Dados irrelevantes**: variáveis que não contribuem para a tarefa de aprendizado de máquina e podem prejudicar o desempenho do modelo.

Para lidar com esses problemas, é comum realizar um processo de limpeza e pré-processamento dos dados, que pode incluir:

- **Remoção de dados ausentes**: excluir registros com valores ausentes ou imputar valores com base em outras observações.
- **Remoção de duplicatas**: identificar e remover registros duplicados.
- **Tratamento de dados ruidosos**: aplicar técnicas de suavização ou filtragem para reduzir o ruído nos dados.
- **Balanceamento de classes**: técnicas como subamostragem ou superamostragem para lidar com classes desbalanceadas.
- **Normalização**: ajustar os valores das variáveis para uma escala comum, garantindo que todas as variáveis contribuam igualmente para o modelo.
- **Transformação de variáveis**: aplicar técnicas como logaritmo, raiz quadrada ou Box-Cox para transformar variáveis não lineares em lineares.
- **Codificação de variáveis categóricas**: converter variáveis categóricas em um formato que os algoritmos possam entender, como usando codificação one-hot ou label encoding.

Além disso, é importante considerar a ordem dos dados, especialmente em problemas de séries temporais, onde a sequência dos dados é crucial para a análise e modelagem.

## Volume de Dados

O volume de dados refere-se à quantidade de dados disponíveis para treinamento e teste de modelos de aprendizado de máquina. Quanto maior o volume de dados, mais informações o modelo pode aprender, o que geralmente resulta em melhor desempenho. No entanto, também é importante considerar a qualidade dos dados, pois dados ruidosos ou irrelevantes podem prejudicar o desempenho do modelo.

Além disso, é importante considerar o balanceamento das classes, especialmente em problemas de classificação. O balanceamento de classes refere-se à distribuição equitativa das classes no conjunto de dados. **Se uma classe for muito mais frequente do que outra, isso pode levar a um modelo enviesado**, que tende a prever a classe majoritária.

Para modelos de aprendizado supervisionado, é essencial ter um conjunto de dados rotulado, onde cada exemplo tem uma entrada (características) e uma saída (rótulo). Isso permite que o modelo aprenda a mapear as entradas para as saídas corretas.

Ainda, os dados podem ser classificados em três categorias principais:

| Natureza dos Dados | Descrição |
|--------------------|-----------|
| **Treinamento** | Usados para treinar o modelo, permitindo que ele aprenda os padrões e relações entre as características e os rótulos. |
| **Teste** | Usados para ajustar os hiperparâmetros do modelo e evitar o overfitting, garantindo que ele generalize bem para novos exemplos. |
| **Validação** | Usados para avaliar o desempenho do modelo em dados não vistos, garantindo que ele generalize bem para novos exemplos. |

---

### Exemplos

#### **Salmão vs Robalo**

Um exemplo de conjunto de dados fictício sobre salmão e robalo, onde cada registro é rotulado como "salmão" ou "robalo". O objetivo é entender melhor como os dados podem ser utilizados para diferenciar as duas espécies. Nesse contexto, as características podem incluir, por exemplo: tamanho e brilho[^5].



#### Iris Dataset

[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris): o Iris Dataset é um conjunto de dados clássico usado para classificação de flores. Ele contém 150 amostras de três espécies diferentes de flores Iris (Iris setosa, Iris versicolor e Iris virginica), com quatro características: comprimento e largura das pétalas e sépalas.

![](iris_dataset.png)

O conjunto de dados é amplamente utilizado para demonstrar algoritmos de aprendizado de máquina, especialmente em tarefas de classificação. Ele é simples o suficiente para ser facilmente compreendido, mas também apresenta desafios interessantes para modelos mais complexos.

Uma amostra do conjunto de dados Iris é apresentada na tabela abaixo:

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
Amostra do conjunto de dados Iris, contendo características como comprimento e largura das pétalas e sépalas, além da classe da flor.
///

Abaixo está um exemplo de como carregar o conjunto de dados Iris usando Python:

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/classroom/concepts/iris_data.py"
```

Também é possível visualizar o conjunto de dados usando bibliotecas como `matplotlib` e `seaborn`. A seguir, um exemplo de visualização do conjunto de dados Iris:

```python exec="1" html="1"
--8<-- "docs/classroom/concepts/iris_visualization.py"
```
/// caption
Visualização do conjunto de dados Iris, mostrando a relação entre as características das flores e suas classes. Cada característica, representada por um eixo, é confrontada com as outras, permitindo identificar padrões e separações entre as classes.
///


## Resumo

Os dados são a base de qualquer modelo de aprendizado de máquina. A qualidade, quantidade e natureza dos dados disponíveis são fundamentais para o sucesso do modelo. É importante entender como os dados são estruturados, processados e utilizados para treinar modelos, além de considerar o volume de dados e o balanceamento das classes.

Além disso, é essencial realizar um pré-processamento adequado dos dados, que pode incluir limpeza, transformação e normalização, para garantir que os modelos possam aprender de maneira eficaz e fazer previsões precisas.

O grande desafio em aprendizado de máquina é buscar a melhor separação entre as classes, a fim de maximizar a precisão do modelo. Isso envolve não apenas a escolha do algoritmo, mas também o entendimento profundo dos dados e das relações entre as variáveis.



[^1]: [One-Hot Encoding - Wikipedia](https://en.wikipedia.org/wiki/One-hot){:target="_blank"}

[^2]: [Label Encoding - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html){:target="_blank"}

[^3]: Fisher, R. A.. 1936. Iris. UCI Machine Learning Repository.
[https://doi.org/10.24432/C56C76.](https://doi.org/10.24432/C56C76){:target="_blank"}

[^4]: [Iris Dataset - Wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set){:target="_blank"}

[^5]: Richard O. Duda, Peter E. Hart, and David G. Stork. 2000. [Pattern Classification (2nd Edition)](https://dl.acm.org/doi/book/10.5555/954544){:target="_blank"}. Wiley-Interscience, USA.