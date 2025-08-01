
K-Nearest Neighbors (KNN)

K-Nearest Neighbors (KNN) é um algoritmo de aprendizado de máquina usado para classificação e regressão. Ele funciona identificando os K vizinhos mais próximos de um ponto de dados e fazendo previsões com base nesses vizinhos.

O KNN é um algoritmo baseado em instâncias, o que significa que ele não faz suposições sobre a distribuição dos dados. Em vez disso, ele armazena todos os pontos de dados de treinamento e, durante a previsão, calcula a distância entre o ponto de dados de entrada e todos os pontos de dados de treinamento. Os K pontos mais próximos são então usados para fazer a previsão, geralmente por meio de votação (no caso de classificação) ou média (no caso de regressão).

Uma das principais vantagens do KNN é sua simplicidade e facilidade de implementação. No entanto, ele pode ser computacionalmente caro, especialmente com grandes conjuntos de dados, e sua performance pode ser afetada pela escolha da métrica de distância e pelo valor de K.
## Implementação do KNN
A implementação do KNN pode ser feita de várias maneiras, mas uma das mais comuns é usar a biblioteca `scikit-learn` em Python. Aqui está um exemplo básico de como implementar o KNN para classificação:

!!! example "Exercício"

    Utilizando