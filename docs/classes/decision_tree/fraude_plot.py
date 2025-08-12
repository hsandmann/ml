import matplotlib.pyplot as plt
import pandas as pd

from io import StringIO

plt.figure(figsize=(5, 4))

df = pd.read_csv('https://raw.githubusercontent.com/hsandmann/ml/refs/heads/main/data/fraude.csv')
fraudes = df[df['Classe'] == 'Fraude']
normais = df[df['Classe'] == 'Normal']
plt.plot(
    normais['Periodo'], normais['Valor'], 'ob',
    fraudes['Periodo'], fraudes['Valor'], 'or',
)

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())