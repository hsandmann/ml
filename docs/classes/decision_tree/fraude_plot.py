import matplotlib.pyplot as plt
import pandas as pd

from io import StringIO

df = pd.read_csv('https://raw.githubusercontent.com/hsandmann/ml/refs/heads/main/data/fraude.csv')
fraudes = df[df['Classe'] == 'Fraude']
normais = df[df['Classe'] == 'Normal']
plt.plot(
    normais['Periodo'], normais['Valor'], '.b',
    fraudes['Periodo'], fraudes['Valor'], '.r',
)

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())