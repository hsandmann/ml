import matplotlib.pyplot as plt
from io import StringIO

fig, ax = plt.subplots(1, 2)

fig.set_size_inches(10, 5)

size = 0.3

cmap = plt.get_cmap("tab20c")
outer_colors = ['paleturquoise', 'mistyrose']
inner_colors = [
    'cornflowerblue', 'skyblue', 'powderblue', 'lightsteelblue',
    'salmon', 'lightcoral', 'indianred', 'white'
]

ax[0].pie(
    [70, 30],
    labels=['Individual\n70%', 'Grupo\n30%'],
    radius=1,
    colors=outer_colors,
    wedgeprops=dict(width=size, edgecolor='w')
)

ax[0].pie(
    [20, 20, 20, 10, 10, 10, 10],
    labels=[
        'Exercícios', 'Parcial', 'Final', 'Integrativa',
        'Projeto I', 'Projeto II', 'Integrado'
    ],
    radius=1-size,
    colors=inner_colors,
    wedgeprops=dict(width=size, edgecolor='w')
)

ax[0].set(aspect="equal")

ax[1].pie(
    [20, 20, 20, 10, 10, 10, 10],
    labels=[
        'Exercícios', 'Parcial', 'Final', 'Integrativa',
        'Projeto I', 'Projeto II', 'Integrado'
    ],
    autopct='%1.0f%%',
    colors=inner_colors,
)

ax[0].set(aspect="equal")

plt.tight_layout()

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
