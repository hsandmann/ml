import matplotlib.pyplot as plt
from io import StringIO

fig, ax = plt.subplots(1, 1)

fig.set_size_inches(8, 8)

size = 0.3

outer_colors = ['aliceblue', 'lightcyan']
inner_colors = [
    'cornflowerblue', 'skyblue', 'powderblue', 'lightsteelblue',
    'turquoise', 'mediumturquoise', 'aquamarine', 'white'
]

ax.pie(
    [70, 30],
    radius=1-size,
    colors=outer_colors,
    wedgeprops=dict(width=size, edgecolor='w'),
    textprops={
        'fontsize': 12
    }
)

ax.pie(
    [20, 20, 20, 10, 10, 10, 10],
    labels=[
        'Exercícios\n20%', 'Parcial\n20%', 'Final\n20%', 'Integrativa\n10%',
        'Projeto I\n10%', 'Projeto II\n10%', 'Integrado\n10%'
    ],
    radius=1,
    colors=inner_colors,
    wedgeprops=dict(width=size, edgecolor='w'),
    textprops={
        'fontsize': 12
    }
)

ax.set(aspect="equal")
ax.text(-.36, .35, "Individual\n70%", color='black', fontsize=12, ha='center')
ax.text(.3, -.48, "Grupo\n30%", color='black', fontsize=12, ha='center')


# ax[1].pie(
#     [20, 20, 20, 10, 10, 10, 10],
#     labels=[
#         'Exercícios', 'Parcial', 'Final', 'Integrativa',
#         'Projeto I', 'Projeto II', 'Integrado'
#     ],
#     autopct='%1.0f%%',
#     colors=inner_colors,
# )

ax.set(aspect="equal")

plt.tight_layout()

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
