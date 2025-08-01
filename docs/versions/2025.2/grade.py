import matplotlib.pyplot as plt
from io import StringIO

fig, ax = plt.subplots(1, 1)

fig.set_size_inches(10, 5)

ax.pie(
    [20, 20, 20, 10, 10, 10, 10],
    labels=["Exercícios", "Parcial", "Final", "Integrativa", "Projeto I", "Projeto II", "Integrador"],
    explode=[0, 0, 0, 0, 0, 0, 0],
    autopct='%1.0f%%',
    startangle=90)
ax.title.set_text("Composição Final")

plt.tight_layout()

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
