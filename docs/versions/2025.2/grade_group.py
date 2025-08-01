import matplotlib.pyplot as plt
from io import StringIO

fig, ax = plt.subplots(1, 1)

fig.set_size_inches(10, 5)

ax.pie(
    [10, 10, 10],
    labels=["Projeto I", "Projeto II", "Integrador"],
    explode=[0, 0, 0],
    autopct='%1.0f%%',
    startangle=90)
ax.title.set_text("Composição Grupo")

plt.tight_layout()

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
