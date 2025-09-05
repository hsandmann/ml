import matplotlib.pyplot as plt
from io import StringIO

fig, ax = plt.subplots()

fig.set_size_inches(8, 8)

circle1 = plt.Circle((0.7, 0.7), 0.7, alpha=0.9, color='orange')
circle2 = plt.Circle((0.8, 0.6), 0.5, alpha=0.6, color='gray')
circle3 = plt.Circle((.85, .55), 0.35, alpha=0.6, color='green')
circle4 = plt.Circle((.9, .5), 0.2, alpha=0.6, color='blue')

ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)
ax.add_patch(circle4)

ax.text(.7, 1.2, 'Artificial Intelligence', horizontalalignment='center', fontsize=12, color='white')
ax.text(.8, .95, 'Machine Learning', horizontalalignment='center', fontsize=12, color='white')
ax.text(.85, .75, 'Neural Networks', horizontalalignment='center', fontsize=12, color='white')
ax.text(.9, .5, 'Deep Learning', horizontalalignment='center', fontsize=12, color='white')

ax.set_aspect(1.0)

plt.xlim(0, 1.4)
plt.ylim(0, 1.4)
plt.axis('off')

plt.title('Hierarchical Representation of AI Approaches')

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
