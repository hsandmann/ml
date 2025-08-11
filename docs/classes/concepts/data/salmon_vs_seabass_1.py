from io import StringIO

import matplotlib.pyplot as plt

# features of salmons (length and lightness)
salmon = [
    [60, 6], [78, 7], [90, 5.2], [71, 9], [64, 6]
]
seabass = [
    [45, 5], [80, 3], [58, 2], [63, 6.8], [50, 4]
]

zeros = [0] * len(salmon)

# definindo o tamanho da figura
fig, ax = plt.subplots(1, 2, figsize=(7, 1))

for i in range(len(ax)):
    ax[i].set_frame_on(False)
    ax[i].yaxis.set_visible(False)
    ax[i].spines['left'].set_position('zero')
    ax[i].spines['bottom'].set_position('zero')
    ax[i].set_xlabel(['Length', 'Brightness'][i])

    ax[i].plot(
        [[40, 100], [0, 10]][i], [0, 0], 'k',
        [fish[i] for fish in salmon], zeros, 'o',
        [fish[i] for fish in seabass], zeros, 'o'
    )

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())