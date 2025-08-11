from io import StringIO

import matplotlib.pyplot as plt

salmon = [
    [60, 6], [78, 7], [90, 5.2], [71, 9], [64, 6]
]
seabass = [
    [45, 5], [80, 3], [58, 2], [63, 6.8], [50, 4]
]

zeros = [0] * len(salmon)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

for i in range(len(ax)):
    ax[i].plot(

        [fish[0] for fish in salmon],
        [fish[1] for fish in salmon],
        'o',

        [fish[0] for fish in seabass],
        [fish[1] for fish in seabass],
        'o',

        [40, 100], [7, 3], '--m', lw=3

    )
    ax[i].set_xlim(40, 100)
    ax[i].set_ylim(0, 10)
    ax[i].set_xlabel('Length')
    ax[i].set_ylabel('Brightness')

ax[1].plot(70, 4, 'xg', markersize=12)

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())