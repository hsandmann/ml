import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles
from io import StringIO

# Create a Venn diagram with two sets (Symbolic AI and Connectionist AI)
plt.figure(figsize=(9, 9))

# Define subsets for Symbolic AI, Connectionist AI, and their overlap
venn_diagram = venn2(subsets=(7, 8, 4), 
                     set_labels=('Symbolic AI', 'Connectionist AI'), 
                     alpha=0.5)

# Customize labels for each region
venn_diagram.get_label_by_id('10').set_text('\n'.join([
    'Logic-Based Systems',
    'Propositional Logic',
    'First-Order Logic',
    'Expert Systems',
    'Knowledge Graphs',
    'Rule-Based Systems',
    'Ontologies'
]))  # Symbolic AI only

venn_diagram.get_label_by_id('01').set_text('\n'.join([
    'Artificial Neural Networks',
    'Convolutional Neural Networks',
    'Recurrent Neural Networks',
    'Transformers',
    'Generative Adversarial Networks',
    'Diffusion Models',
    'Spiking Neural Networks',
    'Deep Belief Networks',
    'Autoencoders'
]))  # Connectionist AI only

venn_diagram.get_label_by_id('11').set_text('\n'.join([
    'Reinforcement Learning\nw/ Symbolic Planning',
    'Knowledge-Augmented LLMs',
    'Graph Neural Networks'
]))  # Overlap

# Adjust font size for readability
for label in venn_diagram.subset_labels:
    if label:
        label.set_fontsize(10)

# Add circles outline for clarity
venn2_circles(subsets=(7, 8, 4), linestyle='solid', linewidth=1)

# Add title
plt.title("Venn Diagram of Symbolic AI, Connectionist AI, and Overlap", fontsize=14)

# Add annotation for Other Techniques outside the Venn diagram
plt.text(0.5, 0, 
         'Other Techniques (Outside Categories):\n' + 
         '\n'.join([
             '- Evolutionary Algorithms',
             '- Swarm Intelligence',
             '- Bayesian Networks',
             '- Markov Models',
             '- Fuzzy Logic',
             '- Decision Trees',
             '- Support Vector Machines'
         ]),
         fontsize=10, ha='center', va='top', transform=plt.gca().transAxes)

# Add an arrow pointing to the intersection with label "Neuro-Symbolic AI"
plt.annotate('Neuro-Symbolic AI', 
             xy=(0.0, -0.2),  # Approximate center of overlap
             xytext=(0.0, -0.5),  # Position of text
             fontsize=12, 
             fontweight='regular',
             ha='center',
             va='center',
            #  bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black'))

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())