import matplotlib.pyplot as plt
import numpy as np

def visualize_agent():
    # Define figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define node positions (x, y coordinates)
    nodes = {
        'Input': (0, 2),
        'Dynamic': (2, 2),
        'Prompt': (4, 2),
        'Response': (6, 2),
        'Output': (8, 2),
        'Memory': (4, 0)
    }
    
    # Node labels with descriptions
    node_labels = {
        'Input': 'User Input',
        'Dynamic': 'Dynamic Prompt Chain\n(Analyzes input, reconstructs prompt)',
        'Prompt': 'Prompt Abstraction Chain\n(Refines prompt with memory context)',
        'Response': 'Response Abstraction Chain\n(Refines as Sayaka Justine\'s\nMagical Girl Angel)',
        'Output': 'Final Response',
        'Memory': 'Long-Term Memory\n(JSON file storage)'
    }
    
    # Define edges (source, target, style, label)
    edges = [
        ('Input', 'Dynamic', 'solid', 'User input'),
        ('Dynamic', 'Prompt', 'solid', 'Structured prompt'),
        ('Prompt', 'Response', 'solid', 'Abstracted prompt'),
        ('Response', 'Output', 'solid', 'Final response'),
        ('Memory', 'Prompt', 'dashed', 'Memory context'),
        ('Output', 'Memory', 'dashed', 'Save prompt & response')
    ]
    
    # Plot nodes
    for node, (x, y) in nodes.items():
        ax.scatter(x, y, s=1000, color='lightblue' if node != 'Memory' else 'lightgrey', 
                   edgecolors='black', zorder=3)
        ax.text(x, y, node_labels[node], ha='center', va='center', fontsize=8, 
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    
    # Plot edges with arrows
    for src, dst, style, label in edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        # Adjust arrow start/end to avoid overlapping with node circles
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx, dy = dx / length, dy / length
            offset = 0.3  # Offset to keep arrows outside node circles
            ax.arrow(x1 + dx * offset, y1 + dy * offset, 
                     dx * (length - 2 * offset), dy * (length - 2 * offset),
                     head_width=0.1, head_length=0.2, fc='black', ec='black',
                     linestyle=style, zorder=2)
            # Place label at midpoint of edge
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.1, label, fontsize=7, ha='center')
    
    # Set plot parameters
    ax.set_xlim(-1, 9)
    ax.set_ylim(-1, 3)
    ax.set_title("Model Agent Chain Structure\n(Sayaka Justine's Magical Girl Angel Persona)", fontsize=12, pad=20)
    ax.axis('off')  # Hide axes for cleaner look
    
    # Save and show the plot
    try:
        plt.savefig('agent_chain_graph.png', bbox_inches='tight', dpi=150)
        plt.close()
        print("Graph has been generated and saved as 'agent_chain_graph.png'")
    except Exception as e:
        print(f"Error generating graph: {e}")

if __name__ == "__main__":
    visualize_agent()