from spacy import displacy
import cairosvg

def spacy2pdf(doc, filename = 'spacy2pdf_output.pdf'):
    # Render the dependency parse as SVG
    svg = displacy.render(doc, style='dep', jupyter=False)

    # Save the SVG to a file
    svg_path = filename.replace('.pdf', '.svg')
    with open(svg_path, "w", encoding="utf-8") as file:
        file.write(svg)

    # Convert SVG to PDF
    pdf_path = f"{filename}.pdf"
    cairosvg.svg2pdf(url=svg_path, write_to=filename)

    print(f"Dependency graph saved as PDF at {pdf_path}")



# # Function to visualize the tree
# def plot_tree(G, title="Tree Structure", highlight_nodes=None, highlight_edges=None, index = 'root'):
#     pos = graphviz_layout(G, prog="twopi")
#     nx.draw(G, pos)
#     plt.savefig(f'./trees/tree_{index}.pdf', format = 'pdf')

# # Plot the full tree
# plot_tree(G, title="Full Tree Structure")

# # Dictionary to store all possible sub-branches for each node
# sub_branches = {}

# # Function to get subgraph rooted at a given node
# def get_subgraph(G, root):
#     descendants = nx.descendants(G, root) | {root}  # include root itself
#     subgraph = G.subgraph(descendants).copy()  # Create a subgraph with these nodes
#     return subgraph

# # Generate and visualize all possible sub-branches
# for i, node in enumerate(G.nodes()):
#     # Get the subgraph rooted at the current node
#     subgraph = get_subgraph(G, node)
#     sub_branches[node] = subgraph
    
#     # Extract the nodes and edges for visualization
#     highlight_nodes = list(subgraph.nodes())
#     highlight_edges = list(subgraph.edges())
    
#     # Plot each sub-branch
#     plot_tree(G, title=f"Sub-branch rooted at {node}", highlight_nodes=highlight_nodes, highlight_edges=highlight_edges, index=i)