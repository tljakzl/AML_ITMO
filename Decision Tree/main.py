
print("Hello DecisionTree!")
import graphviz
with open('tree.dot') as f:
    dot_graph = f.read()
    graph = graphviz.Source(dot_graph)
    graph.render(view=True)