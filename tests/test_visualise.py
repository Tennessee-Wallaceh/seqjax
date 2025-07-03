import graphviz
from seqjax.model.ar import AR1Target
from seqjax.model.visualise import graph_model

def test_graph_model_returns_digraph() -> None:
    model = AR1Target()
    g = graph_model(model)
    assert isinstance(g, graphviz.Digraph)
