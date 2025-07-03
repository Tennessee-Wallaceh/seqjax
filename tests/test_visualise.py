import graphviz
from seqjax.model.ar import AR1Target
from seqjax.model.visualise import graph_model


def test_graph_model_includes_field_nodes() -> None:
    model = AR1Target()
    g = graph_model(model)
    assert "x0_x" in g.source
    assert "y0_y" in g.source


def test_graph_model_render_called(monkeypatch) -> None:
    model = AR1Target()
    called = {}

    def fake_render(self, filename, cleanup=True, format="png"):
        called["filename"] = filename
        called["cleanup"] = cleanup
        called["format"] = format
        return filename

    monkeypatch.setattr(graphviz.Digraph, "render", fake_render)
    g = graph_model(model, render="out")
    assert isinstance(g, graphviz.Digraph)
    assert called["filename"] == "out"

def test_graph_model_returns_digraph() -> None:
    model = AR1Target()
    g = graph_model(model)
    assert isinstance(g, graphviz.Digraph)
