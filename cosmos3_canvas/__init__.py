"""The package's former name: ``renderer.backend: cosmos3_canvas:Cosmos3Renderer`` keeps working.

New configurations name ``cosmos_canvas.sim_auto:Cosmos3Renderer``.
"""

__all__ = ["Cosmos3Renderer"]


def __getattr__(name: str):
    if name == "Cosmos3Renderer":
        from cosmos_canvas.sim_auto.renderer import Cosmos3Renderer

        return Cosmos3Renderer
    raise AttributeError(name)
