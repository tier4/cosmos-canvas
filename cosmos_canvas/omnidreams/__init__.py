"""OmniDreams (Cosmos 2.5) as a carla_canvas backend.

``renderer.backend: cosmos_canvas.omnidreams:OmniDreamsRenderer``.

The renderer imports carla_canvas; ``cosmos_canvas.omnidreams.stream`` (the model
alone) and ``ludus.prepare_clipgt`` (the map) do not.
"""

__all__ = ["OmniDreamsRenderer"]


def __getattr__(name: str):
    if name == "OmniDreamsRenderer":
        from .renderer import OmniDreamsRenderer

        return OmniDreamsRenderer
    raise AttributeError(name)
