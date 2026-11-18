"""Cosmos world models as carla_canvas renderers.

Two backends, each run in the environment its model needs:

* ``cosmos_canvas.omnidreams:OmniDreamsRenderer``: OmniDreams (Cosmos 2.5,
  distilled), one camera, through FlashDreams in-process; its HD-map raster
  is drawn from the live scene for every chunk.
* ``cosmos_canvas.sim_auto:Cosmos3Renderer``: Cosmos3-Nano-Sim-Auto, cameras
  and LiDAR, through cosmos-framework.

Nothing is imported at package level: each backend pulls in its own model stack.
"""
