.. _api_ref:

.. automodule:: autofloods

.. currentmodule:: autofloods


AutoFloods API reference
======================
	
Utility
-----------------

.. autosummary::
   :toctree: generated/

    autofloods.utils.date_range
    autofloods.utils.string_to_date_range
    autofloods.utils.gpd_to_json
    autofloods.utils.search_sentinel_data
    autofloods.utils.s1item_footprint
    autofloods.utils.seggregate_sentinel_search
    autofloods.utils.export_xarray

Preprocessing
-----------------

.. autosummary::
   :toctree: generated/

    autofloods.preprocessing.read_sentinel1_stac
    autofloods.preprocessing.reproject_clip_stac
    autofloods.preprocessing.stack_images
