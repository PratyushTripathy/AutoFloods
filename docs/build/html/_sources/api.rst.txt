.. _api_ref:
.. automodule:: ifmiap

.. currentmodule:: ifmiap


ifmiap API reference
======================

Authentication of files
------------------------------

.. autosummary::
   :toctree: generated/

    ifmiap.authenticate.sign_in
    
	
Preprocessing of files
-------------------------------

.. autosummary::
   :toctree: generated/

    ifmiap.preprocessing.read_reproject
    ifmiap.preprocessing.clip_stac
    ifmiap.preprocessing.stack_images


Mapping of floods
-------------------------------

.. autosummary::
   :toctree: generated/

    ifmiap.mapfloods.anomaly_cells

Postprocessing of files
-------------------------------

.. autosummary::
   :toctree: generated/

    ifmiap.postprocessing.polygonize_flood_raster