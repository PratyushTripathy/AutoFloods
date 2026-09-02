![AutoFloods](autofloods_logo.png)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22258069.svg)](https://doi.org/10.5281/zenodo.22258069)

AutoFloods is a Python package for automated flood mapping at scale from Sentinel-1 SAR imagery. It maps flooded areas over large regions by splitting them into tiles and processing each one independently, so the same workflow scales from a single tile to a country-sized run.

Both the data source and the flood-detection method are pluggable. `MPCSource` (Microsoft Planetary Computer) and `OPERASource` (NASA OPERA RTC-S1) are interchangeable STAC data backends; `ZScoreDetector` (default) flags anomalies against a dry-season baseline, and `OtsuDetector` classifies each scene against its own histogram with no baseline required. New sources or detectors can be added without changing the rest of the pipeline.

Full documentation, including setup and usage, is available at the docs site (TBD).

**Suggested citation**<br/>
The algorithm of this flood mapping workflow is an extension of our previous work on open-source flood mapping tools. If you are using this repository (or part of it) for your own work, consider citing the following:

Tripathy, P., & Malladi, T. (2022). Global Flood Mapper: a novel Google Earth Engine application for rapid flood mapping using Sentinel-1 SAR. _Natural Hazards_, 114(2), 1341-1363. https://doi.org/10.1007/s11069-022-05428-2<br/>

Tripathy, P., Malladi, T., Balakrishnan, K., & Parath, N. (n.d.). _AutoFloods: A Python package for automated flood mapping at scale using Sentinel-1 SAR_ [Manuscript in preparation].

**Affiliation**<br/>
University of California, Santa Barbara, United States.

**Funding**<br/>
Pratyush Tripathy was supported by a NASA FINESST award (80NSSC25K0392) for this work.
