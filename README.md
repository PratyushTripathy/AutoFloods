# AutoFloods: India Flood Monitoring and Impact Assessment Portal
This repository contains the source code of the flood mapping piece of the India Flood Mapping and Impact Assessment Portal.

Data is pulled from the Microsoft Planetary Computer (MPC) Spatio Temporal Asset Catalog (STAC) API. The workflow is capable of mapping floods from Sentinel-1 SAR data for large areas. To keep the scaling up of the flood mapping algorithm stable, this workflow relies of processing large areas by processing them in smaller fragments. Flood detection is pluggable (`autofloods.detectors`): `ZScoreDetector` (default) compares each scene against a dry-season baseline, and `OtsuDetector` classifies a single scene against its own histogram, needing no dry-season baseline at all.

**Suggested citation**<br/>
The algorithm of this flood mapping workflow is an extension of our previous work on open-source flood mapping tools. If you are using this repository (or part of it) for your own work, consider citing the following:

Tripathy, P. and Malladi, T. (2021). Global Flood Mapper: Democratising open EO resources for flood mapping. _EGU General Assembly 2021_, online, 19–30 Apr 2021, EGU21-16194, https://doi.org/10.5194/egusphere-egu21-16194<br/>

Tripathy, P. & Malladi, T. (2022). Global Flood Mapper: a novel Google Earth Engine application for rapid flood mapping using Sentinel-1 SAR. _Natural Hazards_. https://doi.org/10.1007/s11069-022-05428-2<br/>

A novel workflow for large scale automated flood mapping. (Manuscript in preparation)

**Affiliation**<br/>
MapSolve AI Pvt. Ltd., India.<br/>
University of California, Santa Barbara, United States.

**Funding**<br/>
Pratyush Tripathy was supported by a NASA FINESST award (80NSSC25K0392) for this work.
