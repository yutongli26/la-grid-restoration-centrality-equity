Robustness and Post-Earthquake Recovery of the Los Angeles Power Grid at Census-Tract Scale

Team: Yinchen Yi, Yutong Li

A census-tract–scale modeling pipeline to (i) simulate earthquake-induced substation damage, (ii) propagate disruption to tract-level service via a tract–substation mapping matrix, and (iii) compare post-earthquake repair prioritization strategies under three objectives:

Network robustness (connectivity / fragmentation proxies)

Population service restoration

Equity-aware restoration across census tracts (e.g., SVI-weighted objectives)

The pipeline supports both an unconstrained baseline and a logistics-aware restoration setting (multi-crew, multi-depot, travel-time constraints). We further perform PCA + K-means to group census tracts into resilience typologies using simulated outage/recovery metrics and socioeconomic indicators.

Scope note: robustness metrics here are connectivity-based (e.g., largest connected component) and do not enforce power-flow feasibility unless operational constraints are added.

Key features

Scenario-based stress testing using historical ShakeMap PGA fields (e.g., 1933 Long Beach, 1971 San Fernando, 1994 Northridge)

Probabilistic substation damage-state modeling using lognormal fragility with Monte Carlo sampling

Tract-level service/outage estimation through a tract–substation weighting matrix (W)

Restoration simulation under:

Baseline (no crew constraints)

Logistics-aware scheduling (multi-crew, multi-depot, travel-time constraints)

Strategy comparison (e.g., Random / Centrality-first / Population-impact-first / Hospital-first) using service and network metrics

Tract resilience typology via PCA + K-means (simulated + socioeconomic features)

Repository structure

Topology_and_Weight.py — Build transmission topology and tract–substation mapping matrix (W)

IDW.py — PGA interpolation (IDW) utilities (if needed)

build_travel_matrices_osm.py — Travel-time matrices from road network (OSM-based, optional)

C257H_Project_Main.py — Main pipeline (simulation + restoration + KPIs + typology)

Project_Visualizer.py — Mapping and figure generation

Workflow (high-level)

Build a topological transmission network from substation locations (OpenStreetMap) and transmission-line geometries (California Energy Commission).

Use USGS ShakeMap PGA fields for historical earthquakes as test scenarios (e.g., 1933 Long Beach, 1971 San Fernando, 1994 Northridge).

Estimate probabilistic substation damage states and repair times using lognormal fragility functions with Monte Carlo sampling.

Propagate substation disruption to tract-level service using a tract–substation mapping matrix (W), and generate spatial impact maps.

Simulate restoration under:

an unconstrained baseline; and

a logistics-aware setting (e.g., 135 crews from 7 bases with travel-time constraints).

Compare prioritization strategies (e.g., Random, Centrality-first, Population-impact-first, Hospital-first) using service restoration metrics and connectivity-based robustness metrics (e.g., LCC).

Classify census tracts into resilience typologies using PCA + K-means with simulated recovery metrics and socioeconomic indicators.

Quickstart (suggested run order)

Topology + W-matrix

python Topology_and_Weight.py


Main pipeline (simulation + restoration + KPIs + typology)

python C257H_Project_Main.py


Visualization

python Project_Visualizer.py


If you want logistics-aware travel times from real road paths, run:

python build_travel_matrices_osm.py

Inputs

This repository expects external datasets (not included) from the sources below. You may need to standardize IDs/CRS and file naming to match the pipeline configuration.

OpenStreetMap contributors (Open Database License): https://www.openstreetmap.org/

California Energy Commission GIS Data (transmission lines): https://gis.data.ca.gov/

USGS ShakeMap (Peak Ground Acceleration): https://earthquake.usgs.gov/data/shakemap/

US Census TIGER/Line (tract boundaries): https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

US Census ACS 5-year estimates: https://api.census.gov/data.html

CDC/ATSDR Social Vulnerability Index (SVI): https://www.atsdr.cdc.gov/place-health/php/svi/index.html

Outputs (typical)

Depending on enabled stages, the pipeline can export:

Substation-level damage-state probabilities and recovery trajectories

Tract-level service/outage and recovery trajectories

Strategy-level KPI tables (system / population-weighted / equity-weighted)

Maps and summary plots

PCA loadings, clustering labels, and tract typology summaries

Limitations

Connectivity-based robustness metrics are graph proxies and do not guarantee operational feasibility (e.g., AC/DC power flow, voltage constraints) unless such modules are added.

Service estimation depends on the tract–substation mapping matrix (W); results are sensitive to how W is constructed (distance decay, assignment rules, normalization, etc.).

Repair/recovery parameterization should be interpreted as scenario-based and comparative (not a calibrated operational forecast) unless validated against empirical restoration data.

Citation

If you use this codebase, please cite the following references as appropriate:

Cheng, B., Nozick, L., Dobson, I., Davidson, R., Obiang, D., Dias, J., & Granados, M. (2024). Quantifying the earthquake risk to the electric power transmission system in Los Angeles at the census tract level. IEEE Access. https://doi.org/10.1109/ACCESS.2024.3408797

Çağnan, Z., Davidson, R. A., & Guikema, S. D. (2006). Post-earthquake restoration planning for Los Angeles electric power. Earthquake Spectra, 22(3), 589–608. https://doi.org/10.1193/1.2222400

Xu, N., Guikema, S. D., Davidson, R. A., Nozick, L. K., Çağnan, Z., & Vaziri, K. (2007). Optimizing scheduling of post-earthquake electric power restoration tasks. Earthquake Engineering & Structural Dynamics, 36(3), 265–284. https://doi.org/10.1002/eqe.623

Cavdaroglu, B., Hammel, E., Mitchell, J. E., Sharkey, T. C., & Wallace, W. A. (2013). Integrating restoration and scheduling decisions for disrupted interdependent infrastructure systems. Annals of Operations Research, 203(1), 279–294. https://doi.org/10.1007/s10479-011-0959-3
