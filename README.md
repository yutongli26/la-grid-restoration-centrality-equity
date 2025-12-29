# Robustness and Post-Earthquake Recovery of the Los Angeles Power Grid at Census-Tract Scale

This repository presents a tract-resolved modeling pipeline to evaluate earthquake-driven disruption of the City of Los Angeles transmission grid and compare substation repair prioritization strategies under three objectives: restoring network structure, restoring electricity service to the population, and advancing equitable restoration across census tracts. Network-structure priorities are derived from substation centrality metrics, including degree, betweenness, closeness, and population-weighted impact centrality. K-means clustering is then used to group census tracts into resilience typologies based on simulated outage and recovery metrics and socioeconomic indicators.

## Workflow
1. Build a topological transmission network from substation locations (OpenStreetMap) and transmission-line geometries (California Energy Commission).
2. Use U.S. Geological Survey ShakeMap Peak Ground Acceleration (PGA) fields for three historical earthquakes (1933 Long Beach, 1971 San Fernando, 1994 Northridge) as test scenarios.
3. Estimate probabilistic substation damage states and repair times using lognormal fragility functions with Monte Carlo sampling.
4. Translate substation disruption into tract-level outage probabilities using a tract–substation weighting (mapping) matrix and generate spatial impact maps.
5. Simulate restoration under an unconstrained baseline and a logistics-aware setting (28 crews dispatched from 7 bases with travel-time constraints).
6. Compare prioritization strategies (Random, Centrality-First, Population-Weighted Impact-First, Hospital-First) using service restoration and robustness metrics based on the largest connected component.
7. Classify census tracts into resilience typologies using principal component analysis and k-means clustering with simulated recovery metrics and socioeconomic indicators.

## Data inputs
- OpenStreetMap contributors (Open Database License). https://www.openstreetmap.org/
- California Energy Commission GIS Data (transmission lines). https://gis.data.ca.gov/
- U.S. Geological Survey ShakeMap (Peak Ground Acceleration). https://earthquake.usgs.gov/data/shakemap/
- U.S. Census Bureau TIGER/Line (census tract boundaries). https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
- U.S. Census Bureau American Community Survey (ACS) 5-year estimates. https://api.census.gov/data.html
- CDC/ATSDR Social Vulnerability Index (SVI). https://www.atsdr.cdc.gov/placeandhealth/svi/

## Notes
The pipeline is intended for scenario-based stress testing and comparative evaluation of restoration strategies. Connectivity-based robustness metrics quantify fragmentation but do not enforce power-flow feasibility unless operational constraints are added.

Suggested run order: `Topology and Weight` → `C257H_Project_Main` → `Project Visualizer`.

The function `run_replica_od_analysis(...)` in the main pipeline is not used in the current experiments. Stage 5 is reserved for a future optimization module.

## Citation

Key references
- Cheng, B., Nozick, L., Dobson, I., Davidson, R., Obiang, D., Dias, J., & Granados, M. (2024). Quantifying the earthquake risk to the electric power transmission system in Los Angeles at the census tract level. *IEEE Access*. https://doi.org/10.1109/ACCESS.2024.3408797.
- Çağnan, Z., Davidson, R. A., & Guikema, S. D. (2006). Post-earthquake restoration planning for Los Angeles electric power. *Earthquake Spectra*, 22(3), 589–608. https://doi.org/10.1193/1.2222400.
- Xu, N., Guikema, S. D., Davidson, R. A., Nozick, L. K., Çağnan, Z., & Vaziri, K. (2007). Optimizing scheduling of post-earthquake electric power restoration tasks. *Earthquake Engineering & Structural Dynamics*, 36(3), 265–284. https://doi.org/10.1002/eqe.623.
- Cavdaroglu, B., Hammel, E., Mitchell, J. E., Sharkey, T. C., & Wallace, W. A. (2013). Integrating restoration and scheduling decisions for disrupted interdependent infrastructure systems. *Annals of Operations Research*, 203(1), 279–294. https://doi.org/10.1007/s10479-011-0959-3.
