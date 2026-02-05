
# Robustness and Post-Earthquake Recovery of the Los Angeles Power Grid at Census-Tract Scale

**Team:** Yinchen Yi, Yutong Li

A census-tract–scale modeling pipeline to (i) simulate earthquake-induced substation damage, (ii) propagate disruption to tract-level service via a tract–substation mapping matrix, and (iii) compare post-earthquake repair prioritization strategies under three objectives:

- **Network robustness** (connectivity / fragmentation proxies)
- **Population service restoration**
- **Equity-aware restoration** across census tracts (e.g., SVI-weighted objectives)

The pipeline supports both an **unconstrained baseline** and a **logistics-aware restoration** setting (multi-crew, multi-depot, travel-time constraints). We further perform **PCA + K-means** to group census tracts into **resilience typologies** using simulated outage/recovery metrics and socioeconomic indicators.

> **Scope note:** robustness metrics here are connectivity-based (e.g., largest connected component) and do **not** enforce power-flow feasibility unless operational constraints are added.

---

## Workflow
1. Topological network construction from transmission line and substation data using endpoint snapping, line splitting at substations, and graph extraction of substation-to-substation direct links
2. Tract-substation weighting matrix (**W**) construction based on Inverse Distance Weighting
3. Centrality and percolation analysis of topological network
4. Scenario-based stress testing using ShakeMap PGA fields (1933 Long Beach, 1971 San Fernando, 1994 Northridge, 2%-in-50yr)
5. Probabilistic substation damage-state modeling using lognormal fragility with Monte Carlo sampling
6. Tract-level service estimation through tract–substation weighting matrix (**W**)
7. Restoration simulation under:
  - Baseline (no constraints)
  - Logistics-aware rule-based and GA optimisation-based scheduling (multi-crew, multi-depot, travel-time constraints)
8. Strategy comparison (Theoretical Limit / Random Baseline / Betweenness (Bridges) First / Impact (Population) First / Degree (Hubs) First / Hospital First / Impact λ2 (Grid) First / Closeness First / Balanced GA / HospFirst GA / Efficiency GA) using service restoration metrics and connectivity-based robustness metrics
9. Tract resilience typology via PCA + K-means using simulated + socioeconomic features
---

## Repository structure

- `Topology_and_Weight.py` — Construct substation-level transmission topology (snapping/splitting + direct links) and export tract–substation influence weights (W / mapping CSVs)
- `IDW.py` — Interpolate scenario PGA grids to substations (IDW/KDTree), producing per-scenario PGA_* columns for downstream simulations
- `build_travel_matrices_osm.py` — Precompute base→substation and substation→substation travel-time matrices from OSM road network for crew scheduling
- `C257H_Project_Main.py` — End-to-end pipeline (hazard→damage MC→network impact→restoration scheduling/GA→KPIs→typology clustering), outputs organized by Stage folders
- `Project_Visualizer.py` — Post-processing visualizer: maps and comparative figures from Stage outputs (e.g., supply maps + histograms, logistics heatmaps, KPI bars, Gantt charts, cluster profile plots)

## Quickstart (suggested run order)

1. **Topology + W-matrix**

    ```bash
    python Topology_and_Weight.py
    ```
2. **Intensity measure interpolation**

    ```bash
    python IDW.py
    ```
3. **Travel matrix**
 
    ```bash
    python build_travel_matrices_osm.py
    ```
4. **Main pipeline (simulation + restoration + KPIs + typology)**

    ```bash
    python C257H_Project_Main.py
    ```

5. **Visualization**

    ```bash
    python Project_Visualizer.py
    ```

---

## Inputs

This repository expects external datasets, all of which included in the "data" folder.

- OpenStreetMap contributors (Open Database License): <https://www.openstreetmap.org/>
- California Energy Commission GIS Data (transmission lines): <https://gis.data.ca.gov/>
- USGS ShakeMap (Peak Ground Acceleration): <https://earthquake.usgs.gov/data/shakemap/>
- US Census TIGER/Line (tract boundaries): <https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html>
- US Census ACS 5-year estimates: <https://api.census.gov/data.html>
- CDC/ATSDR Social Vulnerability Index (SVI): <https://www.atsdr.cdc.gov/place-health/php/svi/index.html>

---

## Outputs (typical)

Depending on enabled stages, the pipeline can export:

- Substation-level damage-state probabilities and recovery trajectories
- Tract-level service/outage and recovery trajectories
- Strategy-level KPI tables (system / population-weighted / equity-weighted)
- Maps and summary plots
- PCA loadings, clustering labels, and tract typology summaries

---

## Limitations

- Connectivity-based robustness metrics are graph proxies and do not guarantee operational feasibility (e.g., AC/DC power flow, voltage constraints) unless such modules are added.
- Service estimation depends on the tract–substation mapping matrix (W); results are sensitive to how W is constructed (distance decay, assignment rules, normalization, etc.).

---

## Citation

- Cheng, B., Nozick, L., Dobson, I., Davidson, R., Obiang, D., Dias, J., & Granados, M. (2024). Quantifying the earthquake risk to the electric power transmission system in Los Angeles at the census tract level. *IEEE Access*. <https://doi.org/10.1109/ACCESS.2024.3408797>
- Çağnan, Z., Davidson, R. A., & Guikema, S. D. (2006). Post-earthquake restoration planning for Los Angeles electric power. *Earthquake Spectra*, 22(3), 589–608. <https://doi.org/10.1193/1.2222400>
- Xu, N., Guikema, S. D., Davidson, R. A., Nozick, L. K., Çağnan, Z., & Vaziri, K. (2007). Optimizing scheduling of post-earthquake electric power restoration tasks. *Earthquake Engineering & Structural Dynamics*, 36(3), 265–284. <https://doi.org/10.1002/eqe.623>
- Cavdaroglu, B., Hammel, E., Mitchell, J. E., Sharkey, T. C., & Wallace, W. A. (2013). Integrating restoration and scheduling decisions for disrupted interdependent infrastructure systems. *Annals of Operations Research*, 203(1), 279–294. <https://doi.org/10.1007/s10479-011-0959-3>
