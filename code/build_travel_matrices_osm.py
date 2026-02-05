"""
build_travel_matrices_osm.py

Description:
    This script precomputes travel time matrices using an OSMnx road network graph.
    It replaces Euclidean/Geodesic distance estimates with real-world road network travel times.

Outputs (saved to 'Stage 4 Output'):
    1. travel_base_to_task.csv: Travel times from Crew Bases to Substations.
    2. travel_task_to_task.csv: Travel times between all pairs of Substations.

Compatibility:
    - Designed to match the 'load_travel_matrices()' function in 'C257H_Project_Main.py'.
    - Handles OSMnx version differences (nearest_nodes API).
    - Optimizes Dijkstra calculations by grouping substations at the same graph node.
"""

import os
import logging
from typing import List, Tuple, Dict, Set, Optional

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from pathlib import Path

# ======================= USER CONFIGURATION =======================

# --- File Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
GRAPHML_PATH = str(DATA_DIR / "la_drive.graphml")
SUBS_CSV = str(DATA_DIR / "Los_Angeles_City_SUBSTATION_with_fragility.csv")

# --- Output Settings ---
OUTPUT_ROOT = str(BASE_DIR)
STAGE4_DIR = "Stage 4 Output"
TRAVEL_BASE_TO_TASK_CSV = "travel_base_to_task.csv"
TRAVEL_TASK_TO_TASK_CSV = "travel_task_to_task.csv"

# --- Column Mappings (Input CSV) ---
# The script will prioritize these columns but falls back to defaults if not found.
ID_COL = "HIFLD_ID"
LON_COL = "LONGITUDE"
LAT_COL = "LATITUDE"

# --- Crew Bases ---
# CRITICAL: The IDs (base_0, base_1) MUST match the generation logic in the main pipeline.
# Coordinates must match 'C257H_Project_Main.py' exactly to prevent data drift.
CREW_BASES: List[Tuple[str, float, float]] = [
    ("base_0", 34.2318, -118.3817),  # Valley Yard (Sun Valley)
    ("base_1", 34.0375, -118.2555),  # Central Yard
]

# --- Calculation Parameters ---
# Substation IDs to include (None = Process all substations in CSV)
LIMIT_TO_SUB_IDS: Optional[List[str]] = None

# Max travel time in seconds (e.g., 6 hours).
# Dijkstra will stop searching beyond this limit to save time.
MAX_TRAVEL_TIME_SEC = 6 * 3600.0


# ======================= HELPER FUNCTIONS =======================

def setup_logging():
    """Initialize logging format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def ensure_dir(path: str):
    """Ensure the output directory exists."""
    os.makedirs(path, exist_ok=True)


def nearest_nodes_compat(G, xs, ys):
    """
    Compatibility wrapper for OSMnx's nearest_nodes function.
    Handles API changes between OSMnx < 1.0, 1.x, and 2.x.
    
    Args:
        G: The graph object.
        xs: List/Array of X coordinates (Longitude).
        ys: List/Array of Y coordinates (Latitude).
    """
    try:
        # Modern OSMnx (v1.3.0+)
        from osmnx.distance import nearest_nodes
        return nearest_nodes(G, xs, ys)
    except (ImportError, AttributeError):
        # Older OSMnx versions
        if hasattr(ox, "distance") and hasattr(ox.distance, "nearest_nodes"):
            return ox.distance.nearest_nodes(G, xs, ys)
        elif hasattr(ox, "get_nearest_nodes"):
            # Deprecated function (very old versions), works for single points
            return ox.get_nearest_nodes(G, xs, ys)
        else:
            raise RuntimeError("Could not find a compatible 'nearest_nodes' function in OSMnx.")


def load_graph_with_travel_time(graphml_path: str) -> nx.MultiDiGraph:
    """
    Load the GraphML file and ensure 'travel_time' attributes exist on edges.
    If 'travel_time' is missing or corrupted, it recalculates speeds and times.
    """
    logger = logging.getLogger()
    logger.info(f"Loading road network graph from {graphml_path} ...")
    
    # Load graph
    G = ox.load_graphml(graphml_path)

    # Validation: Check if 'travel_time' exists and is valid on edges
    logger.info("Validating edge 'travel_time' attributes...")
    missing_tt = False
    
    # Check a sample or all edges to ensure attribute existence
    for u, v, k, data in G.edges(keys=True, data=True):
        val = data.get("travel_time")
        if val is None or not isinstance(val, (int, float)) or not np.isfinite(val):
            missing_tt = True
            break
    
    if missing_tt:
        logger.warning("Missing or invalid 'travel_time' detected. Re-calculating edge speeds and times...")
        # Add free-flow speeds based on highway type
        G = ox.add_edge_speeds(G)
        # Calculate travel time (length / speed)
        G = ox.add_edge_travel_times(G)
    else:
        logger.info("Graph validation passed: 'travel_time' attributes are present.")

    return G


def load_substations() -> pd.DataFrame:
    """
    Load substation data from CSV and standardize columns to ['id', 'lon', 'lat'].
    """
    logger = logging.getLogger()
    logger.info(f"Loading substations from {SUBS_CSV} ...")

    df = pd.read_csv(SUBS_CSV)
    cols = list(df.columns)

    # 1. Identify ID Column
    if ID_COL in cols:
        curr_id = ID_COL
    elif "id" in cols:
        curr_id = "id"
    else:
        raise ValueError(f"ID column not found. Expected '{ID_COL}' or 'id'. Available: {cols}")

    # 2. Identify Coordinate Columns
    if LON_COL in cols and LAT_COL in cols:
        curr_lon, curr_lat = LON_COL, LAT_COL
    elif "lon.1" in cols and "lat.1" in cols:
        curr_lon, curr_lat = "lon.1", "lat.1"
    else:
        raise ValueError(f"Coordinates not found. Expected {LON_COL}/{LAT_COL}. Available: {cols}")

    # 3. Standardize
    df[curr_id] = df[curr_id].astype(str).str.strip()
    out = df[[curr_id, curr_lon, curr_lat]].copy()
    out = out.rename(columns={curr_id: "id", curr_lon: "lon", curr_lat: "lat"})

    # 4. Optional Filtering
    if LIMIT_TO_SUB_IDS is not None:
        limit_ids = set(str(s) for s in LIMIT_TO_SUB_IDS)
        out = out[out["id"].isin(limit_ids)].copy()
        logger.info(f"Filtered to {len(out)} substations based on user settings.")

    return out


def map_points_to_nodes(G, df_subs: pd.DataFrame) -> Dict[str, int]:
    """
    Map each substation (lat/lon) to the nearest node ID in the road graph.
    Returns: Dictionary {substation_id: graph_node_id}
    """
    logger = logging.getLogger()
    xs = df_subs["lon"].values
    ys = df_subs["lat"].values

    logger.info("Snapping substations to nearest graph nodes...")
    nodes = nearest_nodes_compat(G, xs, ys)

    df_subs = df_subs.copy()
    df_subs["node"] = nodes

    mapping = dict(zip(df_subs["id"].astype(str), df_subs["node"]))
    
    n_unique = df_subs["node"].nunique()
    logger.info(f"Mapped {len(mapping)} substations to {n_unique} unique road network nodes.")
    
    return mapping


def get_base_nodes(G) -> List[Tuple[str, int]]:
    """
    Map Crew Base coordinates to nearest graph nodes.
    Returns: List of tuples [(base_id, graph_node_id)]
    """
    logger = logging.getLogger()
    base_nodes: List[Tuple[str, int]] = []
    
    for base_id, lat, lon in CREW_BASES:
        node_id = nearest_nodes_compat(G, [lon], [lat])[0]
        base_nodes.append((base_id, node_id))
        
    logger.info(f"Mapped {len(base_nodes)} crew bases to graph nodes.")
    return base_nodes


def compute_travel_times(
    G: nx.MultiDiGraph,
    base_nodes: List[Tuple[str, int]],
    sub_to_node: Dict[str, int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute shortest path travel times using Dijkstra's algorithm.
    
    Returns:
        1. Base->Task DataFrame (Rows: Bases, Cols: Substations)
        2. Task->Task DataFrame (Rows: Substations, Cols: Substations)
    """
    logger = logging.getLogger()
    sub_ids = sorted(sub_to_node.keys())
    n_sub = len(sub_ids)
    n_base = len(base_nodes)

    # ---------------------------------------------------------
    # PART 1: Base -> Task Matrix
    # ---------------------------------------------------------
    logger.info(f"Computing Base->Task Matrix ({n_base} bases)...")
    base_mat_sec = np.full((n_base, n_sub), np.nan, dtype=float)

    for i, (base_id, base_node) in enumerate(base_nodes):
        # Use 'cutoff' to stop searching nodes further than MAX_TRAVEL_TIME
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, base_node, weight="travel_time", cutoff=MAX_TRAVEL_TIME_SEC
            )
        except TypeError:
            # Fallback for older NetworkX versions that don't support 'cutoff'
            lengths = nx.single_source_dijkstra_path_length(
                G, base_node, weight="travel_time"
            )
            
        for j, sid in enumerate(sub_ids):
            node = sub_to_node[sid]
            t_sec = lengths.get(node, np.nan)
            
            # Enforce cutoff manually if fallback was used
            if t_sec is not None and t_sec > MAX_TRAVEL_TIME_SEC:
                t_sec = np.nan
                
            base_mat_sec[i, j] = t_sec

    # Convert to hours and create DataFrame
    base_to_task_hr = base_mat_sec / 3600.0
    base_to_task_df = pd.DataFrame(
        base_to_task_hr,
        index=[b[0] for b in base_nodes], # Index = base_0, base_1...
        columns=sub_ids,
    )
    base_to_task_df.index.name = "base_id"

    # ---------------------------------------------------------
    # PART 2: Task -> Task Matrix (Optimized)
    # ---------------------------------------------------------
    logger.info("Computing Task->Task Matrix...")
    
    # Optimization: Substations often snap to the same road node.
    # We only run Dijkstra for *unique* graph nodes to save computation time.
    unique_nodes = sorted(list(set(sub_to_node.values())))
    n_unique = len(unique_nodes)
    logger.info(f"Optimization: Calculating paths for {n_unique} unique nodes (instead of {n_sub} substations).")
    
    # Dictionary to store pre-computed distances: {source_node: {target_node: time}}
    node_dists = {} 
    
    for i, u_node in enumerate(unique_nodes):
        if i % 50 == 0:
            logger.info(f"Progress: Processed {i}/{n_unique} unique source nodes...")
        
        try:
            dists = nx.single_source_dijkstra_path_length(
                G, u_node, weight="travel_time", cutoff=MAX_TRAVEL_TIME_SEC
            )
        except TypeError:
            dists = nx.single_source_dijkstra_path_length(
                G, u_node, weight="travel_time"
            )
        node_dists[u_node] = dists

    # Fill the full (Sub x Sub) matrix using the pre-computed node distances
    task_mat_sec = np.full((n_sub, n_sub), np.nan, dtype=float)
    
    for i, sid_o in enumerate(sub_ids):
        node_o = sub_to_node[sid_o]
        dists_from_o = node_dists.get(node_o, {})
        
        for j, sid_d in enumerate(sub_ids):
            if i == j:
                task_mat_sec[i, j] = 0.0
                continue
                
            node_d = sub_to_node[sid_d]
            t_sec = dists_from_o.get(node_d, np.nan)
            
            if t_sec is not None and t_sec > MAX_TRAVEL_TIME_SEC:
                t_sec = np.nan
            task_mat_sec[i, j] = t_sec

    # Convert to hours and create DataFrame
    task_mat_hr = task_mat_sec / 3600.0
    task_df = pd.DataFrame(task_mat_hr, index=sub_ids, columns=sub_ids)
    task_df.index.name = "id"

    # ---------------------------------------------------------
    # Quality Assurance Check
    # ---------------------------------------------------------
    nan_ratio = np.isnan(task_mat_hr).mean()
    if nan_ratio > 0.05: # Warn if > 5% of pairs are unreachable
        logger.error(f"CRITICAL WARNING: {nan_ratio:.2%} of Task-to-Task pairs are unreachable (NaN).")
        logger.error("Check if the graph is connected or if MAX_TRAVEL_TIME_SEC is too low.")
        # We raise an error to prevent bad data from entering the simulation
        raise ValueError("Too many unreachable substations in travel matrix.")
    else:
        logger.info(f"Travel matrix computed successfully. Unreachable pair ratio: {nan_ratio:.2%}")

    return base_to_task_df, task_df


def save_outputs(base_to_task: pd.DataFrame, task_to_task: pd.DataFrame):
    """
    Save matrices to CSV in the format expected by 'C257H_Project_Main.py'.
    """
    logger = logging.getLogger()
    stage4_path = os.path.join(OUTPUT_ROOT, STAGE4_DIR)
    ensure_dir(stage4_path)

    # 1. Save Base -> Task
    # Format: Index matches 'base_id' (base_0, base_1...), Columns match 'sub_id'
    out_base = os.path.join(stage4_path, TRAVEL_BASE_TO_TASK_CSV)
    base_to_task.to_csv(out_base, index=True)
    logger.info(f"Saved Base->Task matrix to: {out_base}")

    # 2. Save Task -> Task
    # Format: First column 'id', subsequent columns are sub_ids.
    # Note: We reset index to make 'id' a standard column, matching standard pandas read_csv behavior.
    df_task = task_to_task.copy()
    df_task.reset_index(inplace=True) 
    
    out_task = os.path.join(stage4_path, TRAVEL_TASK_TO_TASK_CSV)
    df_task.to_csv(out_task, index=False)
    logger.info(f"Saved Task->Task matrix to: {out_task}")


# ======================= MAIN EXECUTION =======================

def main():
    setup_logging()
    logger = logging.getLogger()
    logger.info("=== build_travel_matrices_osm: START ===")

    try:
        # 1. Load Data
        G = load_graph_with_travel_time(GRAPHML_PATH)
        subs_df = load_substations()
        
        # 2. Map Coordinates to Graph
        sub_to_node = map_points_to_nodes(G, subs_df)
        base_nodes = get_base_nodes(G)
        
        # 3. Compute Matrices
        base_to_task, task_to_task = compute_travel_times(G, base_nodes, sub_to_node)
        
        # 4. Save
        save_outputs(base_to_task, task_to_task)
        
        logger.info("=== build_travel_matrices_osm: DONE ===")
        
    except Exception as e:
        logger.exception("Fatal error during execution:")
        raise e

if __name__ == "__main__":
    main()