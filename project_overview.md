# Project Overview: SCA_DARK_STORES (Dark Store Placement + Integrated Logistics)

This document provides a complete summary of the existing `SCA_DARK_STORES` codebase. The project builds a comprehensive, end-to-end mathematical and machine learning pipeline using the Olist Brazilian E-Commerce dataset. 

The core objective of the project is to simultaneously place micro-fulfilment centers (dark stores), forecast returns, route deliveries, and orchestrate hybrid routes to minimize total logistics cost.

## 1. Project Architecture and Data Flow

The project is structured as a 12-stage dependency-driven pipeline (orchestrated via `main.py`). The pipeline operates as follows:

1. **Data Pipeline (`data_pipeline.py`)**: Ingests 9 raw Olist CSVs and merges them into a single `master_df.parquet`.
2. **Demand Baseline (`demand_baseline.py`)**: Profiles historical demand and creates initial geographical maps.
3. **Haversine Matrix (`haversine_matrix.py`)**: Computes a 500x500 dense distance matrix for the greater São Paulo area using Haversine formulations.
4. **Clustering & p-Median (`clustering.py`)**: Employs K-Means to identify candidate locations followed by a **p-Median MILP** (Mixed Integer Linear Programming) model to optimally place $K=11$ dark stores, minimizing the distance between customers and the nearest store.
5. **Return Classifier (`return_classifier.py`)**: Uses an XGBoost model on historical data to predict the probability (`return_prob`) that a specific future order will result in a return/pickup request.
6. **Demand Forecasting (`demand_forecasting.py`)**: Uses Facebook Prophet to forecast expected delivery volume by region.
7. **Scenario Builder (`scenario_builder.py`)**: Generates base (A), high demand (B), and high return (C) scenarios, converting DataFrame rows into structured nodes for the Routing Solvers.

## 2. Optimization Engines

Once the data is prepped and clustered, the project utilizes heavy optimization techniques (via Google OR-Tools and PuLP) to solve the logistics problems.

### A. Independent VRP Solvers (Forward & Reverse)
- **Forward VRP (`forward_vrp.py`)**: Formulates a Capacitated Vehicle Routing Problem with Time Windows (CVRPTW) for standard deliveries.
- **Reverse VRP (`reverse_vrp.py`)**: Formulates a separate CVRPTW to exclusively handle pickups. 
- *Solver details*: Simulated Annealing metaheuristic combined with Path Cheapest Arc routing.

### B. Integrated Logistics (SDVRP)
- **SDVRP Hybrid Solver (`joint_optimizer.py / solve_sdvrp_hybrid`)**: Solves a Simultaneous Delivery and Pickup Vehicle Routing Problem (SDVRP). 
- *Mechanism*: Rather than separating pickups and deliveries, this model dynamically merges delivery and pickup nodes into single routes for each vehicle. It tracks instantaneous vehicle load dynamically (Load = Initial + Pickups - Deliveries) ensuring capacity isn't broken at any point mid-route.

### C. Joint MILP Optimiser & Pareto Sweep
- **Joint MILP (`joint_optimizer.py`)**: Solves a meta-level optimization problem balancing variable cost constraints vs penalty metrics.
- *Objective*: $Z = \alpha \cdot C_{fwd} + \beta \cdot C_{rev} + \gamma \cdot T_{pen} + \delta \cdot N_{veh}$
- *Trade-off*: Analyzes whether it's cheaper to deploy an extra truck or to incur a "return penalty" (unserved returns) using an $\epsilon$-constraint Pareto sweep.

## 3. Results and Key Metrics

- **Dark Stores Placed**: 11 (Providing 73.7% customer coverage within 5 km).
- **Forward Delivery Operations**: Deployed 22 vehicles traversing 1,070 km for a cost of R$ 2,704.
- **Reverse Return Operations**: Deployed 15 vehicles traversing 946 km for a cost of R$ 2,170.
- **Value of Optimization**: The routing engine provided a 98.6% distance reduction compared to naive (direct point-to-point) routing baselines.

## 4. Technology Stack
- **Data & ML**: `pandas`, `numpy`, `xgboost`, `prophet`, `scikit-learn`
- **Optimization Algorithms**: `or-tools` (Routing/CVRPTW/SDVRP), `pulp` (MILP)
- **Environment**: Python 3.10+, managed using `uv` (or `pip`)

---

*This document serves as the baseline blueprint for all code logic currently residing in `SCA_DARK_STORES/src`.*
