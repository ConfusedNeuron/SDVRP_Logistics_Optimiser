# Project Context & AI IDE Setup (BM60132 OBDA Optimization Project)

## Welcome Back!
If you are an AI assistant who just got instantiated in this folder, here is the exact context of what we are doing:

### 1. Goal
We are building a 5-page Optimization project report for the course **Optimization & Business Decision Analysis (BM60132)**. The group has **4 members**.
We have repurposed an older, massively complex codebase (`SCA_DARK_STORES`) and trimmed it down explicitly to fit the optimization constraint of this assignment. 

### 2. The Narrative
Our optimization project tells this story: **"The Cost-Saving Mathematics of Integrating Forward and Reverse Logistics using SDVRP"**.
- We are showcasing the cost inefficiency of solving two Vehicle Routing Problems separately (Independent Deliveries via CVRPTW + Independent Returns via CVRPTW).
- We demonstrate how formulating an **SDVRP (Simultaneous Delivery and Pickup VRP)** allows trucks to dynamically pick up and drop off items on a single route, generating cost savings, fleet reductions, and route overlap elimination.

### 3. What Exists in this Folder Right Now
We have deliberately avoided dragging the entire ML/Data workflow. We have only copied the optimization engines and the static data pre-processed from the older project:
- **`data/master_df_v3.parquet`**: E-commerce data containing nodes/coords and `return_prob`.
- **`data/dark_stores_final.csv`**: Target facility depots.
- **`src/forward_vrp.py` & `src/reverse_vrp.py`**: The OR-Tools scripts to calculate independent CVRPTW costs.
- **`src/joint_optimizer.py`**: The magic script containing `solve_sdvrp_hybrid` to calculate the integrated SDVRP cost.
- **`optimization_roadmap.md`**: The exact step-by-step gameplan for the 5-page report layout.

### 4. Next Steps (Where to pick up from here):
1. **Set up the virtual environment**: Ensure `pandas`, `numpy`, `ortools`, and `pulp` are installed.
2. **Write `src/main_optimization.py`**: This script doesn't exist yet. It needs to be written to tie the data to the solvers, execute the Independent vs Hybrid runs, and print a consolidated output comparing the costs and vehicle metrics.
3. **Draft the Report**: Walk through the variables and equations. We need to clearly present the mathematical formulation of our SDVRP optimization model in the report.

### 5. AI Notes regarding Reproducibility and Naming
- **Proposed Project Names**: `SDVRP_Logistics_Optimizer` (Recommended), `OptiFlow_Logistics`, `Integrated_Routing_Engine`, `EcoRoute_SDVRP`.
- **Reproducibility Strategy**: The user wants this project to be perfectly reproducible on GitHub while satisfying an Optimization course rubric. Our strategy is:
  1. We assume `master_df_v3.parquet` and `dark_stores_final.csv` are the "raw starting datasets".
  2. We abstract away data-cleaning/ML layers so professors focus on VRP formulations.
  3. Real end-to-end extraction capability has been provided via a link to the original repo (`https://github.com/ConfusedNeuron/SCA_DARK_STORES`) in the `README.md`.

*(Proceed directly to creating `main_optimization.py` based on the blueprints provided in `optimization_roadmap.md`!)*
