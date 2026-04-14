# SDVRP Logistics Optimizer

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![OR-Tools](https://img.shields.io/badge/Google_OR--Tools-Optimization-red?style=for-the-badge)
![Academic](https://img.shields.io/badge/Course-BM60132_OBDA-gold?style=for-the-badge)

**OptiFlow** is the mathematics and optimization codebase supplementing a research initiative to radically improve the unit economics of modern e-commerce last-mile logistics.

## 🎯 Project Objective

The explosive growth of global e-commerce has fundamentally altered urban logistics, resulting in massive inefficiencies whenever deliveries ("forward logistics") and returns ("reverse logistics") are handled by separate fleets. 

The goal of this operations research project is to explicitly model and evaluate the operational advantages of integrating these flows. We model a network transitioning from isolated routing to an integrated **Simultaneous Delivery and Pickup Vehicle Routing Problem (SDVRP)**. Our optimization algorithms demonstrate how dynamic capacitated routing reduces asset deployment, mitigates overlapping routes, and heavily truncates overall logistics expenditure.

## 📈 Key Findings

By implementing the SDVRP model across 11 spatial micro-fulfillment zones (based on real-world Brazilian geocoordinates from the Olist Dataset), the solver proved that dynamic route integration yields undeniable operational improvements:

| Metric | Independent Operations | Integrated (SDVRP) | Net Savings |
| :--- | :--- | :--- | :--- |
| **Routing Cost (R$)*** | R$ 4,701.58 | R$ 3,295.92 | **+ 29.9%** |
| **Total Distance (km)** | 2,034.4 | 1,430.6 | **+ 29.7%** |
| **Vehicles Deployed** | 33 Trucks | 23 Trucks | **10 Trucks** |

*\*Note: Financial metrics are denominated in Brazilian Real (R$) owing to the Olist dataset's origin.*

## 🧠 Methodology & Mathematical Formulation

Instead of dispatching separate forward strings ($V_F$) and reverse strings ($V_R$), our solver dispatches a single unified hybrid fleet $V_{hybrid}$ using the **Google OR-Tools** constraint solver. 

The defining characteristic of SDVRP is that a vehicle drops off goods (opening physical space) and picks up goods (consuming space) continuously along its trip. Let $L_{ik}$ be the load of vehicle $k$ upon departing node $i$. The load state dynamically updates across the route:
$$ L_{jk} = L_{ik} - d_j + p_j \quad \forall j \text{ visited after } i $$

Because OR-Tools evaluates state transitions incrementally, we mapped the transit payload scalar simply as $(p_j - d_j)$. This ensures the underlying engine strictly enforces the absolute capacity constraint dynamically at all stages without failing:
$$ 0 \le L_{ik} \le Q_{max} \quad \forall i,k $$

## ⚖️ Design & Engineering Trade-offs

Building a production-grade routing engine required several rigorous OR decision gates:

- **Decisions Made (Why we succeeded)**: Rather than duplicating a customer node (one for forward, one for reverse) and forcing the solver to visit both, we unified the node vectors and pushed payload constraint validation into the transit callback. This drastically accelerated the `SIMULATED_ANNEALING` global search matrix. Furthermore, we utilized a **p-Median MILP** upstream to cluster data, preventing the routing engine from encountering exponential NP-hard time complexities.
- **Future Scope (Where we can optimize next)**: We forced the model to assume return requests were static and known at morning dispatch. A superior iteration would require a *Stochastic* VRP, actively utilizing Machine Learning to shift expected margin-of-errors inside the objective function dynamically.

## 🌍 Real-World Business Value: The Indian Market 

To contextualize the commercial value of SDVRP, we extrapolated our 30% savings blueprint against the Indian e-commerce landscape (e.g., Delhivery, Ecom Express, Flipkart):
- **Base Rate:** India ships $\approx$ 10 Million parcels/day inherently burdened by a $\sim$ 20% return/rejection rate.
- **Urban Overlap Scale:** Conservatively assuming only 10% of these paths are dense enough to overlap, syncing 200,000 forward and reverse flows saves 30% of their routing expenditure.
- **OPEX Savings:** This translates directly to **₹60 Lakhs saved per day** ($\sim$ ₹180 Crores purely in transportation fuel/var costs annually).
- **CapEx Elimination:** Eliminating $\sim$ 30% of the active fleet footprint (reducing 10 out of every 33 vehicles required) saves integrators massive decay in leased capital (e.g., electric 3-wheelers), amounting to an aggregate macroeconomic ceiling approaching **₹1,000 Crores** annually in capital retention.

---

## 🧬 Reproducibility and Data Origins

This repository isolates the **mathematical optimization formulations** and **OR-Tools network solvers**. For academic focus on operations algorithms, the upstream raw data engineering, geospatial clustering (p-Median), and Machine Learning components (XGBoost Return Probability modeling) have been pre-processed.

- The baseline datasets for the solvers are cached in the `data/` directory (`master_df_v3.parquet` and `dark_stores_final.csv`).

> **Note:** To inspect the full, end-to-end 12-stage machine learning pipeline spanning from raw Olist E-commerce data ingestion to the final structured VRP nodes, please refer to the parent project repository: [ConfusedNeuron/SCA_DARK_STORES](https://github.com/ConfusedNeuron/SCA_DARK_STORES).

## 🚀 Installation & Quickstart

The project relies on a modern `pyproject.toml` packaging ecosystem.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ConfusedNeuron/SDVRP_Logistics_Optimiser.git
   cd SDVRP_Logistics_Optimiser
   ```

2. **Create a virtual environment and install dependencies using `uv`:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   uv pip install -e .
   ```

3. **Execute the Optimization Pipeline:**
   This command sequentially runs the Forward Solvers, Reverse Solvers, and finally the Integrated SDVRP Model, printing the comparative results table directly to your terminal.
   ```bash
   python src/main_optimization.py
   ```

## 🏗️ Core Architecture
- `src/main_optimization.py`: Primary orchestration pipeline.
- `src/forward_vrp.py`: Baseline Isolated Delivery Solver (CVRPTW).
- `src/reverse_vrp.py`: Baseline Isolated Pickup Solver (CVRPTW).
- `src/joint_optimizer.py`: Advanced Integrated SDVRP Hybrid Solver utilizing `PATH_CHEAPEST_ARC` and `SIMULATED_ANNEALING` strategies.
- `src/route_parser.py`: Cost matrices and constraint definitions.
