# Optimization Project Roadmap

This document outlines the step-by-step plan for refactoring the existing `SCA_DARK_STORES` codebase into a pristine Optimization Project for the BM60132 Course (Optimization & Business Decision Analysis).

## 1. The Core Narrative
As requested, the project will tell a very compelling data-driven optimization story:
**"The Sub-optimality of Siloed Logistics: Evaluating Cost Savings via Simultaneous Delivery and Pickup VRP (SDVRP)."**

Instead of predicting returns or merging data, the report will specifically assume returns are known and focus purely on the objective of minimizing operational cost.
The workflow will demonstrate:
1. **Isolated Cost**: The total routing cost and fleet size when we optimize deliveries (CVRPTW) entirely separate from returns/pickups (CVRPTW).
2. **Integrated Cost**: The routing cost and fleet size when we implement SDVRP (Simultaneous Delivery and Pickup VRP) allowing trucks to do both tasks on the same route dynamically.
3. **The Result**: A direct comparative analysis demonstrating the savings in cost, emissions/distance, and vehicle capacity.

## 2. Refactoring the Codebase (Step-by-Step)

When you create your new directory for the Optimization project, do **not** copy the entire 12-stage pipeline. We will treat the ML and clustering outputs as "Secondary Pre-processed Data".

### Step 2.1: Data Migration
1. Copy `data/master_df_v3.parquet` (which already has locations and return probabilities).
2. Copy `data/dark_stores_final.csv` (which has the optimal store locations).
*By doing this, you instantly skip Stages 0 through 6 of the old pipeline.*

### Step 2.2: Isolate the Optimization Logic
Copy only the following necessary files into your new project's `src/` folder:
- `route_parser.py`: Contains the cost calculators and distance matrices.
- `forward_vrp.py`: The isolated delivery solver.
- `reverse_vrp.py`: The isolated pickup solver.
- `joint_optimizer.py` (specifically only the `solve_sdvrp_hybrid` and `run_all_zones_sdvrp` functions).

### Step 2.3: Create `main_optimization.py`
Write a new, highly condensed pipeline script that simply does the following:
```python
# 1. Load the pre-processed data
# 2. Run Forward VRP (Capture Cost 1)
# 3. Run Reverse VRP (Capture Cost 2)
# 4. Total Independent Cost = Cost 1 + Cost 2
# 5. Run SDVRP Hybrid Solver (Capture Integrated Cost)
# 6. Print Cost Savings (%) and Vehicle Fleet Savings (# trucks)
```

## 3. Writing the 5-Page Report

Follow this strict structural outline for your LaTeX/Word document to comply with the 5-page rule.

### I. Introduction (0.75 pages)
- Define the modern e-commerce problem: Returns and deliveries are handled by separate fleets, causing urban congestion and wasted capacity.
- **Objective**: Use operations research to compare the cost characteristics of Independent routing vs. Integrated SDVRP routing.

### II. Methodology & Mathematical Models (2 pages)
*This is the most critical section for an Optimization course! Give the math room to breathe.*
- **Data Assumptions**: State that you are utilizing a localized subset of the Olist E-Commerce dataset (approximate demand and geocoordinates are known).
- **Model 1: Independent CVRPTW**:
  - Show the objective function: Minimize $C = \sum (Distance \cdot VarCost) + \sum (Vehicles \cdot FixedCost)$
  - Explain briefly the time window and capacity constraints.
- **Model 2: SDVRP (Simultaneous Delivery and Pickup)**:
  - Detail the unified node formulation.
  - Explain the dynamic load tracking constraint: `Load(current) = Load(previous) - Delivery(current) + Pickup(current)`.
  - Explain that $0 \le Load \le MaxCapacity$ at all times.
- **Solver**: Mention the use of Google OR-Tools using Simulated Annealing.

### III. Results and Discussion (1.5 pages)
- **Table 1: Cost Comparison (Independent vs Integrated)**
  - Columns: Metric | Independent Ops | Integrated Ops | Savings (%)
  - Rows: Total Route Distance (km) | Vehicles Deployed | Total Routing Cost (R$)
- **Discussion**: Analyze the results. Why did SDVRP save money? (e.g., overlapping routes are eliminated, empty backhauls are utilized). Mention any trade-offs (e.g., solver computation time is higher for SDVRP).

### IV. Conclusion (0.5 pages)
- Summarize that integrating forward and reverse logistics via SDVRP significantly reduces transportation costs.
- Highlight the business value for standard retail fulfillment operations.

### V. Bibliography (0.25 pages)
- Cite OR-Tools.
- Cite the Olist E-commerce dataset.
- Cite 1 or 2 standard Vehicle Routing papers if possible.

---

*This blueprint will ensure your submission is deeply rooted in mathematical optimization, easily fits the page limits, directly attacks the prompt, and minimizes the coding overhead needed to finish the project.*
