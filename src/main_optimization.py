import pandas as pd
from pathlib import Path
import time
import sys

# Ensure root is in path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import forward_vrp
from src import reverse_vrp
from src import joint_optimizer

def main():
    print("===============================================================")
    print("      SDVRP LOGISTICS OPTIMIZER (Independent vs Integrated)    ")
    print("===============================================================\n")

    start_time = time.time()

    # Define paths
    parquet_path = "data/master_df_v3.parquet"
    stores_path = "data/dark_stores_final.csv"
    out_dir = Path("outputs")
    data_dir = Path("data")

    # Ensure output directories exist
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run Isolated Forward Deliveries
    print("\n--- PHASE 1: ISOLATED DELIVERY OPTIMIZATION (CVRPTW) ---")
    fwd_res = forward_vrp.run_full_pipeline(
        parquet_path=parquet_path,
        stores_path=stores_path,
        out_dir=out_dir,
        data_dir=data_dir
    )

    # 2. Run Isolated Pickups/Returns
    print("\n--- PHASE 2: ISOLATED REVERSE OPTIMIZATION (CVRPTW) ---")
    rev_res = reverse_vrp.run_full_pipeline(
        parquet_path=parquet_path,
        stores_path=stores_path,
        out_dir=out_dir,
        data_dir=data_dir
    )

    # 3. Run Integrated Operations (SDVRP Hybrid)
    print("\n--- PHASE 3: INTEGRATED LOGISTICS (SDVRP HYBRID) ---")
    hybrid_kpi_df = joint_optimizer.run_all_zones_sdvrp(
        fwd_zones=fwd_res["zones"],
        rev_zones=rev_res["reverse_zones"],
        fwd_kpi_df=fwd_res["kpi_df"],
        rev_kpi_df=rev_res["kpi_df"],
        num_vehicles=10, # Give it enough vehicles to maneuver
        output_dir=out_dir
    )

    # 4. Computation
    total_fwd_cost = fwd_res["kpi_df"]["routing_cost_R$"].sum()
    total_rev_cost = rev_res["kpi_df"]["routing_cost_R$"].sum()
    independent_cost = total_fwd_cost + total_rev_cost

    total_fwd_dist = fwd_res["kpi_df"]["total_dist_km"].sum()
    total_rev_dist = rev_res["kpi_df"]["total_dist_km"].sum()
    independent_dist = total_fwd_dist + total_rev_dist

    total_fwd_veh = fwd_res["kpi_df"]["n_vehicles_used"].sum()
    total_rev_veh = rev_res["kpi_df"]["n_vehicles_used"].sum()
    independent_veh = total_fwd_veh + total_rev_veh

    integrated_cost = hybrid_kpi_df["routing_cost_R$"].sum()
    integrated_dist = hybrid_kpi_df["total_dist_km"].sum()
    integrated_veh = hybrid_kpi_df["n_vehicles_used"].sum()

    savings_cost_pct = ((independent_cost - integrated_cost) / independent_cost) * 100
    savings_dist_pct = ((independent_dist - integrated_dist) / independent_dist) * 100

    print("\n===============================================================")
    print("                 COMPARATIVE RESULTS SUMMARY                   ")
    print("===============================================================")
    print(f"| Metric                   | Independent Ops | Integrated (SDVRP) | Savings   |")
    print(f"|--------------------------|-----------------|--------------------|-----------|")
    print(f"| Routing Cost (R$)        | R$ {independent_cost:,.2f}  | R$ {integrated_cost:,.2f}     | +{savings_cost_pct:.1f}%    |")
    print(f"| Total Distance (km)      | {independent_dist:,.1f}          | {integrated_dist:,.1f}             | +{savings_dist_pct:.1f}%    |")
    print(f"| Vehicles Deployed        | {independent_veh}              | {integrated_veh}                 | {independent_veh - integrated_veh} trucks |")
    print("===============================================================\n")

    print(f"Pipeline executed in {(time.time() - start_time) / 60:.2f} minutes.")
    print("All outputs saved locally in 'outputs/' and 'data/' directories.")

if __name__ == "__main__":
    main()
