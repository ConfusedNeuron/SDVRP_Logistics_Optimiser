# SDVRP Logistics Optimizer (OptiFlow)

This project contains the codebase for a 5-page Optimization report submitted for the **Optimization & Business Decision Analysis (BM60132)** course. 

## Project Objective
The goal is to mathematically evaluate the cost savings and fleet reduction achieved by transitioning from isolated delivery/pickup routes to an integrated **Simultaneous Delivery and Pickup Vehicle Routing Problem (SDVRP)** structure.

## Reproducibility and Data Origins
This repository focuses strictly on mathematical optimization formulations and OR-Tools network solvers. For clarity and focus, the raw data engineering, geospatial clustering (p-Median), and Machine Learning (Return Probability modeling) have been completely pre-processed and abstracted to provide a clean start for the VRP models.
- The starting datasets for the solvers are located in the `data/` directory (`master_df_v3.parquet` and `dark_stores_final.csv`).

If you wish to fully replicate the data ingestion pipeline from the raw Olist E-commerce dataset up to the structured VRP nodes (including the ML models), please refer to the parent repository which houses the complete 12-stage pipeline:
**[https://github.com/ConfusedNeuron/SCA_DARK_STORES](https://github.com/ConfusedNeuron/SCA_DARK_STORES)**

## Running the Optimizers
1. Install dependencies: `pip install -r requirements.txt`
2. Execute the optimization pipeline: `python src/main_optimization.py`
3. View the generated comparative cost tables and Pareto tradeoffs.
