import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


@dataclass
class Node:
    name: str
    node_type: str
    on_hand: float
    holding_cost: float
    lead_time: int
    order_cost: float
    emission_rate: float
    min_order: float
    capacity: float
    safety_stock: float = 0.0
    reorder_point: float = 0.0
    order_quantity: float = 0.0
    inbound: List[Dict] = field(default_factory=list)
    order_backlog: float = 0.0
    delivered: float = 0.0
    lost_sales: float = 0.0


def build_supply_chain() -> Tuple[Dict[str, Node], Dict[str, Node], Dict[str, Node], List[Tuple[str, str]], List[Tuple[str, str]]]:
    suppliers = {
        "Supplier A": Node(
            name="Supplier A",
            node_type="supplier",
            on_hand=3000,
            holding_cost=0.20,
            lead_time=3,
            order_cost=100,
            emission_rate=0.8,
            min_order=200,
            capacity=5000,
        ),
        "Supplier B": Node(
            name="Supplier B",
            node_type="supplier",
            on_hand=2800,
            holding_cost=0.22,
            lead_time=4,
            order_cost=110,
            emission_rate=0.9,
            min_order=200,
            capacity=5000,
        ),
        "Supplier C": Node(
            name="Supplier C",
            node_type="supplier",
            on_hand=2600,
            holding_cost=0.18,
            lead_time=2,
            order_cost=95,
            emission_rate=0.7,
            min_order=200,
            capacity=5000,
        ),
    }

    dcs = {
        "DC North": Node(
            name="DC North",
            node_type="dc",
            on_hand=1200,
            holding_cost=0.35,
            lead_time=2,
            order_cost=80,
            emission_rate=0.5,
            min_order=150,
            capacity=2500,
        ),
        "DC Central": Node(
            name="DC Central",
            node_type="dc",
            on_hand=1400,
            holding_cost=0.33,
            lead_time=2,
            order_cost=80,
            emission_rate=0.5,
            min_order=150,
            capacity=2500,
        ),
        "DC South": Node(
            name="DC South",
            node_type="dc",
            on_hand=1000,
            holding_cost=0.38,
            lead_time=3,
            order_cost=80,
            emission_rate=0.5,
            min_order=150,
            capacity=2500,
        ),
    }

    stores = {
        "Store 1": Node(
            name="Store 1",
            node_type="store",
            on_hand=240,
            holding_cost=0.60,
            lead_time=1,
            order_cost=40,
            emission_rate=0.25,
            min_order=40,
            capacity=500,
        ),
        "Store 2": Node(
            name="Store 2",
            node_type="store",
            on_hand=220,
            holding_cost=0.58,
            lead_time=1,
            order_cost=40,
            emission_rate=0.25,
            min_order=40,
            capacity=500,
        ),
        "Store 3": Node(
            name="Store 3",
            node_type="store",
            on_hand=260,
            holding_cost=0.62,
            lead_time=1,
            order_cost=40,
            emission_rate=0.25,
            min_order=40,
            capacity=500,
        ),
        "Store 4": Node(
            name="Store 4",
            node_type="store",
            on_hand=200,
            holding_cost=0.55,
            lead_time=1,
            order_cost=40,
            emission_rate=0.25,
            min_order=40,
            capacity=500,
        ),
        "Store 5": Node(
            name="Store 5",
            node_type="store",
            on_hand=230,
            holding_cost=0.57,
            lead_time=1,
            order_cost=40,
            emission_rate=0.25,
            min_order=40,
            capacity=500,
        ),
    }

    supplier_to_dc = [
        ("Supplier A", "DC North"),
        ("Supplier B", "DC Central"),
        ("Supplier C", "DC South"),
        ("Supplier A", "DC Central"),
        ("Supplier B", "DC South"),
    ]

    dc_to_store = [
        ("DC North", "Store 1"),
        ("DC North", "Store 2"),
        ("DC Central", "Store 3"),
        ("DC Central", "Store 4"),
        ("DC South", "Store 5"),
    ]

    return suppliers, dcs, stores, supplier_to_dc, dc_to_store


def create_demand_profile(store_names: List[str], horizon: int, demand_multiplier: float, volatility: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_mean = {
        "Store 1": 45,
        "Store 2": 40,
        "Store 3": 55,
        "Store 4": 50,
        "Store 5": 38,
    }

    data = {"step": np.arange(horizon)}
    for store in store_names:
        base = base_mean[store] * demand_multiplier
        noise = rng.normal(loc=0.0, scale=volatility * base, size=horizon)
        ramp = np.sin(np.linspace(0, 3 * np.pi, horizon)) * 0.15 * base
        demand = np.maximum(0, base + noise + ramp)
        data[store] = np.round(demand, 0)

    return pd.DataFrame(data)


def compute_dynamic_safety_stock(recent_demand: np.ndarray, lead_time: float, z: float = 1.65) -> float:
    if len(recent_demand) < 2:
        return 0.0
    std_demand = np.std(recent_demand, ddof=1)
    return z * std_demand * np.sqrt(max(lead_time, 1.0))


def calculate_order_point(mean_demand: float, safety_stock: float, lead_time: float) -> float:
    return mean_demand * lead_time + safety_stock


def place_order(node: Node, forecast: float, review_period: int) -> float:
    desired = max(node.min_order, round(forecast * review_period))
    if desired > node.capacity:
        desired = node.capacity
    return desired


def simulate_supply_chain(
    steps: int,
    demand_df: pd.DataFrame,
    suppliers: Dict[str, Node],
    dcs: Dict[str, Node],
    stores: Dict[str, Node],
    supplier_to_dc: List[Tuple[str, str]],
    dc_to_store: List[Tuple[str, str]],
    demand_volatility: float,
    lead_time_factor: float,
    emission_cost_weight: float,
    review_period: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    history = []
    all_nodes = {**suppliers, **dcs, **stores}

    # Create lookup maps for faster routing
    store_to_dc_map = {target: dc for dc, target in dc_to_store}
    dc_to_supplier_map = {target: supplier for supplier, target in supplier_to_dc}

    # Reset shipment queues and metrics
    for node in all_nodes.values():
        node.inbound.clear()
        node.order_backlog = 0.0
        node.delivered = 0.0
        node.lost_sales = 0.0

    total_holding_cost = 0.0
    total_emissions = 0.0
    total_order_cost = 0.0
    total_stockout_cost = 0.0
    cumulative_demand = 0.0
    cumulative_fulfilled = 0.0

    for step in range(steps):
        step_demands = demand_df.loc[demand_df["step"] == step].iloc[0]

        # Advance shipments into inventory
        for node in all_nodes.values():
            arrivals = []
            for shipment in node.inbound:
                shipment["remaining"] -= 1
                if shipment["remaining"] <= 0:
                    node.on_hand += shipment["quantity"]
                    arrivals.append(shipment)
            for arrived in arrivals:
                node.inbound.remove(arrived)

        # Process store demand and reorder decisions
        for store_name, store in stores.items():
            demand = float(step_demands[store_name])
            cumulative_demand += demand
            fulfilled = min(store.on_hand, demand)
            store.on_hand -= fulfilled
            store.delivered += fulfilled
            cumulative_fulfilled += fulfilled
            store.lost_sales += demand - fulfilled
            store.order_backlog += max(0.0, demand - fulfilled)

            recent = demand_df.loc[max(0, step - 11) : step, store_name].values
            forecast = np.mean(recent) if len(recent) > 0 else 0.0
            safety = compute_dynamic_safety_stock(recent, store.lead_time * lead_time_factor)
            store.safety_stock = safety
            store.reorder_point = calculate_order_point(forecast, safety, store.lead_time * lead_time_factor)
            order_qty = place_order(store, forecast, review_period)
            store.order_quantity = order_qty

            needed = max(0.0, store.reorder_point - (store.on_hand + sum(s["quantity"] for s in store.inbound)))
            if needed >= store.min_order:
                supplier_name = store_to_dc_map.get(store_name, "")
                if supplier_name:
                    dc = dcs[supplier_name]
                    dc.inbound.append({
                        "quantity": order_qty,
                        "remaining": max(1, int(dc.lead_time * lead_time_factor)),
                        "source": store_name,
                    })
                    total_order_cost += dc.order_cost
                    total_emissions += dc.emission_rate * order_qty

        # Process DC shipments for store replenishment
        for dc_name, dc in dcs.items():
            pending_store_orders = [s for s in stores.values() if store_to_dc_map.get(s.name) == dc_name]
            for store in pending_store_orders:
                needed_qty = max(0.0, store.order_quantity - sum(s["quantity"] for s in store.inbound))
                if needed_qty <= 0:
                    continue
                shipped = min(dc.on_hand, needed_qty)
                if shipped <= 0:
                    continue
                dc.on_hand -= shipped
                store.inbound.append({
                    "quantity": shipped,
                    "remaining": max(1, int(store.lead_time * lead_time_factor)),
                    "source": dc_name,
                })
                total_emissions += dc.emission_rate * shipped
                total_order_cost += dc.order_cost

            recent = np.concatenate([
                demand_df.loc[max(0, step - 11) : step, store_name].values
                for store_name in [s.name for s in pending_store_orders]
            ])
            forecast = np.mean(recent) if len(recent) > 0 else 0.0
            safety = compute_dynamic_safety_stock(recent, dc.lead_time * lead_time_factor)
            dc.safety_stock = safety
            dc.reorder_point = calculate_order_point(forecast, safety, dc.lead_time * lead_time_factor)
            order_qty = place_order(dc, forecast, review_period)
            dc.order_quantity = order_qty

            needed = max(0.0, dc.reorder_point - (dc.on_hand + sum(s["quantity"] for s in dc.inbound)))
            if needed >= dc.min_order:
                supplier_name = dc_to_supplier_map.get(dc_name, "")
                if supplier_name:
                    supplier = suppliers[supplier_name]
                    qty = min(order_qty, supplier.on_hand)
                    supplier.on_hand -= qty
                    dc.inbound.append({
                        "quantity": qty,
                        "remaining": max(1, int(supplier.lead_time * lead_time_factor)),
                        "source": supplier_name,
                    })
                    total_order_cost += supplier.order_cost
                    total_emissions += supplier.emission_rate * qty

        # Compute holding costs and metrics for the step
        for node in all_nodes.values():
            total_holding_cost += node.holding_cost * node.on_hand

        # Capture inventory levels for each node to visualize trends
        node_inventory = {f"inv_{name}": node.on_hand for name, node in all_nodes.items()}

        if cumulative_demand > 0:
            service_level = cumulative_fulfilled / cumulative_demand
        else:
            service_level = 1.0

        total_stockout_cost = sum(node.order_backlog * 5.0 for node in stores.values())
        total_cost = total_holding_cost + total_order_cost + total_stockout_cost + emission_cost_weight * total_emissions

        step_metrics = {
            **node_inventory,
            "step": step,
            "service_level": service_level,
            "holding_cost": total_holding_cost,
            "order_cost": total_order_cost,
            "stockout_cost": total_stockout_cost,
            "emissions": total_emissions,
            "total_cost": total_cost,
            "total_demand": cumulative_demand,
            "fulfilled": cumulative_fulfilled,
        }
        history.append(step_metrics)

    metrics = {
        "service_level": service_level,
        "holding_cost": total_holding_cost,
        "order_cost": total_order_cost,
        "stockout_cost": total_stockout_cost,
        "emissions": total_emissions,
        "total_cost": total_cost,
        "lost_sales": sum(store.lost_sales for store in stores.values()),
    }
    return pd.DataFrame(history), metrics


def build_node_summary(nodes: Dict[str, Node]) -> pd.DataFrame:
    data = []
    for node in nodes.values():
        data.append(
            {
                "name": node.name,
                "type": node.node_type,
                "on_hand": node.on_hand,
                "inbound_qty": sum(s["quantity"] for s in node.inbound),
                "reorder_point": round(node.reorder_point, 1),
                "safety_stock": round(node.safety_stock, 1),
                "order_quantity": node.order_quantity,
            }
        )
    return pd.DataFrame(data)


def main():
    st.set_page_config(page_title="Mini Digital Twin For Inventory Optimisation", layout="wide")
    st.title("Mini Inventory Digital Twin")
    st.write("Real-time simulated supply chain monitoring, dynamic safety stock, and cost/emissions optimization.")

    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()

    with st.sidebar:
        st.header("Scenario Controls")
        demand_multiplier = st.slider("Base demand multiplier", 0.6, 1.6, 1.0, 0.05)
        volatility = st.slider("Demand volatility", 0.05, 0.40, 0.18, 0.01)
        lead_time_factor = st.slider("Lead time multiplier", 0.8, 2.0, 1.0, 0.05)
        emission_cost_weight = st.slider("Emission cost weight", 0.0, 3.0, 1.2, 0.1)
        review_period = st.selectbox("Review period (steps)", [1, 2, 3, 4, 5], index=1)
        refresh_seconds = st.selectbox("Refresh interval (seconds)", [5, 10, 15], index=0)
        st.markdown("---")
        st.write("Use the controls above to test what-if scenarios for demand, lead time, and emissions.")
        st.write("The dashboard auto-refreshes to emulate live monitoring.")

    horizon = 48
    if "demand_df" not in st.session_state:
        st.session_state.demand_df = create_demand_profile(
            store_names=["Store 1", "Store 2", "Store 3", "Store 4", "Store 5"],
            horizon=horizon,
            demand_multiplier=demand_multiplier,
            volatility=volatility,
            seed=42,
        )

    suppliers, dcs, stores, supplier_to_dc, dc_to_store = build_supply_chain()
    current_step = min(int((time.time() - st.session_state.start_time) / refresh_seconds), horizon - 1)

    history, metrics = simulate_supply_chain(
        steps=current_step + 1,
        demand_df=st.session_state.demand_df,
        suppliers=suppliers,
        dcs=dcs,
        stores=stores,
        supplier_to_dc=supplier_to_dc,
        dc_to_store=dc_to_store,
        demand_volatility=volatility,
        lead_time_factor=lead_time_factor,
        emission_cost_weight=emission_cost_weight,
        review_period=review_period,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current step", current_step)
    col2.metric("Service level", f"{metrics['service_level'] * 100:.1f}%")
    col3.metric("Holding cost", f"${metrics['holding_cost']:.0f}")
    col4.metric("Emissions", f"{metrics['emissions']:.1f} units")

    st.subheader("Key KPI Trends")
    kpi_df = history.copy()
    fig = px.line(kpi_df, x="step", y=["service_level", "holding_cost", "emissions", "total_cost"],
                  labels={"value": "Metric", "step": "Simulation step"},
                  title="Inventory and Cost KPIs over Time")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Inventory Levels by Node")
    inv_cols = [c for c in history.columns if c.startswith("inv_")]
    inv_history = history.melt(id_vars=["step"], value_vars=inv_cols, var_name="node", value_name="on_hand")
    inv_history["node"] = inv_history["node"].str.replace("inv_", "")
    inv_fig = px.line(inv_history, x="step", y="on_hand", color="node", 
                      title="Real-time Stock Levels (Suppliers, DCs, and Stores)")
    st.plotly_chart(inv_fig, width="stretch")

    demand_history = st.session_state.demand_df.melt(id_vars=["step"], var_name="store", value_name="demand")
    demand_fig = px.line(demand_history, x="step", y="demand", color="store", title="Simulated Store Demand")
    st.plotly_chart(demand_fig, width="stretch")

    st.subheader("Current Inventory Positions")
    dc_summary = build_node_summary(dcs)
    store_summary = build_node_summary(stores)
    supplier_summary = build_node_summary(suppliers)

    st.markdown("**Distribution Centers**")
    st.dataframe(dc_summary, width="stretch")
    st.markdown("**Stores**")
    st.dataframe(store_summary, width="stretch")
    st.markdown("**Suppliers**")
    st.dataframe(supplier_summary, width="stretch")

    st.subheader("What-if Scenario Insights")
    st.write(
        "This model uses dynamic safety stock calculated from recent demand volatility, "
        "and reorder decisions are reviewed every step based on the current scenario.")
    st.write(
        "Change demand volatility, lead time, or emission weight to see cost and service tradeoffs in real time."
    )

    st.sidebar.markdown("---")
    # Add Export feature
    csv_data = history.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Simulation History",
        data=csv_data,
        file_name='inventory_simulation_results.csv',
        mime='text/csv',
    )
    st.sidebar.write("Run this app with: `streamlit run \"Mini Digital Twin For Inventory Optimisation.py\"`")

    # --- AUTO-REFRESH LOGIC ---
    # To prevent the infinite "loading" loop, we use a controlled sleep at the end of the script.
    # This ensures the browser has time to render the current step before requesting the next one.
    if current_step < horizon - 1:
        time.sleep(refresh_seconds)
        st.rerun()
    else:
        st.sidebar.success("✅ Simulation Complete")


if __name__ == "__main__":
    main()