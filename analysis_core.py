"""
Nassau Candy Distributor — Core Analysis Engine
Product Line Profitability & Margin Performance Analysis

This module is the single source of truth for all calculations used in:
  - The Streamlit dashboard (app.py)
  - The research paper
  - The executive summary

Run directly to regenerate all processed data + charts:
    python analysis_core.py
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ---------------------------------------------------------------------------
# 1. FACTORY & GEOGRAPHY REFERENCE DATA  (from project brief)
# ---------------------------------------------------------------------------

FACTORIES = {
    "Lot's O' Nuts":      (32.881893, -111.768036),
    "Wicked Choccy's":    (32.076176, -81.088371),
    "Sugar Shack":        (48.119140, -96.181150),
    "Secret Factory":     (41.446333, -90.565487),
    "The Other Factory":  (35.117500, -89.971107),
}

PRODUCT_FACTORY_MAP = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":         "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":    "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":        "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy":                       "Sugar Shack",
    "SweeTARTS":                         "Sugar Shack",
    "Nerds":                             "Sugar Shack",
    "Fun Dip":                           "Sugar Shack",
    "Fizzy Lifting Drinks":              "Sugar Shack",
    "Everlasting Gobstopper":            "Secret Factory",
    "Hair Toffee":                       "The Other Factory",
    "Lickable Wallpaper":                "Secret Factory",
    "Wonka Gum":                         "Secret Factory",
    "Kazookles":                         "The Other Factory",
}

# Approximate US state/territory centroids (lat, lon) — used to estimate
# factory-to-customer shipping distance since the dataset has no customer
# coordinates. This is a standard simplification for route-efficiency analysis.
STATE_CENTROIDS = {
    "Alabama": (32.806671, -86.791130), "Alaska": (61.370716, -152.404419),
    "Arizona": (33.729759, -111.431221), "Arkansas": (34.969704, -92.373123),
    "California": (36.116203, -119.681564), "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371), "Delaware": (39.318523, -75.507141),
    "District of Columbia": (38.897438, -77.026817), "Florida": (27.766279, -81.686783),
    "Georgia": (33.040619, -83.643074), "Hawaii": (21.094318, -157.498337),
    "Idaho": (44.240459, -114.478828), "Illinois": (40.349457, -88.986137),
    "Indiana": (39.849426, -86.258278), "Iowa": (42.011539, -93.210526),
    "Kansas": (38.526600, -96.726486), "Kentucky": (37.668140, -84.670067),
    "Louisiana": (31.169546, -91.867805), "Maine": (44.693947, -69.381927),
    "Maryland": (39.063946, -76.802101), "Massachusetts": (42.230171, -71.530106),
    "Michigan": (43.326618, -84.536095), "Minnesota": (45.694454, -93.900192),
    "Mississippi": (32.741646, -89.678696), "Missouri": (38.456085, -92.288368),
    "Montana": (46.921925, -110.454353), "Nebraska": (41.125370, -98.268082),
    "Nevada": (38.313515, -117.055374), "New Hampshire": (43.452492, -71.563896),
    "New Jersey": (40.298904, -74.521011), "New Mexico": (34.840515, -106.248482),
    "New York": (42.165726, -74.948051), "North Carolina": (35.630066, -79.806419),
    "North Dakota": (47.528912, -99.784012), "Ohio": (40.388783, -82.764915),
    "Oklahoma": (35.565342, -96.928917), "Oregon": (44.572021, -122.070938),
    "Pennsylvania": (40.590752, -77.209755), "Rhode Island": (41.680893, -71.511780),
    "South Carolina": (33.856892, -80.945007), "South Dakota": (44.299782, -99.438828),
    "Tennessee": (35.747845, -86.692345), "Texas": (31.054487, -97.563461),
    "Utah": (40.150032, -111.862434), "Vermont": (44.045876, -72.710686),
    "Virginia": (37.769337, -78.169968), "Washington": (47.400902, -121.490494),
    "West Virginia": (38.491226, -80.954456), "Wisconsin": (44.268543, -89.616508),
    "Wyoming": (42.755966, -107.302490),
}

SHIP_MODE_TARGET_DAYS = {
    "Same Day": 1, "First Class": 2, "Second Class": 4, "Standard Class": 6,
}


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    p1, p2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlmb = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dlmb / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ---------------------------------------------------------------------------
# 2. LOAD & CLEAN
# ---------------------------------------------------------------------------

def load_and_clean(path="Nassau_Candy_Distributor.csv"):
    df = pd.read_csv(path)

    # --- Validation / cleaning ---
    before = len(df)
    df = df.drop_duplicates(subset="Row ID")
    df = df[(df["Sales"] > 0) & (df["Units"] > 0) & (df["Cost"] >= 0)]
    df = df.dropna(subset=["Sales", "Units", "Gross Profit", "Cost", "Division", "Product Name"])
    removed = before - len(df)

    # Standardize text labels
    for col in ["Division", "Region", "State/Province", "City", "Ship Mode", "Product Name"]:
        df[col] = df[col].astype(str).str.strip()

    # Parse dates (DD-MM-YYYY as confirmed from the data)
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y", errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y", errors="coerce")
    df["Lead Time (days)"] = (df["Ship Date"] - df["Order Date"]).dt.days
    # Guard against negative/garbage lead times from bad rows
    df.loc[df["Lead Time (days)"] < 0, "Lead Time (days)"] = np.nan

    # --- Core KPIs ---
    df["Gross Margin %"] = (df["Gross Profit"] / df["Sales"]) * 100
    df["Profit per Unit"] = df["Gross Profit"] / df["Units"]
    df["Cost per Unit"] = df["Cost"] / df["Units"]

    # --- Factory mapping ---
    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY_MAP)
    df["Factory Lat"] = df["Factory"].map(lambda f: FACTORIES.get(f, (np.nan, np.nan))[0])
    df["Factory Lon"] = df["Factory"].map(lambda f: FACTORIES.get(f, (np.nan, np.nan))[1])

    # --- Approximate shipping distance (factory -> customer state centroid) ---
    def get_centroid(state):
        return STATE_CENTROIDS.get(state, (np.nan, np.nan))

    cust_lat, cust_lon = zip(*df["State/Province"].map(get_centroid))
    df["Customer Lat"] = cust_lat
    df["Customer Lon"] = cust_lon

    def calc_dist(row):
        if pd.isna(row["Factory Lat"]) or pd.isna(row["Customer Lat"]):
            return np.nan
        return haversine_miles(row["Factory Lat"], row["Factory Lon"],
                                row["Customer Lat"], row["Customer Lon"])

    df["Ship Distance (mi)"] = df.apply(calc_dist, axis=1)
    df["Target Lead Time"] = df["Ship Mode"].map(SHIP_MODE_TARGET_DAYS)
    df["Delay (days)"] = df["Lead Time (days)"] - df["Target Lead Time"]
    df["Delayed"] = df["Delay (days)"] > 0

    meta = {"rows_before": before, "rows_removed": removed, "rows_after": len(df)}
    return df, meta


# ---------------------------------------------------------------------------
# 3. PRODUCT / DIVISION / PARETO / COST ANALYSES
# ---------------------------------------------------------------------------

def product_summary(df):
    g = df.groupby("Product Name").agg(
        Division=("Division", "first"),
        Factory=("Factory", "first"),
        Orders=("Order ID", "nunique"),
        Units=("Units", "sum"),
        Sales=("Sales", "sum"),
        Cost=("Cost", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
    ).reset_index()
    g["Gross Margin %"] = (g["Gross_Profit"] / g["Sales"]) * 100
    g["Profit per Unit"] = g["Gross_Profit"] / g["Units"]
    total_sales, total_profit = g["Sales"].sum(), g["Gross_Profit"].sum()
    g["Revenue Contribution %"] = g["Sales"] / total_sales * 100
    g["Profit Contribution %"] = g["Gross_Profit"] / total_profit * 100
    return g.sort_values("Gross_Profit", ascending=False).rename(
        columns={"Gross_Profit": "Gross Profit"})


def division_summary(df):
    g = df.groupby("Division").agg(
        Orders=("Order ID", "nunique"),
        Units=("Units", "sum"),
        Sales=("Sales", "sum"),
        Cost=("Cost", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Products=("Product Name", "nunique"),
    ).reset_index()
    g["Gross Margin %"] = (g["Gross_Profit"] / g["Sales"]) * 100
    g["Avg Profit per Unit"] = g["Gross_Profit"] / g["Units"]
    total_sales, total_profit = g["Sales"].sum(), g["Gross_Profit"].sum()
    g["Revenue Share %"] = g["Sales"] / total_sales * 100
    g["Profit Share %"] = g["Gross_Profit"] / total_profit * 100
    return g.rename(columns={"Gross_Profit": "Gross Profit"}).sort_values(
        "Gross Profit", ascending=False)


def pareto_analysis(df, value_col="Sales"):
    g = df.groupby("Product Name")[value_col].sum().sort_values(ascending=False).reset_index()
    g["Cumulative"] = g[value_col].cumsum()
    g["Cumulative %"] = g["Cumulative"] / g[value_col].sum() * 100
    g["Product Rank %"] = (np.arange(1, len(g) + 1) / len(g)) * 100
    n_for_80 = (g["Cumulative %"] <= 80).sum() + 1
    pct_products_for_80 = n_for_80 / len(g) * 100
    return g, {"n_products": len(g), "n_for_80pct": n_for_80, "pct_products_for_80": pct_products_for_80}


def factory_summary(df):
    g = df.groupby("Factory").agg(
        Orders=("Order ID", "nunique"),
        Units=("Units", "sum"),
        Sales=("Sales", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Avg_Lead_Time=("Lead Time (days)", "mean"),
        Avg_Ship_Distance=("Ship Distance (mi)", "mean"),
        Delay_Rate=("Delayed", "mean"),
    ).reset_index()
    g["Gross Margin %"] = (g["Gross_Profit"] / g["Sales"]) * 100
    g["Delay_Rate"] = g["Delay_Rate"] * 100
    return g.rename(columns={
        "Gross_Profit": "Gross Profit", "Avg_Lead_Time": "Avg Lead Time (days)",
        "Avg_Ship_Distance": "Avg Ship Distance (mi)", "Delay_Rate": "Delay Rate %"
    }).sort_values("Gross Profit", ascending=False)


def cost_diagnostics(df):
    """Flag margin-risk products: above-median sales, below-median margin."""
    p = product_summary(df)
    med_sales = p["Sales"].median()
    med_margin = p["Gross Margin %"].median()
    p["Risk Flag"] = np.where(
        (p["Sales"] >= med_sales) & (p["Gross Margin %"] < med_margin),
        "High-Sales / Low-Margin (Risk)",
        np.where(
            (p["Gross Profit"] >= p["Gross Profit"].median()) & (p["Gross Margin %"] >= med_margin),
            "High-Profit / High-Margin (Star)",
            np.where(
                (p["Sales"] < med_sales) & (p["Gross Profit"] < p["Gross Profit"].median()),
                "Low-Sales / Low-Profit (Tail)",
                "Average"
            )
        )
    )
    return p


def state_summary(df):
    g = df.groupby(["State/Province", "Region"]).agg(
        Orders=("Order ID", "nunique"),
        Sales=("Sales", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Avg_Lead_Time=("Lead Time (days)", "mean"),
    ).reset_index().rename(columns={"Gross_Profit": "Gross Profit", "Avg_Lead_Time": "Avg Lead Time (days)"})
    g["Gross Margin %"] = (g["Gross Profit"] / g["Sales"]) * 100
    return g.sort_values("Sales", ascending=False)


def monthly_trend(df):
    t = df.copy()
    t["Month"] = t["Order Date"].dt.to_period("M").astype(str)
    g = t.groupby("Month").agg(Sales=("Sales", "sum"), Gross_Profit=("Gross Profit", "sum")).reset_index()
    g["Gross Margin %"] = (g["Gross_Profit"] / g["Sales"]) * 100
    return g.rename(columns={"Gross_Profit": "Gross Profit"})


# ---------------------------------------------------------------------------
# MAIN — regenerate processed data for reuse by dashboard & report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df, meta = load_and_clean()
    print("Cleaning summary:", meta)

    df.to_csv("/home/claude/processed_data.csv", index=False)
    product_summary(df).to_csv("/home/claude/product_summary.csv", index=False)
    division_summary(df).to_csv("/home/claude/division_summary.csv", index=False)
    factory_summary(df).to_csv("/home/claude/factory_summary.csv", index=False)
    cost_diagnostics(df).to_csv("/home/claude/cost_diagnostics.csv", index=False)
    state_summary(df).to_csv("/home/claude/state_summary.csv", index=False)
    monthly_trend(df).to_csv("/home/claude/monthly_trend.csv", index=False)
    pareto_df, pareto_meta = pareto_analysis(df)
    pareto_df.to_csv("/home/claude/pareto_sales.csv", index=False)
    print("Pareto:", pareto_meta)

    print("\nAll processed files written to /home/claude/")
