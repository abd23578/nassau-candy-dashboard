"""
Nassau Candy Distributor — Product Line Profitability & Margin Performance Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from analysis_core import (
    load_and_clean, product_summary, division_summary, pareto_analysis,
    factory_summary, cost_diagnostics, state_summary, monthly_trend, FACTORIES
)

# ---------------------------------------------------------------------------
# PAGE CONFIG & STYLE
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Nassau Candy | Profitability Dashboard",
    page_icon="🍬",
    layout="wide",
)

PRIMARY = "#5B2C6F"
ACCENT = "#E67E22"
GOOD = "#2E8B57"
RISK = "#C0392B"
NEUTRAL = "#7f8c8d"

RISK_COLORS = {
    "High-Profit / High-Margin (Star)": GOOD,
    "High-Sales / Low-Margin (Risk)": RISK,
    "Low-Sales / Low-Profit (Tail)": NEUTRAL,
    "Average": ACCENT,
}

st.markdown(f"""
<style>
.metric-card {{
    background: #faf7fc; border: 1px solid #e8dff0; border-radius: 10px;
    padding: 14px 18px; text-align: center;
}}
.metric-card h3 {{ margin: 0; font-size: 13px; color: {NEUTRAL}; font-weight: 600; text-transform: uppercase; letter-spacing: .03em;}}
.metric-card p {{ margin: 4px 0 0 0; font-size: 26px; font-weight: 700; color: {PRIMARY}; }}
.block-title {{ font-size: 22px; font-weight: 700; color: {PRIMARY}; margin-bottom: 0px;}}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    df, meta = load_and_clean()
    return df, meta


df, meta = get_data()

# ---------------------------------------------------------------------------
# SIDEBAR — GLOBAL FILTERS
# ---------------------------------------------------------------------------
st.sidebar.title("🍬 Nassau Candy")
st.sidebar.caption("Product Line Profitability & Margin Performance")

min_date, max_date = df["Order Date"].min(), df["Order Date"].max()
date_range = st.sidebar.date_input(
    "Order date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

divisions = st.sidebar.multiselect(
    "Division", options=sorted(df["Division"].unique()), default=sorted(df["Division"].unique())
)

margin_threshold = st.sidebar.slider("Minimum gross margin %", 0, 100, 0, step=5)

search = st.sidebar.text_input("🔎 Product search", "")

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Rows loaded: **{meta['rows_after']:,}** · Rows removed in cleaning: **{meta['rows_removed']}**"
)

# Apply filters
mask = (
    (df["Division"].isin(divisions)) &
    (df["Gross Margin %"] >= margin_threshold)
)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    mask &= (df["Order Date"] >= start) & (df["Order Date"] <= end)
if search.strip():
    mask &= df["Product Name"].str.contains(search.strip(), case=False, na=False)

fdf = df[mask]

if fdf.empty:
    st.warning("No records match the current filters. Try widening your selection.")
    st.stop()

# ---------------------------------------------------------------------------
# HEADER KPIs
# ---------------------------------------------------------------------------
st.title("Product Line Profitability & Margin Performance")
st.caption("Nassau Candy Distributor — interactive analytics dashboard")

total_sales = fdf["Sales"].sum()
total_profit = fdf["Gross Profit"].sum()
overall_margin = total_profit / total_sales * 100 if total_sales else 0
n_products = fdf["Product Name"].nunique()

c1, c2, c3, c4 = st.columns(4)
for col, label, value in zip(
    [c1, c2, c3, c4],
    ["Total Revenue", "Total Gross Profit", "Overall Margin", "Active Products"],
    [f"${total_sales:,.0f}", f"${total_profit:,.0f}", f"{overall_margin:.1f}%", f"{n_products}"],
):
    col.markdown(f'<div class="metric-card"><h3>{label}</h3><p>{value}</p></div>', unsafe_allow_html=True)

st.markdown("")

tabs = st.tabs([
    "📦 Product Profitability",
    "🏷️ Division Performance",
    "⚖️ Cost vs Margin Diagnostics",
    "📊 Profit Concentration (Pareto)",
    "🚚 Factory & Logistics",
])

# ---------------------------------------------------------------------------
# TAB 1 — PRODUCT PROFITABILITY OVERVIEW
# ---------------------------------------------------------------------------
with tabs[0]:
    st.markdown('<p class="block-title">Product-Level Margin Leaderboard</p>', unsafe_allow_html=True)
    ps = product_summary(fdf).sort_values("Gross Profit", ascending=True)

    fig = px.bar(
        ps, x="Gross Profit", y="Product Name", orientation="h",
        color="Gross Margin %", color_continuous_scale="RdYlGn",
        hover_data={"Sales": ":.2f", "Gross Margin %": ":.1f", "Profit per Unit": ":.2f"},
        height=max(420, 28 * len(ps)),
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        st.markdown('<p class="block-title">Revenue Contribution</p>', unsafe_allow_html=True)
        fig2 = px.pie(ps, names="Product Name", values="Sales", hole=0.45)
        fig2.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)
    with colB:
        st.markdown('<p class="block-title">Profit Contribution</p>', unsafe_allow_html=True)
        fig3 = px.pie(ps, names="Product Name", values="Gross Profit", hole=0.45)
        fig3.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
        fig3.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<p class="block-title">Full Product Table</p>', unsafe_allow_html=True)
    st.dataframe(
        ps[["Product Name", "Division", "Factory", "Sales", "Gross Profit", "Gross Margin %",
            "Profit per Unit", "Revenue Contribution %", "Profit Contribution %"]]
        .round(2).sort_values("Gross Profit", ascending=False),
        use_container_width=True, hide_index=True,
    )

# ---------------------------------------------------------------------------
# TAB 2 — DIVISION PERFORMANCE
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown('<p class="block-title">Revenue vs. Profit by Division</p>', unsafe_allow_html=True)
    ds = division_summary(fdf)

    fig = go.Figure()
    fig.add_bar(x=ds["Division"], y=ds["Sales"], name="Revenue", marker_color=PRIMARY)
    fig.add_bar(x=ds["Division"], y=ds["Gross Profit"], name="Gross Profit", marker_color=ACCENT)
    fig.update_layout(barmode="group", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        st.markdown('<p class="block-title">Margin by Division</p>', unsafe_allow_html=True)
        fig2 = px.bar(ds.sort_values("Gross Margin %"), x="Gross Margin %", y="Division",
                       orientation="h", color="Gross Margin %", color_continuous_scale="RdYlGn")
        fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    with colB:
        st.markdown('<p class="block-title">Revenue vs. Profit Share</p>', unsafe_allow_html=True)
        sh = ds.melt(id_vars="Division", value_vars=["Revenue Share %", "Profit Share %"])
        fig3 = px.bar(sh, x="Division", y="value", color="variable", barmode="group",
                       color_discrete_sequence=[PRIMARY, ACCENT])
        fig3.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title="")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<p class="block-title">Division Summary Table</p>', unsafe_allow_html=True)
    st.dataframe(ds.round(2), use_container_width=True, hide_index=True)

    st.markdown('<p class="block-title">Monthly Trend</p>', unsafe_allow_html=True)
    mt = monthly_trend(fdf)
    fig4 = px.line(mt, x="Month", y=["Sales", "Gross Profit"], markers=True,
                    color_discrete_sequence=[PRIMARY, ACCENT])
    fig4.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title="")
    st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 — COST VS MARGIN DIAGNOSTICS
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown('<p class="block-title">Cost Structure Diagnostics</p>', unsafe_allow_html=True)
    st.caption("Bubble size = sales volume · Color = risk category")

    cdiag = cost_diagnostics(fdf)
    fig = px.scatter(
        cdiag, x="Cost", y="Sales", size="Units", color="Risk Flag",
        color_discrete_map=RISK_COLORS, hover_name="Product Name",
        hover_data={"Gross Margin %": ":.1f", "Profit per Unit": ":.2f"},
        size_max=45,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    risk_products = cdiag[cdiag["Risk Flag"] == "High-Sales / Low-Margin (Risk)"]
    if not risk_products.empty:
        st.error(
            f"⚠️ **{len(risk_products)} product(s) flagged as margin risk** "
            f"(above-median sales, below-median margin): "
            f"{', '.join(risk_products['Product Name'])}. "
            "Consider repricing, cost renegotiation, or a discontinuation review."
        )

    st.markdown('<p class="block-title">Risk Category Breakdown</p>', unsafe_allow_html=True)
    st.dataframe(
        cdiag[["Product Name", "Division", "Sales", "Gross Profit", "Gross Margin %", "Risk Flag"]]
        .round(2).sort_values("Risk Flag"),
        use_container_width=True, hide_index=True,
    )

# ---------------------------------------------------------------------------
# TAB 4 — PROFIT CONCENTRATION (PARETO)
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown('<p class="block-title">Revenue Concentration (Pareto)</p>', unsafe_allow_html=True)
    pareto_sales, meta_sales = pareto_analysis(fdf, "Sales")
    pareto_profit, meta_profit = pareto_analysis(fdf, "Gross Profit")

    colA, colB = st.columns(2)
    with colA:
        st.info(f"**{meta_sales['n_for_80pct']} of {meta_sales['n_products']} products** "
                f"({meta_sales['pct_products_for_80']:.0f}%) generate 80% of total revenue.")
    with colB:
        st.info(f"**{meta_profit['n_for_80pct']} of {meta_profit['n_products']} products** "
                f"({meta_profit['pct_products_for_80']:.0f}%) generate 80% of total profit.")

    def pareto_chart(p, label):
        fig = go.Figure()
        fig.add_bar(x=p["Product Name"], y=p[label], name=label, marker_color=PRIMARY)
        fig.add_trace(go.Scatter(x=p["Product Name"], y=p["Cumulative %"], name="Cumulative %",
                                  yaxis="y2", line=dict(color=ACCENT, width=3), marker=dict(size=7)))
        fig.add_hline(y=80, line_dash="dash", line_color=RISK, yref="y2")
        fig.update_layout(
            yaxis=dict(title=label),
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
            margin=dict(l=10, r=10, t=10, b=80),
        )
        return fig

    st.plotly_chart(pareto_chart(pareto_sales, "Sales"), use_container_width=True)
    st.plotly_chart(pareto_chart(pareto_profit, "Gross Profit"), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 5 — FACTORY & LOGISTICS
# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown('<p class="block-title">Factory Profitability & Shipping Footprint</p>', unsafe_allow_html=True)

    st.warning(
        "⚠️ **Data quality note:** the `Ship Date` field in this dataset does not track `Order Date` "
        "the way a real shipping log would (average computed lead time is ~1,300 days, and "
        "Same-Day shipments show the same 'delay' as Standard Class). This indicates `Ship Date` was "
        "populated independently of `Order Date` upstream. We therefore exclude raw lead-time/delay "
        "metrics from headline KPIs below and rely on shipping distance and mode mix instead, which "
        "remain valid. We'd recommend the data engineering team review the ETL job that populates "
        "`Ship Date` before using it for SLA reporting."
    )

    fs = factory_summary(fdf)
    colA, colB = st.columns(2)
    with colA:
        fig = px.bar(fs.sort_values("Gross Profit"), x="Gross Profit", y="Factory", orientation="h",
                      color="Gross Margin %", color_continuous_scale="RdYlGn")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        fig2 = px.bar(fs.sort_values("Avg Ship Distance (mi)"), x="Avg Ship Distance (mi)", y="Factory",
                       orientation="h", color_discrete_sequence=[ACCENT])
        fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        fs[["Factory", "Sales", "Gross Profit", "Gross Margin %", "Units", "Avg Ship Distance (mi)"]]
        .round(2), use_container_width=True, hide_index=True,
    )

    st.markdown('<p class="block-title">Factory Locations</p>', unsafe_allow_html=True)
    fac_df = pd.DataFrame(
        [{"Factory": k, "lat": v[0], "lon": v[1]} for k, v in FACTORIES.items()]
    )
    fig3 = px.scatter_geo(fac_df, lat="lat", lon="lon", text="Factory", scope="usa",
                           color_discrete_sequence=[PRIMARY])
    fig3.update_traces(marker=dict(size=14))
    fig3.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<p class="block-title">Top States by Revenue</p>', unsafe_allow_html=True)
    ss = state_summary(fdf).head(15)
    fig4 = px.bar(ss, x="Sales", y="State/Province", orientation="h", color="Gross Margin %",
                   color_continuous_scale="RdYlGn")
    fig4.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")
st.caption(
    "Built for Nassau Candy Distributor · Product Line Profitability & Margin Performance Analysis"
)
