"""Workforce Attrition Patterns & Risk Hotspot Analysis - Palo Alto Networks
Run:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Attrition Risk Hotspots", page_icon="📉", layout="wide")

DATA = Path(__file__).parent / "Palo_Alto_Networks.csv"
RED, GREY = "#E4572E", "#9AA5B1"


# ------------------------------------------------------------------ data
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)

    # --- validation & cleaning
    df = df.drop_duplicates()
    key_cols = ["Attrition", "Department", "JobRole", "OverTime", "BusinessTravel"]
    df = df.dropna(subset=key_cols)
    if df["Attrition"].dtype == object:  # support Yes/No labels too
        df["Attrition"] = df["Attrition"].str.strip().str.title().map({"Yes": 1, "No": 0})
    df = df[df["Attrition"].isin([0, 1])].copy()
    df["Attrition"] = df["Attrition"].astype(int)

    # standardise category text
    for c in ["Department", "JobRole", "BusinessTravel", "EducationField", "MaritalStatus", "Gender", "OverTime"]:
        df[c] = df[c].astype(str).str.strip().str.replace("_", " ")
    df["Status"] = df["Attrition"].map({1: "Exited", 0: "Retained"})

    # --- derived bands
    df["AgeGroup"] = pd.cut(df.Age, [17, 25, 35, 45, 55, 100],
                            labels=["18-25", "26-35", "36-45", "46-55", "56+"])
    df["TenureBand"] = pd.cut(df.YearsAtCompany, [-1, 1, 3, 5, 10, 100],
                              labels=["0-1 yrs", "2-3 yrs", "4-5 yrs", "6-10 yrs", "11+ yrs"])
    df["CareerStage"] = pd.cut(df.YearsAtCompany, [-1, 2, 7, 100],
                               labels=["Early (0-2)", "Mid (3-7)", "Senior (8+)"])
    df["PromoGap"] = pd.cut(df.YearsSinceLastPromotion, [-1, 0, 2, 5, 100],
                            labels=["0 yrs", "1-2 yrs", "3-5 yrs", "6+ yrs"])
    df["DistanceBand"] = pd.cut(df.DistanceFromHome, [0, 5, 10, 20, 100],
                                labels=["1-5", "6-10", "11-20", "21+"])
    df["Education"] = df.Education.map({1: "Below College", 2: "College", 3: "Bachelor",
                                        4: "Master", 5: "Doctor"})
    return df


def rate_table(d: pd.DataFrame, by, min_n: int = 1) -> pd.DataFrame:
    g = d.groupby(by, observed=True).Attrition.agg(Employees="count", Exits="sum", Rate="mean").reset_index()
    g["Rate"] = (g.Rate * 100).round(1)
    return g[g.Employees >= min_n]


def bar(d, x, title, horizontal=False, baseline=None):
    t = rate_table(d, x)
    if horizontal:
        t = t.sort_values("Rate")
        fig = px.bar(t, x="Rate", y=x, orientation="h", text="Rate", hover_data=["Employees", "Exits"])
    else:
        fig = px.bar(t, x=x, y="Rate", text="Rate", hover_data=["Employees", "Exits"])
    fig.update_traces(marker_color=RED, texttemplate="%{text}%", textposition="outside")
    if baseline is not None:
        line = dict(line_dash="dash", line_color=GREY)
        if horizontal:
            fig.add_vline(x=baseline, **line)
        else:
            fig.add_hline(y=baseline, **line)
    fig.update_layout(title=title, yaxis_title=None if horizontal else "Attrition rate (%)",
                      xaxis_title="Attrition rate (%)" if horizontal else None,
                      margin=dict(t=50, b=10), height=340)
    return fig


df = load_data()

# ------------------------------------------------------------------ sidebar filters
st.sidebar.header("Filters")
depts = st.sidebar.multiselect("Department", sorted(df.Department.unique()), default=sorted(df.Department.unique()))
roles_avail = sorted(df[df.Department.isin(depts)].JobRole.unique())
roles = st.sidebar.multiselect("Job role", roles_avail, default=roles_avail)
ten_min, ten_max = int(df.YearsAtCompany.min()), int(df.YearsAtCompany.max())
tenure = st.sidebar.slider("Tenure (years at company)", ten_min, ten_max, (ten_min, ten_max))
ot = st.sidebar.radio("Overtime", ["All", "Yes", "No"], horizontal=True)
travel = st.sidebar.multiselect("Business travel", sorted(df.BusinessTravel.unique()),
                                default=sorted(df.BusinessTravel.unique()))

st.sidebar.subheader("Demographics")
genders = st.sidebar.multiselect("Gender", sorted(df.Gender.unique()), default=sorted(df.Gender.unique()))
age_min, age_max = int(df.Age.min()), int(df.Age.max())
ages = st.sidebar.slider("Age", age_min, age_max, (age_min, age_max))
edu_order = ["Below College", "College", "Bachelor", "Master", "Doctor"]
edus = st.sidebar.multiselect("Education", edu_order, default=edu_order)

f = df[
    df.Department.isin(depts) & df.JobRole.isin(roles)
    & df.YearsAtCompany.between(*tenure) & df.BusinessTravel.isin(travel)
    & df.Gender.isin(genders) & df.Age.between(*ages) & df.Education.isin(edus)
]
if ot != "All":
    f = f[f.OverTime == ot]

st.title("📉 Workforce Attrition Patterns & Risk Hotspots")
st.caption("Palo Alto Networks · diagnostic view of where attrition is concentrated")

if f.empty:
    st.warning("No employees match the current filters.")
    st.stop()

org_rate = df.Attrition.mean() * 100
rate = f.Attrition.mean() * 100

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Department & Role Heatmaps",
                                   "Demographic Explorer", "Tenure & Workload"])

# ------------------------------------------------------------------ 1 overview
with tab1:
    early = f[f.YearsAtCompany <= 2]
    wl = f[(f.OverTime == "Yes") & (f.BusinessTravel == "Travel Frequently")]
    c = st.columns(5)
    c[0].metric("Employees (filtered)", f"{len(f):,}")
    c[1].metric("Exits", f"{int(f.Attrition.sum()):,}")
    c[2].metric("Attrition rate", f"{rate:.1f}%", f"{rate - org_rate:+.1f} pts vs org", delta_color="inverse")
    c[3].metric("Early-tenure attrition (0-2 yrs)", f"{early.Attrition.mean() * 100:.1f}%" if len(early) else "n/a")
    c[4].metric("Workload Index (OT + frequent travel)", f"{wl.Attrition.mean() * 100:.1f}%" if len(wl) else "n/a",
                help=f"Attrition among overtime workers who also travel frequently (n={len(wl)}).")

    a, b = st.columns(2)
    dist = f.Status.value_counts().reset_index()
    pie = px.pie(dist, names="Status", values="count", hole=0.55, color="Status",
                 color_discrete_map={"Exited": RED, "Retained": GREY}, title="Retained vs exited")
    pie.update_layout(height=340, margin=dict(t=50, b=10))
    a.plotly_chart(pie, width='stretch')
    b.plotly_chart(bar(f, "Department", "Attrition rate by department", baseline=org_rate),
                   width='stretch')
    st.caption(f"Dashed line = organisation-wide baseline ({org_rate:.1f}%).")

# ------------------------------------------------------------------ 2 heatmaps
with tab2:
    min_n = st.slider("Hide groups smaller than (employees)", 1, 30, 10,
                      help="Small groups give unstable rates.")
    t = rate_table(f, ["Department", "JobRole"], min_n)
    if t.empty:
        st.info("No groups meet the minimum size.")
    else:
        hm = t.pivot(index="JobRole", columns="Department", values="Rate")
        fig = px.imshow(hm, text_auto=".1f", aspect="auto", color_continuous_scale="Reds",
                        labels=dict(color="Attrition %"), title="Attrition intensity: department × job role (%)")
        fig.update_layout(height=450)
        st.plotly_chart(fig, width='stretch')

    a, b = st.columns(2)
    a.plotly_chart(bar(f, "JobRole", "Attrition rate by role", horizontal=True, baseline=org_rate),
                   width='stretch')
    ex = f.groupby("JobRole").Attrition.sum().sort_values().reset_index(name="Exits")
    fx = px.bar(ex, x="Exits", y="JobRole", orientation="h", text="Exits", title="Exit frequency by role")
    fx.update_traces(marker_color=RED)
    fx.update_layout(height=340, margin=dict(t=50, b=10))
    b.plotly_chart(fx, width='stretch')

    st.subheader("High-risk groups")
    risk = rate_table(f, "JobRole", min_n)
    risk["Status"] = pd.cut(risk.Rate / org_rate, [-1, 0.8, 1.2, 100], labels=["Below baseline", "Near baseline", "High risk"])
    st.dataframe(risk.sort_values("Rate", ascending=False), hide_index=True, width='stretch')

# ------------------------------------------------------------------ 3 demographics
with tab3:
    a, b = st.columns(2)
    a.plotly_chart(bar(f, "AgeGroup", "By age group", baseline=org_rate), width='stretch')
    b.plotly_chart(bar(f, "MaritalStatus", "By marital status", baseline=org_rate), width='stretch')
    a, b = st.columns(2)
    a.plotly_chart(bar(f, "Gender", "By gender", baseline=org_rate), width='stretch')
    b.plotly_chart(bar(f, "Education", "By education level", baseline=org_rate), width='stretch')
    st.plotly_chart(bar(f, "EducationField", "By education field", horizontal=True, baseline=org_rate),
                    width='stretch')

# ------------------------------------------------------------------ 4 tenure & workload
with tab4:
    a, b = st.columns(2)
    a.plotly_chart(bar(f, "TenureBand", "By tenure band", baseline=org_rate), width='stretch')
    b.plotly_chart(bar(f, "CareerStage", "Early vs mid vs senior", baseline=org_rate), width='stretch')
    a, b = st.columns(2)
    a.plotly_chart(bar(f, "PromoGap", "By years since last promotion", baseline=org_rate), width='stretch')
    b.plotly_chart(bar(f, "DistanceBand", "By distance from home (km)", baseline=org_rate), width='stretch')

    st.subheader("Workload impact")
    a, b = st.columns(2)
    a.plotly_chart(bar(f, "OverTime", "Overtime vs not", baseline=org_rate), width='stretch')
    b.plotly_chart(bar(f, "BusinessTravel", "By business travel", baseline=org_rate), width='stretch')

    combo = rate_table(f, ["OverTime", "BusinessTravel"], 5)
    if not combo.empty:
        hm = combo.pivot(index="OverTime", columns="BusinessTravel", values="Rate")
        fig = px.imshow(hm, text_auto=".1f", color_continuous_scale="Reds", aspect="auto",
                        title="Workload Attrition Index: overtime × travel (%)")
        fig.update_layout(height=300)
        st.plotly_chart(fig, width='stretch')
    st.caption("Rates are descriptive, not causal. Junior staff are over-represented among overtime and low-pay groups.")

with st.expander("Data & method notes"):
    st.write(f"{len(df):,} records after validation. Attrition rate = exits ÷ employees. "
             "Bands are fixed so results are comparable across filters. "
             "Small groups produce volatile rates; check Employees and Exits counts in hover text.")
