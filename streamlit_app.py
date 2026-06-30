import os
import sys
import json
import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# Setup paths
# ─────────────────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))

csv_path          = os.path.join(project_root, "outputs/top_100_candidates.csv")
explanations_path = os.path.join(project_root, "outputs/top_100_explanations.json")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Top 100 Candidates — RedRob Analytics",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS (dark glassmorphism)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0B1120;
    color: #CBD5E1;
}
h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif !important;
    color: #F1F5F9 !important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }

.kpi-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.85) 0%, rgba(15,23,42,0.85) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 22px 20px;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    margin-bottom: 14px;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(99,179,237,0.2);
}
.kpi-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.2px; color: #94A3B8; font-weight: 600; }
.kpi-value { font-family: 'Outfit', sans-serif; font-size: 2.2rem; font-weight: 800; color: #7DD3FC; line-height: 1.1; }
.kpi-sub   { font-size: 0.73rem; color: #64748B; margin-top: 4px; }

.cand-card {
    background: rgba(30,41,59,0.6);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 16px;
    transition: border-color 0.2s ease;
}
.cand-card:hover { border-color: #3B82F6; }

.rank-badge {
    display: inline-block;
    background: linear-gradient(135deg,#1D4ED8,#7C3AED);
    color: white;
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    border-radius: 50%;
    width: 40px; height: 40px;
    line-height: 40px;
    text-align: center;
}

.skill-pill {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.35);
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.73rem;
    font-weight: 600;
    color: #93C5FD;
    margin: 3px;
}
.skill-pill-pref {
    background: rgba(168,85,247,0.12);
    border-color: rgba(168,85,247,0.35);
    color: #C4B5FD;
}

.sec-header {
    border-left: 4px solid #3B82F6;
    padding-left: 12px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Color constants
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG = "#0B1120"
CARD_BG  = "#1E293B"
BLUE     = "#3B82F6"
PURPLE   = "#8B5CF6"
TEAL     = "#2DD4BF"
AMBER    = "#F59E0B"
RED      = "#F87171"
GREEN    = "#4ADE80"


def set_dark_figure(fig, ax_or_axes):
    fig.patch.set_facecolor(DARK_BG)
    axes = [ax_or_axes] if not isinstance(ax_or_axes, (list, np.ndarray)) else ax_or_axes
    for ax in np.ravel(axes):
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors="#CBD5E1", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor((1.0, 1.0, 1.0, 0.08))  # matplotlib RGBA tuple (0-1 scale)
            spine.set_linewidth(0.6)
    return fig, axes


def parse_reasoning(reasoning: str) -> dict:
    out = {"role": "Unknown", "experience": 0.0, "skills": 0, "response_rate": 0.0}
    # Use \d+(?:\.\d+)? so the pattern stops before trailing punctuation
    # e.g. "response rate 0.76." — the final '.' is NOT part of the number
    m = re.match(r"^(.+?) with (\d+(?:\.\d+)?) yrs", reasoning)
    if m:
        out["role"] = m.group(1).strip()
        out["experience"] = float(m.group(2))
    m2 = re.search(r"(\d+) AI core skills", reasoning)
    if m2:
        out["skills"] = int(m2.group(1))
    m3 = re.search(r"response rate (\d+(?:\.\d+)?)", reasoning)
    if m3:
        out["response_rate"] = float(m3.group(1).rstrip("."))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(csv_path)
    with open(explanations_path, "r", encoding="utf-8") as f:
        explanations = json.load(f)
    exp_dict = {item["candidate_id"]: item for item in explanations}

    parsed = df["reasoning"].apply(parse_reasoning).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    names, locations, companies, titles = [], [], [], []
    for cid in df["candidate_id"]:
        item = exp_dict.get(cid, {})
        ph = item.get("profile_highlights", {})
        names.append(ph.get("name", "Unknown"))
        locations.append(ph.get("location", "Unknown"))
        companies.append(ph.get("current_company", "Unknown"))
        titles.append(ph.get("current_title", "Unknown"))

    df["name"]     = names
    df["location"] = locations
    df["company"]  = companies
    df["title"]    = titles

    return df, exp_dict


if not os.path.exists(csv_path) or not os.path.exists(explanations_path):
    st.error("Output files not found. Please run the ranking pipeline first.")
    st.stop()

df, exp_dict = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
      <span style='font-family:Outfit;font-size:1.6rem;font-weight:800;color:#7DD3FC'>🏆 RedRob</span><br>
      <span style='font-size:0.75rem;color:#64748B;letter-spacing:1px;text-transform:uppercase'>Top 100 Analytics</span>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        options=[
            "📊 Overview & KPIs",
            "📈 Data Analysis",
            "🗺️ Location Insights",
            "💼 Role & Experience",
            "⚙️ Skills Intelligence",
            "🏢 Company Insights",
            "🔍 Candidate Deep-Dive",
            "📋 Full Candidates Table",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#475569;text-align:center'>100 candidates · AI/ML Talent Pipeline</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Overview & KPIs
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Overview & KPIs":
    st.markdown("<h1 style='margin-bottom:4px'>🏆 Top 100 Candidates Overview</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748B;font-size:0.9rem;margin-bottom:28px'>Senior AI Engineer (Ranking & Retrieval) — Talent Pipeline Analysis</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Total Selected", "100", "From 100,000 evaluated"),
        ("Best Score", f"{df['score'].max():.4f}", f"Rank 1 · {df.loc[df['score'].idxmax(),'name']}"),
        ("Avg Score", f"{df['score'].mean():.4f}", "Mean across top 100"),
        ("Avg Experience", f"{df['experience'].mean():.1f} yrs", f"Range: {df['experience'].min():.1f}–{df['experience'].max():.1f}"),
        ("Avg AI Skills", f"{df['skills'].mean():.1f}", "Matched AI core skills"),
    ]
    for col, (label, val, sub) in zip([c1, c2, c3, c4, c5], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    c6, c7, c8, c9, c10 = st.columns(5)
    avg_rr = df["response_rate"].mean()
    locations_unique = df["location"].nunique()
    roles_unique = df["role"].nunique()
    top_loc = df["location"].value_counts().idxmax()
    top_role = df["role"].value_counts().idxmax()

    kpis2 = [
        ("Avg Response Rate", f"{avg_rr*100:.1f}%", "Recruiter responsiveness"),
        ("Unique Locations", str(locations_unique), "Cities represented"),
        ("Distinct Roles", str(roles_unique), "Job title variants"),
        ("Top Location", top_loc.split(",")[0], top_loc),
        ("Dominant Role", (top_role[:22] + "..." if len(top_role) > 22 else top_role),
         f"{df['role'].value_counts().max()} candidates"),
    ]
    for col, (label, val, sub) in zip([c6, c7, c8, c9, c10], kpis2):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="font-size:1.5rem">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sec-header'><h3>🥇 Top 10 Ranked Candidates</h3></div>", unsafe_allow_html=True)
    top10 = df.head(10)[["rank", "name", "title", "location", "company", "experience", "skills", "response_rate", "score"]].copy()
    top10.columns = ["Rank", "Name", "Title", "Location", "Company", "Exp (yrs)", "AI Skills", "Resp. Rate", "Score"]
    top10["Resp. Rate"] = top10["Resp. Rate"].apply(lambda x: f"{x*100:.0f}%")
    top10["Score"] = top10["Score"].apply(lambda x: f"{x:.4f}")
    st.dataframe(top10, use_container_width=True, hide_index=True)

    st.markdown("<br><div class='sec-header'><h3>📉 Score Band Distribution</h3></div>", unsafe_allow_html=True)
    bands = pd.cut(
        df["score"],
        bins=[0.76, 0.78, 0.79, 0.80, 0.82, 0.83],
        labels=["0.76–0.78", "0.78–0.79", "0.79–0.80", "0.80–0.82", "0.82–0.83"],
    )
    band_counts = bands.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    set_dark_figure(fig, ax)
    colors = [BLUE, PURPLE, TEAL, AMBER, GREEN]
    bars = ax.bar(
        band_counts.index.astype(str), band_counts.values,
        color=colors[: len(band_counts)], edgecolor="none", width=0.55,
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(int(bar.get_height())), ha="center", va="bottom",
            color="#CBD5E1", fontweight="bold", fontsize=11,
        )
    ax.set_ylabel("# Candidates", color="#CBD5E1", fontsize=9)
    ax.set_xlabel("Score Band", color="#CBD5E1", fontsize=9)
    ax.set_title("Candidates by Score Band", color="#F1F5F9", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Data Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Data Analysis":
    st.markdown("<h1>📈 Comprehensive Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748B;margin-bottom:24px'>Statistical breakdown of the top 100 selected candidates</p>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='sec-header'><h3>Score Distribution</h3></div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        set_dark_figure(fig, ax)
        n, bins, patches = ax.hist(df["score"], bins=15, color=BLUE, edgecolor="#0B1120", alpha=0.9)
        for i, patch in enumerate(patches):
            patch.set_facecolor(plt.cm.cool(i / len(patches)))
        ax.axvline(df["score"].mean(), color=AMBER, linestyle="--", linewidth=1.5, label=f"Mean: {df['score'].mean():.4f}")
        ax.axvline(df["score"].median(), color=GREEN, linestyle="--", linewidth=1.5, label=f"Median: {df['score'].median():.4f}")
        ax.legend(facecolor=CARD_BG, edgecolor="none", labelcolor="#CBD5E1", fontsize=8)
        ax.set_xlabel("Final Score", color="#CBD5E1", fontsize=9)
        ax.set_ylabel("Count", color="#CBD5E1", fontsize=9)
        ax.set_title("Score Distribution (Top 100)", color="#F1F5F9", fontsize=11, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.markdown("<div class='sec-header'><h3>Experience Distribution</h3></div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        set_dark_figure(fig, ax)
        exp_bins = pd.cut(df["experience"], bins=[3, 5, 6, 7, 8, 9, 10], labels=["3–5", "5–6", "6–7", "7–8", "8–9", "9–10"])
        exp_counts = exp_bins.value_counts().sort_index()
        colors_exp = [TEAL, BLUE, PURPLE, AMBER, GREEN, RED]
        wedges, texts, autotexts = ax.pie(
            exp_counts.values, labels=exp_counts.index, autopct="%1.0f%%",
            colors=colors_exp[: len(exp_counts)], startangle=140,
            pctdistance=0.75, textprops={"color": "#CBD5E1", "fontsize": 8},
        )
        for at in autotexts:
            at.set_color("#0B1120")
            at.set_fontweight("bold")
        ax.set_title("Experience Range (years)", color="#F1F5F9", fontsize=11, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("<div class='sec-header'><h3>Recruiter Response Rate Bands</h3></div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        set_dark_figure(fig, ax)
        rr_bins = pd.cut(
            df["response_rate"],
            bins=[0, 0.3, 0.5, 0.7, 0.9, 1.01],
            labels=["<30%", "30–50%", "50–70%", "70–90%", ">90%"],
        )
        rr_counts = rr_bins.value_counts().sort_index()
        bars = ax.bar(
            rr_counts.index.astype(str), rr_counts.values,
            color=[RED, AMBER, TEAL, BLUE, GREEN][: len(rr_counts)],
            edgecolor="none", width=0.55,
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(int(bar.get_height())), ha="center", va="bottom",
                color="#CBD5E1", fontweight="bold", fontsize=10,
            )
        ax.set_xlabel("Response Rate Band", color="#CBD5E1", fontsize=9)
        ax.set_ylabel("# Candidates", color="#CBD5E1", fontsize=9)
        ax.set_title("Recruiter Response Rate Bands", color="#F1F5F9", fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.15)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_d:
        st.markdown("<div class='sec-header'><h3>AI Core Skills Matched</h3></div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        set_dark_figure(fig, ax)
        skill_counts = df["skills"].value_counts().sort_index()
        bars = ax.bar(
            skill_counts.index.astype(str), skill_counts.values,
            color=PURPLE, edgecolor="none", width=0.55, alpha=0.9,
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(int(bar.get_height())), ha="center", va="bottom",
                color="#CBD5E1", fontweight="bold", fontsize=10,
            )
        ax.set_xlabel("AI Core Skills Count", color="#CBD5E1", fontsize=9)
        ax.set_ylabel("# Candidates", color="#CBD5E1", fontsize=9)
        ax.set_title("Distribution of AI Core Skills Matched", color="#F1F5F9", fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.15)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("<div class='sec-header'><h3>Score vs Experience Scatter</h3></div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    set_dark_figure(fig, ax)
    sc = ax.scatter(
        df["experience"], df["score"], c=df["skills"],
        cmap="cool", s=80, alpha=0.85, edgecolors="none", zorder=3,
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("AI Skills", color="#CBD5E1", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="#CBD5E1")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#CBD5E1")
    ax.set_xlabel("Experience (years)", color="#CBD5E1", fontsize=9)
    ax.set_ylabel("Final Score", color="#CBD5E1", fontsize=9)
    ax.set_title("Score vs Experience (color = AI core skills count)", color="#F1F5F9", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.12)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown("<div class='sec-header'><h3>Score vs Response Rate</h3></div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        set_dark_figure(fig, ax)
        ax.scatter(df["response_rate"] * 100, df["score"], color=TEAL, s=55, alpha=0.75, edgecolors="none")
        z = np.polyfit(df["response_rate"], df["score"], 1)
        p = np.poly1d(z)
        xline = np.linspace(df["response_rate"].min(), df["response_rate"].max(), 100)
        ax.plot(xline * 100, p(xline), color=AMBER, linewidth=1.5, linestyle="--", label="Trend")
        ax.legend(facecolor=CARD_BG, edgecolor="none", labelcolor="#CBD5E1", fontsize=8)
        ax.set_xlabel("Response Rate (%)", color="#CBD5E1", fontsize=9)
        ax.set_ylabel("Final Score", color="#CBD5E1", fontsize=9)
        ax.set_title("Score vs Recruiter Response Rate", color="#F1F5F9", fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_f:
        st.markdown("<div class='sec-header'><h3>Score Rank Trajectory</h3></div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        set_dark_figure(fig, ax)
        ax.plot(df["rank"], df["score"], color=BLUE, linewidth=2, zorder=3)
        ax.fill_between(df["rank"], df["score"], df["score"].min() - 0.002, alpha=0.15, color=BLUE)
        ax.set_xlabel("Rank", color="#CBD5E1", fontsize=9)
        ax.set_ylabel("Score", color="#CBD5E1", fontsize=9)
        ax.set_title("Score Drop-off by Rank", color="#F1F5F9", fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("<br><div class='sec-header'><h3>📐 Summary Statistics</h3></div>", unsafe_allow_html=True)
    stats = df[["score", "experience", "skills", "response_rate"]].describe().T
    stats.columns = ["Count", "Mean", "Std Dev", "Min", "25%", "50% (Median)", "75%", "Max"]
    stats.index = ["Final Score", "Experience (yrs)", "AI Skills Matched", "Response Rate"]
    st.dataframe(stats.style.format("{:.4f}"), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Location Insights
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🗺️ Location Insights":
    st.markdown("<h1>🗺️ Location Insights</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748B;margin-bottom:24px'>Where are the top 100 candidates located?</p>",
        unsafe_allow_html=True,
    )

    loc_counts = df["location"].value_counts()
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("<div class='sec-header'><h3>Candidates by City/Location</h3></div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, max(5, len(loc_counts) * 0.4)))
        set_dark_figure(fig, ax)
        gradient = plt.cm.cool(np.linspace(0.2, 0.9, len(loc_counts)))
        bars = ax.barh(
            loc_counts.index[::-1], loc_counts.values[::-1],
            color=gradient, edgecolor="none", height=0.65,
        )
        for bar in bars:
            ax.text(
                bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())), va="center", ha="left",
                color="#CBD5E1", fontweight="bold", fontsize=9,
            )
        ax.set_xlabel("# Candidates", color="#CBD5E1", fontsize=9)
        ax.set_title("Location Distribution (Top 100 Candidates)", color="#F1F5F9", fontsize=12, fontweight="bold")
        ax.xaxis.grid(True, linestyle="--", alpha=0.12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.markdown("<div class='sec-header'><h3>Top Locations Share</h3></div>", unsafe_allow_html=True)
        top_locs = loc_counts.head(8)
        others = loc_counts.iloc[8:].sum()
        pie_data = pd.concat([top_locs, pd.Series({"Others": others})])
        fig, ax = plt.subplots(figsize=(6, 5))
        set_dark_figure(fig, ax)
        colors_pie = plt.cm.cool(np.linspace(0.1, 0.95, len(pie_data)))
        wedges, texts, autotexts = ax.pie(
            pie_data.values, labels=pie_data.index, autopct="%1.1f%%",
            colors=colors_pie, startangle=130, pctdistance=0.78,
            textprops={"color": "#CBD5E1", "fontsize": 7.5},
        )
        for at in autotexts:
            at.set_color("#0B1120")
            at.set_fontweight("bold")
            at.set_fontsize(8)
        ax.set_title("Location Share (Top 100)", color="#F1F5F9", fontsize=11, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("<div class='sec-header'><h3>Avg Score by Location</h3></div>", unsafe_allow_html=True)
        loc_score = df.groupby("location")["score"].mean().sort_values(ascending=False).head(10)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        set_dark_figure(fig2, ax2)
        bars2 = ax2.barh(
            loc_score.index[::-1], loc_score.values[::-1],
            color=AMBER, edgecolor="none", height=0.55,
        )
        for bar in bars2:
            ax2.text(
                bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}", va="center", ha="left",
                color="#CBD5E1", fontsize=8,
            )
        ax2.set_xlim(0.76, loc_score.max() + 0.01)
        ax2.set_xlabel("Avg Score", color="#CBD5E1", fontsize=9)
        ax2.set_title("Avg Score by Top Locations", color="#F1F5F9", fontsize=11, fontweight="bold")
        ax2.xaxis.grid(True, linestyle="--", alpha=0.12)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("<br><div class='sec-header'><h3>📋 Location Summary Table</h3></div>", unsafe_allow_html=True)
    loc_tbl = (
        df.groupby("location")
        .agg(
            Candidates=("candidate_id", "count"),
            Avg_Score=("score", lambda x: round(x.mean(), 4)),
            Avg_Exp=("experience", lambda x: round(x.mean(), 1)),
            Avg_RR=("response_rate", lambda x: f"{x.mean()*100:.0f}%"),
        )
        .sort_values("Candidates", ascending=False)
        .reset_index()
    )
    loc_tbl.columns = ["Location", "# Candidates", "Avg Score", "Avg Experience", "Avg Response Rate"]
    st.dataframe(loc_tbl, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Role & Experience
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💼 Role & Experience":
    st.markdown("<h1>💼 Role & Experience Analysis</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748B;margin-bottom:24px'>Job title distribution and experience insights across the top 100</p>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='sec-header'><h3>Role Distribution</h3></div>", unsafe_allow_html=True)
        role_counts = df["role"].value_counts()
        fig, ax = plt.subplots(figsize=(7, max(4, len(role_counts) * 0.5)))
        set_dark_figure(fig, ax)
        gradient = plt.cm.plasma(np.linspace(0.2, 0.9, len(role_counts)))
        bars = ax.barh(
            role_counts.index[::-1], role_counts.values[::-1],
            color=gradient, edgecolor="none", height=0.65,
        )
        for bar in bars:
            ax.text(
                bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())), va="center", ha="left",
                color="#CBD5E1", fontweight="bold", fontsize=9,
            )
        ax.set_xlabel("# Candidates", color="#CBD5E1", fontsize=9)
        ax.set_title("Role Breakdown (Top 100)", color="#F1F5F9", fontsize=12, fontweight="bold")
        ax.xaxis.grid(True, linestyle="--", alpha=0.12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.markdown("<div class='sec-header'><h3>Avg Score by Role</h3></div>", unsafe_allow_html=True)
        role_score = df.groupby("role")["score"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, max(4, len(role_score) * 0.5)))
        set_dark_figure(fig, ax)
        gradient2 = plt.cm.cool(np.linspace(0.15, 0.85, len(role_score)))
        bars2 = ax.barh(
            role_score.index[::-1], role_score.values[::-1],
            color=gradient2, edgecolor="none", height=0.65,
        )
        for bar in bars2:
            ax.text(
                bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}", va="center", ha="left",
                color="#CBD5E1", fontsize=8,
            )
        ax.set_xlim(0.76, role_score.max() + 0.01)
        ax.set_xlabel("Avg Score", color="#CBD5E1", fontsize=9)
        ax.set_title("Avg Score per Role", color="#F1F5F9", fontsize=12, fontweight="bold")
        ax.xaxis.grid(True, linestyle="--", alpha=0.12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("<div class='sec-header'><h3>Experience Range per Role (Box Plot)</h3></div>", unsafe_allow_html=True)
    role_order = df.groupby("role")["experience"].median().sort_values(ascending=False).index.tolist()
    role_groups = [df[df["role"] == r]["experience"].values for r in role_order]
    fig, ax = plt.subplots(figsize=(14, 5))
    set_dark_figure(fig, ax)
    bp = ax.boxplot(
        role_groups, patch_artist=True, vert=True, widths=0.55,
        medianprops=dict(color=AMBER, linewidth=2),
        whiskerprops=dict(color="#CBD5E1", linewidth=1),
        capprops=dict(color="#CBD5E1", linewidth=1.5),
        flierprops=dict(markerfacecolor=RED, marker="o", markersize=5, linestyle="none"),
    )
    cmap = plt.cm.cool(np.linspace(0.2, 0.85, len(role_order)))
    for patch, color in zip(bp["boxes"], cmap):
        patch.set_facecolor((*color[:3], 0.55))
        patch.set_edgecolor("#CBD5E1")
    ax.set_xticks(range(1, len(role_order) + 1))
    ax.set_xticklabels(role_order, rotation=30, ha="right", color="#CBD5E1", fontsize=8)
    ax.set_ylabel("Experience (years)", color="#CBD5E1", fontsize=9)
    ax.set_title("Experience Distribution by Role", color="#F1F5F9", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.12)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("<br><div class='sec-header'><h3>Experience Tier Summary</h3></div>", unsafe_allow_html=True)
    df2 = df.copy()
    df2["exp_tier"] = pd.cut(
        df2["experience"], bins=[0, 5, 7, 9, 20],
        labels=["Junior (<=5y)", "Mid (5-7y)", "Senior (7-9y)", "Expert (9y+)"],
    )
    tier_tbl = (
        df2.groupby("exp_tier")
        .agg(
            Count=("candidate_id", "count"),
            Avg_Score=("score", lambda x: round(x.mean(), 4)),
            Avg_Skills=("skills", lambda x: round(x.mean(), 1)),
        )
        .reset_index()
    )
    tier_tbl.columns = ["Experience Tier", "Count", "Avg Score", "Avg AI Skills"]
    st.dataframe(tier_tbl, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — Skills Intelligence
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚙️ Skills Intelligence":
    st.markdown("<h1>⚙️ Skills Intelligence</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748B;margin-bottom:24px'>Required and preferred skill coverage across the top 100 candidates</p>",
        unsafe_allow_html=True,
    )

    req_skills_all, pref_skills_all = [], []
    for cid in df["candidate_id"]:
        item = exp_dict.get(cid, {})
        s = item.get("strengths", {})
        req_skills_all.extend(s.get("matched_required_skills", []))
        pref_skills_all.extend(s.get("matched_preferred_skills", []))

    req_counter  = Counter(req_skills_all)
    pref_counter = Counter(pref_skills_all)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='sec-header'><h3>Most Matched Required Skills</h3></div>", unsafe_allow_html=True)
        req_df = pd.DataFrame(req_counter.most_common(15), columns=["Skill", "Count"])
        fig, ax = plt.subplots(figsize=(7, 5))
        set_dark_figure(fig, ax)
        gradient = plt.cm.Blues(np.linspace(0.4, 0.9, len(req_df)))
        bars = ax.barh(
            req_df["Skill"][::-1], req_df["Count"][::-1],
            color=gradient, edgecolor="none", height=0.65,
        )
        for bar in bars:
            ax.text(
                bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())), va="center", ha="left",
                color="#CBD5E1", fontweight="bold", fontsize=9,
            )
        ax.set_xlabel("Frequency (# candidates)", color="#CBD5E1", fontsize=9)
        ax.set_title("Top Required Skills (Top 100)", color="#F1F5F9", fontsize=11, fontweight="bold")
        ax.xaxis.grid(True, linestyle="--", alpha=0.12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.markdown("<div class='sec-header'><h3>Most Matched Preferred Skills</h3></div>", unsafe_allow_html=True)
        pref_df = pd.DataFrame(pref_counter.most_common(15), columns=["Skill", "Count"])
        fig, ax = plt.subplots(figsize=(7, 5))
        set_dark_figure(fig, ax)
        gradient = plt.cm.Purples(np.linspace(0.4, 0.9, len(pref_df)))
        bars = ax.barh(
            pref_df["Skill"][::-1], pref_df["Count"][::-1],
            color=gradient, edgecolor="none", height=0.65,
        )
        for bar in bars:
            ax.text(
                bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())), va="center", ha="left",
                color="#CBD5E1", fontweight="bold", fontsize=9,
            )
        ax.set_xlabel("Frequency (# candidates)", color="#CBD5E1", fontsize=9)
        ax.set_title("Top Preferred Skills (Top 100)", color="#F1F5F9", fontsize=11, fontweight="bold")
        ax.xaxis.grid(True, linestyle="--", alpha=0.12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("<div class='sec-header'><h3>AI Core Skills Count vs Score</h3></div>", unsafe_allow_html=True)
    col_c, col_d = st.columns(2)

    with col_c:
        skill_score = df.groupby("skills")["score"].agg(["mean", "count"]).reset_index()
        skill_score.columns = ["skills", "avg_score", "count"]
        fig, ax = plt.subplots(figsize=(6, 4))
        set_dark_figure(fig, ax)
        sc = ax.scatter(
            skill_score["skills"], skill_score["avg_score"],
            s=skill_score["count"] * 20, color=TEAL, alpha=0.85,
            edgecolors="#0B1120", linewidth=0.5, zorder=3,
        )
        for _, row in skill_score.iterrows():
            ax.annotate(
                f"n={int(row['count'])}", (row["skills"], row["avg_score"]),
                textcoords="offset points", xytext=(6, 4),
                color="#94A3B8", fontsize=7.5,
            )
        ax.set_xlabel("AI Core Skills Matched", color="#CBD5E1", fontsize=9)
        ax.set_ylabel("Avg Final Score", color="#CBD5E1", fontsize=9)
        ax.set_title("Avg Score by AI Skills Count", color="#F1F5F9", fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_d:
        missing_skills = [
            exp_dict.get(cid, {}).get("weaknesses", {}).get("missing_skills_count", 0)
            for cid in df["candidate_id"]
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        set_dark_figure(fig, ax)
        ax.hist(missing_skills, bins=12, color=RED, edgecolor="#0B1120", alpha=0.85)
        ax.set_xlabel("Missing Skills Count", color="#CBD5E1", fontsize=9)
        ax.set_ylabel("# Candidates", color="#CBD5E1", fontsize=9)
        ax.set_title("Distribution of Missing Skills", color="#F1F5F9", fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("<br><div class='sec-header'><h3>📋 Skills Frequency Tables</h3></div>", unsafe_allow_html=True)
    req_full  = pd.DataFrame(req_counter.most_common(), columns=["Skill", "# Candidates"])
    pref_full = pd.DataFrame(pref_counter.most_common(), columns=["Skill", "# Candidates"])
    tab1, tab2 = st.tabs(["Required Skills", "Preferred Skills"])
    with tab1:
        st.dataframe(req_full, use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(pref_full, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — Company Insights
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🏢 Company Insights":
    st.markdown("<h1>🏢 Company Insights</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748B;margin-bottom:24px'>Current companies of the top 100 candidates</p>",
        unsafe_allow_html=True,
    )

    comp_counts = df["company"].value_counts()
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("<div class='sec-header'><h3>Candidates by Current Company</h3></div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, max(5, len(comp_counts) * 0.45)))
        set_dark_figure(fig, ax)
        gradient = plt.cm.viridis(np.linspace(0.2, 0.85, len(comp_counts)))
        bars = ax.barh(
            comp_counts.index[::-1], comp_counts.values[::-1],
            color=gradient, edgecolor="none", height=0.65,
        )
        for bar in bars:
            ax.text(
                bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())), va="center", ha="left",
                color="#CBD5E1", fontweight="bold", fontsize=9,
            )
        ax.set_xlabel("# Candidates", color="#CBD5E1", fontsize=9)
        ax.set_title("Current Company Distribution", color="#F1F5F9", fontsize=12, fontweight="bold")
        ax.xaxis.grid(True, linestyle="--", alpha=0.12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.markdown("<div class='sec-header'><h3>Top Companies Share</h3></div>", unsafe_allow_html=True)
        top_comps = comp_counts.head(8)
        others_c = comp_counts.iloc[8:].sum()
        pie_c = pd.concat([top_comps, pd.Series({"Others": others_c})])
        fig, ax = plt.subplots(figsize=(6, 5))
        set_dark_figure(fig, ax)
        colors_c = plt.cm.viridis(np.linspace(0.1, 0.9, len(pie_c)))
        wedges, texts, autotexts = ax.pie(
            pie_c.values, labels=pie_c.index, autopct="%1.1f%%",
            colors=colors_c, startangle=130, pctdistance=0.78,
            textprops={"color": "#CBD5E1", "fontsize": 7.5},
        )
        for at in autotexts:
            at.set_color("#0B1120")
            at.set_fontweight("bold")
            at.set_fontsize(8)
        ax.set_title("Company Share (Top 100)", color="#F1F5F9", fontsize=11, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("<div class='sec-header'><h3>Avg Score by Company (Top 10)</h3></div>", unsafe_allow_html=True)
    comp_score = df.groupby("company")["score"].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 4))
    set_dark_figure(fig, ax)
    bars = ax.bar(comp_score.index, comp_score.values, color=BLUE, edgecolor="none", width=0.6, alpha=0.9)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
            f"{bar.get_height():.4f}", ha="center", va="bottom",
            color="#CBD5E1", fontsize=8, fontweight="bold",
        )
    ax.set_xlabel("Company", color="#CBD5E1", fontsize=9)
    ax.set_ylabel("Avg Score", color="#CBD5E1", fontsize=9)
    ax.set_ylim(0.76, comp_score.max() + 0.01)
    ax.set_title("Average Score by Company", color="#F1F5F9", fontsize=12, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.12)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("<br><div class='sec-header'><h3>📋 Company Summary Table</h3></div>", unsafe_allow_html=True)
    comp_tbl = (
        df.groupby("company")
        .agg(
            Candidates=("candidate_id", "count"),
            Avg_Score=("score", lambda x: round(x.mean(), 4)),
            Avg_Exp=("experience", lambda x: round(x.mean(), 1)),
        )
        .sort_values("Candidates", ascending=False)
        .reset_index()
    )
    comp_tbl.columns = ["Company", "# Candidates", "Avg Score", "Avg Exp (yrs)"]
    st.dataframe(comp_tbl, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 — Candidate Deep-Dive
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Candidate Deep-Dive":
    st.markdown("<h1>🔍 Candidate Deep-Dive</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748B;margin-bottom:20px'>Explore the full profile and scoring breakdown of each top candidate</p>",
        unsafe_allow_html=True,
    )

    options = [f"#{row['rank']} — {row['name']} ({row['candidate_id']})" for _, row in df.iterrows()]
    selected_opt = st.selectbox("Select a candidate:", options)
    selected_cid = selected_opt.split("(")[1].rstrip(")")

    if selected_cid in exp_dict:
        item       = exp_dict[selected_cid]
        ph         = item.get("profile_highlights", {})
        breakdown  = item.get("scoring_breakdown", {})
        strengths  = item.get("strengths", {})
        weaknesses = item.get("weaknesses", {})
        row        = df[df["candidate_id"] == selected_cid].iloc[0]

        st.markdown(f"""
        <div class="cand-card">
          <div style="display:flex; align-items:center; gap:16px; margin-bottom:8px">
            <div class="rank-badge">#{int(row['rank'])}</div>
            <div>
              <div style="font-family:Outfit;font-size:1.5rem;font-weight:700;color:#F1F5F9">{ph.get('name','—')}</div>
              <div style="color:#7DD3FC;font-size:0.9rem">{ph.get('headline','—')}</div>
            </div>
            <div style="margin-left:auto;text-align:right">
              <div style="font-family:Outfit;font-size:2rem;font-weight:800;color:#4ADE80">{item['final_score']:.4f}</div>
              <div style="font-size:0.75rem;color:#64748B">Final Score</div>
            </div>
          </div>
          <hr style="border-color:rgba(255,255,255,0.05)">
          <div style="display:flex;gap:32px;flex-wrap:wrap;font-size:0.85rem;color:#94A3B8">
            <span>📍 {ph.get('location','N/A')}</span>
            <span>🏢 {ph.get('current_title','N/A')} @ {ph.get('current_company','N/A')}</span>
            <span>📅 {strengths.get('experience_years',0)} yrs experience</span>
            <span>📬 {strengths.get('recruiter_response_rate',0)*100:.0f}% response rate</span>
            <span>📡 Active {strengths.get('days_since_active',0)} days ago</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.markdown("<div class='sec-header'><h3>Profile Summary</h3></div>", unsafe_allow_html=True)
            summary_txt = ph.get("summary", "No summary available.")
            st.markdown(
                f"<div style='font-size:0.85rem;color:#94A3B8;line-height:1.7'>{summary_txt[:600]}{'...' if len(summary_txt) > 600 else ''}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("<br><div class='sec-header'><h3>Required Skills Matched</h3></div>", unsafe_allow_html=True)
            req_skills = strengths.get("matched_required_skills", [])
            if req_skills:
                pills_html = "".join(f'<span class="skill-pill">{s}</span>' for s in req_skills)
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#64748B'>None</span>", unsafe_allow_html=True)

            st.markdown("<br><div class='sec-header'><h3>Preferred Skills Matched</h3></div>", unsafe_allow_html=True)
            pref_skills = strengths.get("matched_preferred_skills", [])
            if pref_skills:
                pills_html = "".join(f'<span class="skill-pill skill-pill-pref">{s}</span>' for s in pref_skills)
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#64748B'>None</span>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if weaknesses.get("consulting_history"):
                st.warning("Consulting Career Flag: Candidate has history at an IT consulting firm.")
            else:
                st.success("No consulting career flag detected.")

            miss = weaknesses.get("missing_skills_count", 0)
            st.info(f"Missing Skills Count: {miss} skills not matched from JD requirements")

        with col_r:
            st.markdown("<div class='sec-header'><h3>Hybrid Score Breakdown</h3></div>", unsafe_allow_html=True)
            score_labels = [
                "Semantic Fit (40%)",
                "Career Match (20%)",
                "Skill Alignment (15%)",
                "Behavioral Score (10%)",
                "Quality Check (10%)",
                "Product Company (5%)",
            ]
            score_keys = [
                "semantic_score_40pct",
                "career_score_20pct",
                "skill_score_15pct",
                "behavior_score_10pct",
                "quality_score_10pct",
                "product_company_score_5pct",
            ]
            score_vals = [breakdown.get(k, 0.0) for k in score_keys]

            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            set_dark_figure(fig, ax)
            colors_s = [BLUE, PURPLE, TEAL, GREEN, AMBER, RED]
            bars = ax.barh(
                score_labels[::-1], score_vals[::-1],
                color=colors_s[::-1], edgecolor="none", height=0.55,
            )
            for bar in bars:
                ax.text(
                    min(bar.get_width() + 0.02, 1.02), bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.3f}", va="center", ha="left",
                    color="#CBD5E1", fontweight="bold", fontsize=9,
                )
            ax.set_xlim(0, 1.1)
            ax.set_xlabel("Score (0.0 - 1.0)", color="#CBD5E1", fontsize=9)
            ax.set_title("Component Score Breakdown", color="#F1F5F9", fontsize=11, fontweight="bold")
            ax.xaxis.grid(True, linestyle="--", alpha=0.12)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown(f"""
            <div style="background:rgba(30,41,59,0.5);border-radius:10px;padding:14px;margin-top:10px;font-size:0.85rem">
              <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span style="color:#94A3B8">Raw Hybrid Score</span>
                <span style="color:#F1F5F9;font-weight:600">{item['raw_hybrid_score']:.4f}</span>
              </div>
              <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span style="color:#94A3B8">Multiplier Applied</span>
                <span style="color:#F1F5F9;font-weight:600">x{item['multiplier']:.2f} ({item['category']})</span>
              </div>
              <div style="display:flex;justify-content:space-between;border-top:1px solid rgba(255,255,255,0.08);padding-top:8px;margin-top:4px">
                <span style="color:#94A3B8">Final Ranking Score</span>
                <span style="color:#4ADE80;font-weight:800;font-size:1.1rem">{item['final_score']:.4f}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Radar chart
            st.markdown("<br>", unsafe_allow_html=True)
            categories = ["Semantic\nFit", "Career\nMatch", "Skill\nAlign", "Behavioral", "Quality", "Product\nCo."]
            vals_radar  = score_vals + [score_vals[0]]
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            fig_r, ax_r = plt.subplots(figsize=(5, 4), subplot_kw=dict(polar=True))
            fig_r.patch.set_facecolor(DARK_BG)
            ax_r.set_facecolor(CARD_BG)
            ax_r.plot(angles, vals_radar, color=BLUE, linewidth=2)
            ax_r.fill(angles, vals_radar, color=BLUE, alpha=0.25)
            ax_r.set_xticks(angles[:-1])
            ax_r.set_xticklabels(categories, color="#CBD5E1", fontsize=7.5)
            ax_r.set_ylim(0, 1)
            ax_r.tick_params(colors="#CBD5E1")
            ax_r.grid(color=(1.0, 1.0, 1.0, 0.08), linestyle="--")  # matplotlib RGBA tuple
            ax_r.set_title("Score Radar", color="#F1F5F9", fontsize=11, fontweight="bold", pad=15)
            fig_r.tight_layout()
            st.pyplot(fig_r)
            plt.close(fig_r)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 8 — Full Candidates Table
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Full Candidates Table":
    st.markdown("<h1>📋 All 100 Candidates — Full Table</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748B;margin-bottom:20px'>Browse, filter, and export the complete top 100 candidate list</p>",
        unsafe_allow_html=True,
    )

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        min_score = st.slider(
            "Minimum Score",
            float(df["score"].min()), float(df["score"].max()),
            float(df["score"].min()), step=0.001, format="%.4f",
        )
    with col_f2:
        roles_list = ["All"] + sorted(df["role"].unique().tolist())
        selected_role = st.selectbox("Filter by Role", roles_list)
    with col_f3:
        locs_list = ["All"] + sorted(df["location"].unique().tolist())
        selected_loc = st.selectbox("Filter by Location", locs_list)

    fdf = df[df["score"] >= min_score].copy()
    if selected_role != "All":
        fdf = fdf[fdf["role"] == selected_role]
    if selected_loc != "All":
        fdf = fdf[fdf["location"] == selected_loc]

    st.markdown(
        f"<p style='color:#94A3B8;font-size:0.85rem'>Showing <b>{len(fdf)}</b> candidates</p>",
        unsafe_allow_html=True,
    )

    display_cols = ["rank", "candidate_id", "name", "title", "location", "company", "experience", "skills", "response_rate", "score"]
    display_df = fdf[display_cols].copy()
    display_df.columns = ["Rank", "Candidate ID", "Name", "Title", "Location", "Company", "Exp (yrs)", "AI Skills", "Resp. Rate", "Score"]
    display_df["Score"] = display_df["Score"].apply(lambda x: f"{x:.4f}")
    display_df["Resp. Rate"] = display_df["Resp. Rate"].apply(lambda x: f"{x*100:.0f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        with open(csv_path, "rb") as f:
            st.download_button(
                "Download top_100_candidates.csv", f,
                file_name="top_100_candidates.csv", mime="text/csv",
            )
    with col_dl2:
        with open(explanations_path, "rb") as f:
            st.download_button(
                "Download top_100_explanations.json", f,
                file_name="top_100_explanations.json", mime="application/json",
            )
