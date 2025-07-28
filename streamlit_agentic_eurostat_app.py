# app.py

import os
import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import streamlit as st
import altair as alt

# ─── API Key Configuration ─────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY is not set; AI insights will be disabled.")

# ─── HTTP-based AI call (bypass SDK) ───────────────────────────────────────────
def call_ai_api(messages):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {"model": "gpt-4o", "messages": messages, "temperature": 0.8}
    try:
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"AI API call failed: {e}")
        return ""

# ─── Streamlit Page Config ─────────────────────────────────────────────────────
st.set_page_config(page_title="Agentic EuroStat Economist", layout="wide")
st.title("🌐 Agentic EuroStat Economist")

# Session memory
if 'ai_history' not in st.session_state:
    st.session_state.ai_history = []

# ─── Sidebar Inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data & Model Settings")
    geo_input = st.text_input("Country codes (comma-separated)", "IE", key='geo_input')
    geos = [g.strip() for g in geo_input.split(',') if g.strip()]

    dset_input = st.text_input("Eurostat Dataset code", "nama_10_gdp", key='dset_input')
    dsets = [dset_input.strip()]

    freq = st.selectbox("Rolling average window (periods)", [0, 3, 5], key='freq')
    model_type = st.selectbox("Econometric model", ["OLS", "VAR", "Monte Carlo"], key='model_type')
    edit_prompt = st.text_area("Custom AI prompt (optional)", height=100, key='edit_prompt')
    run_it = st.button("Run Analysis", key='run_it')

# ─── Helper: parse period labels ────────────────────────────────────────────────
def parse_period(label: str) -> pd.Timestamp:
    for fmt in ["%Y-%m-%d", "%Y-%m", "%Y"]:
        try:
            return pd.to_datetime(label, format=fmt)
        except:
            continue
    try:
        return pd.Period(label).to_timestamp()
    except:
        return None

# ─── Data Fetcher ──────────────────────────────────────────────────────────────
def fetch_eurostat(code: str, geo: str) -> pd.DataFrame:
    url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{code}"
    try:
        r = requests.get(url, params={"geo": geo, "format": "JSON"})
        r.raise_for_status()
        js = r.json()
    except:
        return pd.DataFrame(columns=["value"])
    idx = js.get("dimension", {}).get("time", {}).get("category", {}).get("index", {})
    vals = js.get("value", {})
    rows = []
    for label, cat in idx.items():
        ts = parse_period(label)
        if ts is None:
            continue
        v = vals.get(str(cat))
        rows.append((ts, v))
    df = pd.DataFrame(rows, columns=["date", "value"]).dropna()
    return df.set_index("date").sort_index()

# ─── Scenario Prompt Generator ─────────────────────────────────────────────────
def make_scenarios_description(name: str, vals: list) -> str:
    return (
        f"Dataset '{name}' recent values: {vals}. "
        "Generate two forward-looking 5-period scenarios: adverse downturn and favourable recovery. "
        "Explain key drivers and policy implications."
    )

# ─── Main Flow ────────────────────────────────────────────────────────────────
if run_it:
    try:
        # Fetch and merge time series
        frames = {}
        for ds in dsets:
            for geo in geos:
                key = f"Eurostat_{ds}_{geo}"
                df = fetch_eurostat(ds, geo)
                if freq > 0 and not df.empty:
                    df['value'] = df['value'].rolling(freq).mean()
                frames[key] = df.rename(columns={'value': key})
        if not frames:
            st.warning("No data fetched; check dataset and country codes.")
            st.stop()
        merged = pd.concat(frames.values(), axis=1)

        # Line Chart with tooltip and dynamic crosshair
        st.subheader("Historical Series")
        chart_data = merged.reset_index().melt('date', var_name='series', value_name='value')
        nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['date'], empty='none')
        line = alt.Chart(chart_data).mark_line().encode(
            x='date:T',
            y='value:Q',
            color='series:N'
        )
        selectors = alt.Chart(chart_data).mark_point().encode(
            x='date:T',
            opacity=alt.value(0),
        ).add_selection(nearest)
        points = line.mark_circle(size=100).encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0))
        )
        vline = alt.Chart(chart_data).mark_rule(color='gray').encode(
            x='date:T',
            opacity=alt.condition(nearest, alt.value(0.5), alt.value(0))
        )
        hline = alt.Chart(chart_data).mark_rule(color='gray').encode(
            y='value:Q',
            opacity=alt.condition(nearest, alt.value(0.5), alt.value(0))
        )
        tooltip = alt.Chart(chart_data).mark_text(align='left', dx=5, dy=-5).encode(
            x='date:T',
            y='value:Q',
            text=alt.condition(nearest, 'value:Q', alt.value(' '))
        )
        combined = alt.layer(line, selectors, points, vline, hline, tooltip).interactive()
        st.altair_chart(combined, use_container_width=True)

        # Distribution Visualization
        st.subheader("Data Distribution")
        # Dropdown: show only dataset and geo (strip prefix) but map back to full column name
        full_cols = list(merged.columns)
        display_labels = [col.replace('Eurostat_', '') for col in full_cols]
        label_to_col = dict(zip(display_labels, full_cols))
        selected_label = st.selectbox("Select series to view distribution", display_labels)
        selected_col = label_to_col[selected_label]
        df_dist = merged[selected_col].reset_index().rename(columns={selected_col: 'value'})
        # Traditional histogram
        hist = alt.Chart(df_dist).mark_bar().encode(
            x=alt.X('value:Q', bin=alt.Bin(maxbins=30), title='Value'),
            y=alt.Y('count()', title='Frequency'),
            tooltip=[alt.Tooltip('count()', title='Count'), alt.Tooltip('mean(value):Q', title='Average')]
        ).properties(width=600, height=300).interactive()
        st.altair_chart(hist, use_container_width=True)

        # Econometric Modeling

        # Econometric Modeling
        st.subheader(f"Econometric Model: {model_type}")
        data = merged.dropna()
        if data.empty:
            st.warning("No data for modeling.")
            st.stop()
        if model_type == 'OLS':
            s = data.iloc[:, 0]
            yrs = pd.DatetimeIndex(s.index).year
            res = sm.OLS(s.values, sm.add_constant(yrs)).fit()
            st.text(res.summary())
        elif model_type == 'VAR' and data.shape[1] >= 2:
            irf = VAR(data).fit(maxlags=2).irf(10)
            st.pyplot(irf.plot(orth=False))
        else:
            s = data.iloc[:, 0]
            yrs = pd.DatetimeIndex(s.index).year
            lm = sm.OLS(s.values, sm.add_constant(yrs)).fit()
            resid, last = lm.resid, yrs[-1]
            sims = {}
            for i in range(100):
                cur = s.values[-1]
                vals_list = []
                for t in range(1, 6):
                    cur = lm.params[0] + lm.params[1] * (last + t) + np.random.choice(resid)
                    vals_list.append(cur)
                sims[f"sim_{i}"] = vals_list
            df_sim = pd.DataFrame(
                sims,
                index=pd.date_range(start=pd.Timestamp(f"{last+1}-01-01"), periods=5, freq='Y')
            )
            st.line_chart(df_sim)

        # Agentic AI Insights
        st.subheader("Agentic AI Insights")
        base = merged.columns[0]
        rec = merged[base].dropna().iloc[-5:].tolist()
        prompt_text = edit_prompt.strip() or make_scenarios_description(base, rec)
        msgs = [{"role": "system", "content": f"You are an economist analyzing '{base}'."},
                {"role": "user", "content": prompt_text}]
        ai_out = call_ai_api(msgs)
        if ai_out:
            st.markdown(ai_out)
            st.session_state.ai_history.append({'series': base, 'prompt': prompt_text, 'output': ai_out})
    except Exception as e:
        st.exception(e)
