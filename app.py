# -*- coding: utf-8 -*-
"""Web calculator: persistent organ failure / in-hospital mortality in acute pancreatitis.

Run locally:   streamlit run app.py
Deploy free:   push this folder to GitHub, then share.streamlit.io -> New app.
"""
import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AP Prognosis Calculator", page_icon="🩺", layout="centered")

@st.cache_resource
def load_bundle():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

b = load_bundle()

st.title("Prognosis in Acute Pancreatitis")
st.caption("Persistent organ failure or in-hospital mortality — 4-variable model "
           "developed in MIMIC-IV (n=777) and externally validated in a Chinese cohort (n=384) "
           "and the US multi-center eICU-CRD (n=575)")

with st.sidebar:
    st.header("Model settings")
    recal = st.checkbox("Apply intercept recalibration (−0.400) — recommended outside the US",
                        value=True)
    use_mpv = st.checkbox("Use MPV-augmented model (requires MPV)", value=False)
    st.markdown("---")
    st.markdown("**Risk thresholds**")
    st.markdown(f"- Base model (Youden): `{b['base']['threshold']:.3f}`")
    st.markdown(f"- MPV model (Youden): `{b['mpv']['threshold']:.3f}`")

st.subheader("Admission laboratory values (within 24 h of ICU admission)")
c1, c2 = st.columns(2)
with c1:
    rdw = st.number_input("RDW-CV (%)", min_value=8.0, max_value=40.0, value=14.0, step=0.1)
    bili = st.number_input("Total bilirubin (mg/dL)", min_value=0.1, max_value=40.0, value=1.0, step=0.1)
with c2:
    cr = st.number_input("Creatinine (mg/dL)", min_value=0.1, max_value=15.0, value=1.0, step=0.1)
    bun = st.number_input("BUN (mg/dL)", min_value=1.0, max_value=200.0, value=15.0, step=1.0)
mpv = None
if use_mpv:
    mpv = st.number_input("Mean platelet volume, MPV (fL)", min_value=5.0, max_value=20.0,
                          value=10.5, step=0.1)

def predict_base(rdw, bili, cr, bun, recalibrate):
    x = pd.DataFrame([[rdw, bili, cr, bun]], columns=b['features'])
    z = b['base']['scaler'].transform(x)
    lp = float(np.asarray(b['base']['model'].decision_function(z)).ravel()[0])
    if recalibrate:
        lp = lp + float(b['base']['recal_shift'])
    return float(1 / (1 + np.exp(-lp)))

if st.button("Calculate risk", type="primary"):
    p = float(predict_base(rdw, bili, cr, bun, recal))
    cut = b['base']['threshold']
    st.subheader("Result — base 4-variable model")
    st.metric("Predicted risk of POF or in-hospital death", f"{p:.1%}")
    if p >= cut:
        st.error(f"High risk (≥ {cut:.1%}). In the external Chinese cohort, the observed "
                 f"event rate in this group was 44.9% (vs 9.3% below threshold).")
    else:
        st.success(f"Low risk (< {cut:.1%}). Observed event rate in this group was 9.3% "
                   f"in the external cohort.")

    if use_mpv and mpv is not None:
        x = pd.DataFrame([[rdw, bili, cr, bun, mpv]],
                         columns=['rdw_cv', 'bilirubin', 'creatinine', 'bun', 'mpv'])
        z = b['mpv']['scaler'].transform(x)
        p2 = float(b['mpv']['model'].predict_proba(z)[0, 1])
        cut2 = b['mpv']['threshold']
        st.subheader("Result — MPV-augmented model (local cohort)")
        st.metric("Predicted risk (with MPV)", f"{p2:.1%}")
        if p2 >= cut2:
            st.error(f"High risk (≥ {cut2:.1%}).")
        else:
            st.success(f"Low risk (< {cut2:.1%}).")

st.markdown("---")
st.subheader("About the model")
st.markdown(
    "- **Outcome:** composite of persistent organ failure (modified Marshall score ≥2 in "
    "the respiratory, cardiovascular, or renal system for >48 h, per the revised Atlanta "
    "classification) or in-hospital death.\n"
    "- **Equation:** logit(p) = −0.245 + 0.280·z(RDW-CV) + 0.171·z(bilirubin) "
    "+ 0.535·z(creatinine) + 0.234·z(BUN), z standardized to the full MIMIC-IV cohort (n=777). "
    "The deployment model in this app was re-estimated on the full MIMIC-IV cohort "
    "with identical preprocessing.\n"
    "- **Performance:** internal validation AUC 0.719 (95% CI 0.652–0.781); "
    "Chinese external cohort AUC 0.833 (95% CI 0.782–0.877); "
    "eICU-CRD external validation AUC 0.774 (95% CI 0.719–0.825). "
    "Intercept recalibration (−0.400 in the Chinese cohort) is recommended when "
    "transporting the model to cohorts with lower baseline risk.\n"
    "- **MPV-augmented model** was trained on the Chinese cohort (n=384); "
    "out-of-fold AUC 0.912 vs 0.866 for the base model. The incremental value of MPV "
    "did not replicate in eICU-CRD (ΔAUC +0.011); interpret MPV results with caution.\n"
    "- **Intended use:** research and risk stratification support only; "
    "not a substitute for clinical judgment. Verify local calibration before clinical use.")
