import re
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st

from features import extract_url_features

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;600;700&family=Roboto+Mono:wght@400;500&display=swap');

:root {
    --bg: #05090f;
    --panel: rgba(9, 20, 32, 0.94);
    --line: #123344;
    --cyan: #00f5d4;
    --blue: #00a6fb;
    --text: #d9f7ff;
    --muted: #7695a5;
}

.stApp {
    color: var(--text);
    background-color: var(--bg);
    background-image:
        linear-gradient(rgba(0, 245, 212, .035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 245, 212, .035) 1px, transparent 1px),
        radial-gradient(ellipse at 50% -20%, #103349 0%, transparent 55%);
    background-size: 32px 32px, 32px 32px, auto;
}

.block-container {
    max-width: 1220px;
    padding-top: 5rem;
}

.hero {
    margin-top: 1rem;
}

h1, h2, h3 {
    color: var(--text) !important;
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: .04em;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid var(--cyan);
    border-radius: 4px;
    background: linear-gradient(110deg, rgba(0,245,212,.10), rgba(0,166,251,.04) 55%, transparent);
    box-shadow: 0 0 22px rgba(0,245,212,.12), inset 0 0 24px rgba(0,245,212,.035);
}

.hero::after {
    content: "";
    position: absolute;
    top: 0;
    left: -100%;
    width: 60%;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    animation: scan 5s linear infinite;
}

@keyframes scan {
    to { left: 150%; }
}

.hero h1 {
    margin: 0;
    color: var(--cyan) !important;
    font-size: 2.1rem;
    text-shadow: 0 0 16px rgba(0,245,212,.45);
}

.hero p {
    margin: .45rem 0 0;
    color: #8fb5c4;
    font-family: 'Roboto Mono', monospace;
}

.card {
    min-height: 110px;
    padding: 1rem 1.2rem;
    border: 1px solid var(--line);
    border-left: 3px solid var(--cyan);
    border-radius: 4px;
    background: var(--panel);
    box-shadow: 0 0 16px rgba(0,166,251,.07);
}

.card-label {
    color: var(--muted);
    font-family: 'Roboto Mono', monospace;
    font-size: .78rem;
    text-transform: uppercase;
    letter-spacing: .12em;
}

.card-value {
    padding-top: .45rem;
    color: var(--cyan);
    font-family: 'Orbitron', sans-serif;
    font-size: 1.55rem;
    text-shadow: 0 0 10px rgba(0,245,212,.25);
}

[data-testid="stSidebar"] {
    background: #060d15;
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

[data-testid="stTextInput"] input {
    color: var(--cyan) !important;
    background: #050c13 !important;
    border: 1px solid #1d5365 !important;
    border-radius: 3px !important;
    font-family: 'Roboto Mono', monospace;
}

[data-testid="stTextInput"] input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 12px rgba(0,245,212,.22) !important;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    min-height: 2.8rem;
    border: 1px solid var(--cyan);
    border-radius: 3px;
    color: #001411;
    background: var(--cyan);
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    letter-spacing: .08em;
    box-shadow: 0 0 15px rgba(0,245,212,.25);
    transition: all .18s ease;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    color: #001411;
    border-color: #fff;
    background: #8ffff0;
    box-shadow: 0 0 24px rgba(0,245,212,.55);
    transform: translateY(-1px);
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
}

[data-testid="stAlert"] {
    border-radius: 3px;
}

hr {
    border-color: var(--line);
}

code {
    color: var(--cyan);
    background: #07131c;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/model.joblib")
    metadata = joblib.load("artifacts/metadata.joblib")
    return model, metadata


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


try:
    model, metadata = load_artifacts()
except FileNotFoundError:
    st.error("Model files are missing. Run `python train.py` first.")
    st.stop()


st.markdown("""
<div class="hero">
    <h1>🛡️ PhishGuard AI</h1>
    <p>Machine-learning URL screening with transparent feature analysis</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Research demo only. This scanner does not visit websites or check them live. "
    "A low-risk result does not prove a URL is safe."
)

page = st.sidebar.radio(
    "Workspace",
    ["URL Scanner", "Model Performance", "About"],
)

if page == "URL Scanner":
    st.subheader("Analyze a URL")
    st.write("Paste a URL to inspect its structure and get a model prediction.")

    with st.form("url_scan_form"):
        url = st.text_input(
            "Website URL",
            placeholder="https://example.com",
        )
        submitted = st.form_submit_button("🔎  Scan URL", use_container_width=True)

    if submitted:
        try:
            feature_dict = extract_url_features(url)
            feature_names = metadata["feature_names"]

            row = pd.DataFrame(
                [[feature_dict[name] for name in feature_names]],
                columns=feature_names,
            )

            prediction = int(model.predict(row)[0])
            probabilities = model.predict_proba(row)[0]
            classes = list(model.classes_)
            phishing_probability = float(probabilities[classes.index(1)])

            left, right = st.columns([1.2, 1])

            with left:
                st.markdown("### Result")
                if prediction == 1:
                    st.error("⚠️ Likely phishing")
                else:
                    st.success("✅ Likely legitimate")

                st.progress(phishing_probability)
                st.caption(
                    f"Model phishing score: {phishing_probability:.1%} "
                    "(not a guarantee)"
                )

            parsed = urlparse(url if "://" in url else "https://" + url)
            hostname = parsed.hostname or ""
            is_ip = bool(
                re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", hostname)
            )

            with right:
                st.markdown("### Quick checks")
                checks = [
                    ("HTTPS present", parsed.scheme.lower() == "https"),
                    ("IP address used as hostname", is_ip),
                    ("`@` character in URL", "@" in url),
                    ("Long URL (>75 characters)", len(url) > 75),
                ]
                for label, flagged in checks:
                    st.write(("⚠️ " if flagged else "✓ ") + label)

            st.markdown("### URL feature summary")
            importance = getattr(model, "feature_importances_", None)

            if importance is not None:
                ranked = sorted(
                    zip(feature_names, importance),
                    key=lambda item: item[1],
                    reverse=True,
                )[:8]

                explanation = pd.DataFrame([
                    {
                        "Feature": name.replace("_", " ").title(),
                        "Value for this URL": feature_dict[name],
                        "Global importance": float(weight),
                    }
                    for name, weight in ranked
                ])
                st.dataframe(
                    explanation,
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(
                    "Global importance describes the model overall; it is not "
                    "a guarantee that each listed feature caused this prediction."
                )

        except ValueError as error:
            st.error(str(error))

elif page == "Model Performance":
    st.subheader("Model comparison")
    st.write("Evaluation results saved during the most recent training run.")

    metrics = metadata.get("metrics", [])
    if metrics:
        table_rows = []
        for item in metrics:
            table_rows.append({
                "Model": item["model"],
                "Accuracy": f'{item["accuracy"]:.3f}',
                "Precision": f'{item["precision"]:.3f}',
                "Recall": f'{item["recall"]:.3f}',
                "F1-score": f'{item["f1"]:.3f}',
            })

        st.dataframe(
            pd.DataFrame(table_rows),
            hide_index=True,
            use_container_width=True,
        )

        selected = st.selectbox(
            "View confusion matrix",
            [item["model"] for item in metrics],
        )
        selected_result = next(
            item for item in metrics if item["model"] == selected
        )

        matrix = pd.DataFrame(
            selected_result["confusion_matrix"],
            index=["Actual legitimate", "Actual phishing"],
            columns=["Predicted legitimate", "Predicted phishing"],
        )
        st.markdown("#### Confusion matrix")
        st.dataframe(matrix, use_container_width=True)

    st.info(
        "Your evaluation used a domain-grouped train/test split. "
        "Report the dataset and split method alongside these scores."
    )

elif page == "About":
    st.subheader("About this project")
    st.write(
        "PhishGuard AI extracts structural features from a URL and passes "
        "them to a trained Random Forest classifier."
    )
    st.markdown("""
    **Label mapping**
    - `0` = legitimate
    - `1` = phishing

    **Limitations**
    - The app analyzes the URL text only; it does not inspect the website.
    - HTTPS does not mean a site is legitimate.
    - Model scores describe performance on the evaluation dataset and do not
      guarantee future detection accuracy.
    """)