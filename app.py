from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from shiny import App, reactive, render, ui


# ----------------------------
# Paths / constants
# ----------------------------
HERE = Path(__file__).resolve().parent

MODEL_PATH = HERE / "tabpfn_model.joblib"
MODEL_COLS_PATH = HERE / "model_cols.json"
PLATT_PATH = HERE / "platt_params.json"  # expects {"alpha_ens": float, "beta_ens": float}

UI_WIDTH = "820px"
DESC_WIDTH = "320px"
MISSING_TOKEN = "(missing)"


# ----------------------------
# Inputs / defaults
# ----------------------------
NUM_DEFAULTS = {
    "bmi": 0.0,
    "tumor_distance_from_arj": 0.0,
    "tumor_cradiocaudal_length": 0.0,
    "pretreatment_cea": 0.0,
    "number_of_cycles": 0.0,
}

NUM_FIELDS = [
    "bmi",
    "tumor_distance_from_arj",
    "tumor_cradiocaudal_length",
    "pretreatment_cea",
    "number_of_cycles",
]

# ONE-HOT categoricals in model_cols (MR_TRG is numeric-only)
CATEGORICAL_KEYS = [
    "gender",
    "emvi",
    "extramesorectal_ln",
    "preop_hiso_type",
    "preop_tumor_grading",
    "clinical_t_stage",
    "clinical_n_stage",
    "mrf_involvement",
    "kras",
    "braf_mutation",
    "tnt_regimen",
    "tnt_course",
    "asa",
]

# UI select inputs includes MR_TRG
UI_SELECT_KEYS = CATEGORICAL_KEYS + ["MR_TRG"]


# ----------------------------
# Labels / choices
# ----------------------------
LABELS = {
    "bmi": "BMI",
    "tumor_distance_from_arj": "Tumor distance from ARJ (cm)",
    "tumor_cradiocaudal_length": "Tumor craniocaudal length (cm)",
    "pretreatment_cea": "Pre-treatment CEA",
    "number_of_cycles": "Number of cycles of chemotherapy completed",
    "gender": "Gender",
    "emvi": "EMVI",
    "extramesorectal_ln": "Extramesorectal LN",
    "preop_hiso_type": "Preop histology type",
    "preop_tumor_grading": "Preop tumor grading",
    "clinical_t_stage": "Clinical T stage",
    "clinical_n_stage": "Clinical N stage",
    "mrf_involvement": "MRF involvement",
    "kras": "KRAS",
    "braf_mutation": "BRAF mutation",
    "tnt_regimen": "TNT regimen",
    "tnt_course": "TNT course",
    "asa": "ASA",
    "MR_TRG": "MR TRG",
}


def label(k: str) -> str:
    return LABELS.get(k, k)


CHOICES = {
    "gender": ["Female", "Male"],
    "emvi": ["No", "Yes"],
    "extramesorectal_ln": ["No", "Yes"],
    "preop_hiso_type": ["Mucinous", "Normal", "Signet ring"],
    "preop_tumor_grading": [
        "Moderately differentiated",
        "Poorly differentiated",
        "Well-differentiated",
    ],
    "clinical_t_stage": ["1", "2", "3", "4"],
    "clinical_n_stage": ["0", "1", "2"],
    "mrf_involvement": ["<1mm (involved)", "1-2mm (threatened)", ">2mm"],
    "kras": ["No", "Yes"],
    "braf_mutation": ["No", "Yes"],
    "tnt_regimen": ["CAPOX", "FOLFOX", "FOLFOXIRI", "Other"],
    "tnt_course": ["Con+LCCRT", "Con+SCRT", "Ind+LCCRT", "Ind+SCRT"],
    "asa": ["1", "2", "3", "4"],
    "MR_TRG": ["1", "2", "3", "4"],
}

ALLOW_NA = {
    "gender": False,
    "emvi": True,
    "extramesorectal_ln": True,
    "preop_hiso_type": True,
    "preop_tumor_grading": True,
    "clinical_t_stage": False,
    "clinical_n_stage": False,
    "mrf_involvement": True,
    "kras": True,
    "braf_mutation": True,
    "tnt_regimen": True,
    "tnt_course": False,
    "asa": True,
    "MR_TRG": True,  # numeric column will be NaN if missing
}


# ----------------------------
# UI helper constructors  ✅ (this is what you were missing)
# ----------------------------
def select_input(id_: str, key: str) -> ui.Tag:
    return ui.input_select(
        id_,
        label(key),
        choices=[MISSING_TOKEN] + CHOICES[key],
        selected=(MISSING_TOKEN if ALLOW_NA.get(key, True) else CHOICES[key][0]),
        width="100%",
    )


def numeric_input(id_: str, key: str, value: float) -> ui.Tag:
    return ui.input_numeric(
        id_,
        label(key),
        value=value,
        width="100%",
    )


# ----------------------------
# Artifacts (lazy-loaded + cached)
# ----------------------------
@dataclass(frozen=True)
class Artifacts:
    clf: Any
    model_cols: list[str]
    alpha: float
    beta: float
    base_row: pd.Series


def _check_exists(paths: list[Path]) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(
            "Missing required files next to app.py:\n"
            + "\n".join(f" - {m}" for m in missing)
            + f"\n\nLooking in: {HERE}"
        )


def _load_artifacts() -> Artifacts:
    _check_exists([MODEL_PATH, MODEL_COLS_PATH, PLATT_PATH])

    clf = joblib.load(MODEL_PATH)

    # Force CPU if model exposes a device attribute (helps on Connect)
    if hasattr(clf, "device"):
        try:
            clf.device = "cpu"
        except Exception:
            pass

    with MODEL_COLS_PATH.open("r", encoding="utf-8") as f:
        model_cols = json.load(f)

    if not isinstance(model_cols, list) or not all(isinstance(x, str) for x in model_cols):
        raise RuntimeError("model_cols.json must be a JSON list of strings")

    with PLATT_PATH.open("r", encoding="utf-8") as f:
        platt = json.load(f)

    if not isinstance(platt, dict) or "alpha_ens" not in platt or "beta_ens" not in platt:
        raise RuntimeError("platt_params.json must have keys: 'alpha_ens', 'beta_ens'")

    alpha = float(platt["alpha_ens"])
    beta = float(platt["beta_ens"])

    base_row = pd.Series(0.0, index=model_cols, dtype="float64")

    return Artifacts(clf=clf, model_cols=model_cols, alpha=alpha, beta=beta, base_row=base_row)


_ARTIFACTS: Artifacts | None = None


def get_artifacts() -> Artifacts:
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = _load_artifacts()
    return _ARTIFACTS


# ----------------------------
# Encoding
# ----------------------------
def _coerce_float(x: Any, field: str) -> float:
    if x is None or x == "":
        raise ValueError(f"{label(field)} must be a number")
    try:
        val = float(x)
    except Exception:
        raise ValueError(f"{label(field)} must be a number")
    if not np.isfinite(val):
        raise ValueError(f"{label(field)} must be finite")
    return val


def set_onehot(row: pd.Series, prefix: str, choice: str, allow_na: bool = True) -> None:
    if choice == MISSING_TOKEN:
        if allow_na:
            na_col = f"{prefix}_NA"
            if na_col in row.index:
                row[na_col] = 1.0
        return
    col = f"{prefix}_{choice}"
    if col in row.index:
        row[col] = 1.0


def build_encoded_row(inputs: dict[str, Any]) -> pd.DataFrame:
    a = get_artifacts()
    row = a.base_row.copy()

    # numeric continuous
    for f in NUM_FIELDS:
        if f in row.index:
            row[f] = _coerce_float(inputs[f], f)

    # MR_TRG numeric-only
    if "MR_TRG" in row.index:
        v = inputs["MR_TRG"]
        row["MR_TRG"] = np.nan if v == MISSING_TOKEN else _coerce_float(v, "MR_TRG")

    # ASA numeric + one-hot (if your model_cols contains 'asa' AND 'asa_*')
    if "asa" in row.index:
        v = inputs["asa"]
        row["asa"] = np.nan if v == MISSING_TOKEN else _coerce_float(v, "asa")

    # one-hots
    for k in CATEGORICAL_KEYS:
        set_onehot(row, k, inputs[k], allow_na=ALLOW_NA.get(k, True))

    X_row = row.to_frame().T
    return X_row[a.model_cols]


def apply_platt(p_raw: float, alpha: float, beta: float) -> float:
    p = float(np.clip(p_raw, 1e-6, 1 - 1e-6))
    z = float(logit(p))
    return float(expit(alpha + beta * z))


def current_summary_df(inputs: dict[str, Any]) -> pd.DataFrame:
    ordered = [
        "gender",
        "emvi",
        "extramesorectal_ln",
        "preop_hiso_type",
        "preop_tumor_grading",
        "clinical_t_stage",
        "clinical_n_stage",
        "mrf_involvement",
        "kras",
        "braf_mutation",
        "tnt_regimen",
        "tnt_course",
        "asa",
        "MR_TRG",
    ] + NUM_FIELDS

    return pd.DataFrame(
        {"variable": [label(v) for v in ordered], "value": [inputs.get(v, "") for v in ordered]}
    )


# ----------------------------
# UI (sidebar + tightened spacing)
# ----------------------------
tight_css = f"""
.container-fluid {{
  max-width: {UI_WIDTH};
}}

.shiny-input-container {{
  margin-bottom: 0.35rem !important;
}}

.control-label {{
  width: {DESC_WIDTH} !important;
  margin-bottom: 0.15rem !important;
}}

.form-group {{
  margin-bottom: 0.35rem !important;
}}

.pccr-sidebar {{
  max-height: calc(100vh - 140px);
  overflow-y: auto;
  padding-right: 0.5rem;
}}
"""

app_ui = ui.page_sidebar(
    sidebar=ui.sidebar(
        ui.div(
            ui.h4("Inputs"),
            select_input("gender", "gender"),
            select_input("emvi", "emvi"),
            select_input("extramesorectal_ln", "extramesorectal_ln"),
            select_input("preop_hiso_type", "preop_hiso_type"),
            select_input("preop_tumor_grading", "preop_tumor_grading"),
            select_input("clinical_t_stage", "clinical_t_stage"),
            select_input("clinical_n_stage", "clinical_n_stage"),
            select_input("mrf_involvement", "mrf_involvement"),
            select_input("kras", "kras"),
            select_input("braf_mutation", "braf_mutation"),
            select_input("tnt_regimen", "tnt_regimen"),
            select_input("tnt_course", "tnt_course"),
            select_input("asa", "asa"),
            select_input("MR_TRG", "MR_TRG"),
            numeric_input("bmi", "bmi", NUM_DEFAULTS["bmi"]),
            numeric_input(
                "tumor_distance_from_arj",
                "tumor_distance_from_arj",
                NUM_DEFAULTS["tumor_distance_from_arj"],
            ),
            numeric_input(
                "tumor_cradiocaudal_length",
                "tumor_cradiocaudal_length",
                NUM_DEFAULTS["tumor_cradiocaudal_length"],
            ),
            numeric_input("pretreatment_cea", "pretreatment_cea", NUM_DEFAULTS["pretreatment_cea"]),
            numeric_input("number_of_cycles", "number_of_cycles", NUM_DEFAULTS["number_of_cycles"]),
            ui.input_action_button("predict", "Predict likelihood of pcCR", class_="btn-primary", width="100%"),
            ui.input_action_button("reset", "Reset", width="100%"),
            class_="pccr-sidebar",
        ),
        width=420,
    ),
    # main page body starts here
    ui.tags.style(tight_css),
    ui.h3("pcCR prediction"),
    ui.p(
        "TabPFN model trained on a cohort of 308 patients undergoing TNT+TME, "
        "intended to use persistent clinical complete response (pcCR) among "
        "patients being considered for W/W after TNT."
    ),
    ui.card(
        ui.card_header("Result"),
        ui.output_text_verbatim("result_txt"),
    ),
    ui.br(),
    ui.card(
        ui.card_header("Inputs (summary)"),
        ui.output_data_frame("summary_tbl"),
    ),
)


# ----------------------------
# Server
# ----------------------------
def server(input, output, session):
    result_state = reactive.Value({"status": "idle", "msg": ""})

    def read_inputs() -> dict[str, Any]:
        d: dict[str, Any] = {k: getattr(input, k)() for k in UI_SELECT_KEYS}
        d.update({k: getattr(input, k)() for k in NUM_FIELDS})
        return d

    @output
    @render.data_frame
    def summary_tbl():
        return render.DataGrid(current_summary_df(read_inputs()), height="360px", width="100%")

    @reactive.effect
    @reactive.event(input.predict)
    def _do_predict():
        try:
            a = get_artifacts()
            inputs = read_inputs()
            X_row = build_encoded_row(inputs)

            p_raw = float(a.clf.predict_proba(X_row)[:, 1][0])
            p_platt = apply_platt(p_raw, a.alpha, a.beta)

            msg = (
                f"Model: {MODEL_PATH.name}\n"
                f"Platt params: {PLATT_PATH.name} (alpha_ens={a.alpha:.4f}, beta_ens={a.beta:.4f})\n\n"
                f"Raw TabPFN probability:       {p_raw:.4f}  ({100 * p_raw:.1f}%)\n"
                f"Platt-calibrated probability: {p_platt:.4f}  ({100 * p_platt:.1f}%)"
            )
            result_state.set({"status": "ok", "msg": msg})
        except Exception as e:
            result_state.set({"status": "error", "msg": f"Prediction failed:\n{type(e).__name__}: {e}"})

    @reactive.effect
    @reactive.event(input.reset)
    def _do_reset():
        for k in UI_SELECT_KEYS:
            if ALLOW_NA.get(k, True):
                session.send_input_message(k, {"value": MISSING_TOKEN})
            else:
                session.send_input_message(k, {"value": CHOICES[k][0]})

        for k in NUM_FIELDS:
            session.send_input_message(k, {"value": NUM_DEFAULTS[k]})

        result_state.set({"status": "idle", "msg": ""})

    @output
    @render.text
    def result_txt():
        s = result_state.get()
        return s["msg"] if s["msg"] else "Click 'Predict likelihood of pcCR' to run the model."


app = App(app_ui, server)