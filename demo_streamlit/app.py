from __future__ import annotations

import streamlit as st

from data_browser import (
    get_example_by_index,
    load_pairs,
    random_index,
    resolve_image_path,
)
from gradcam import build_gradcam
from inference import run_inference
from negative_pairs import BUTTONS, make_negative
from report_diff import build_report_diff_html, negative_pair_explanation
from text_attribution import compute_text_attribution
from paths import DEFAULT_THRESHOLD


st.set_page_config(page_title="CXR Consistency Demo", layout="wide")


st.markdown(
    """
    <style>
    .main .block-container {padding-top: 0.85rem; max-width: 1240px;}
    h1 {font-size: 2rem; margin-bottom: 0.1rem;}
    h2, h3 {margin-top: 0.55rem; margin-bottom: 0.35rem;}
    div[data-testid="stVerticalBlock"] {gap: 0.45rem;}
    .result-card {
        border: 1px solid #cfd6e3;
        border-radius: 8px;
        padding: 14px 16px;
        background: #f8fafc;
        margin-top: 0.5rem;
    }
    .result-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        align-items: stretch;
    }
    .result-item {border-left: 3px solid #64748b; padding-left: 10px; min-height: 58px;}
    .result-label {font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.02em;}
    .result-value {font-size: 1.22rem; font-weight: 700; color: #0f172a; overflow-wrap: anywhere;}
    .verdict-ok {color: #047857;}
    .verdict-bad {color: #b91c1c;}
    .info-card {border: 1px solid #d9dee7; border-radius: 8px; padding: 12px 14px; background: #f8fafc;}
    .snippet-box {
        border: 1px solid #d9dee7;
        border-radius: 8px;
        padding: 10px 12px;
        background: #ffffff;
        min-height: 118px;
        font-size: 0.9rem;
        line-height: 1.38;
    }
    .snippet-title {font-size: 0.76rem; color: #64748b; font-weight: 700; margin-bottom: 6px;}
    .attribution-box {
        border: 1px solid #d9dee7;
        border-radius: 8px;
        padding: 12px 14px;
        background: #ffffff;
        max-height: 260px;
        overflow-y: auto;
    }
    table.diff {font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.78rem; width: 100%;}
    .diff_add {background-color: #dcfce7;} .diff_sub {background-color: #fee2e2;} .diff_chg {background-color: #fef9c3;}
    @media (max-width: 900px) {.result-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}}
    </style>
    """,
    unsafe_allow_html=True,
)


def set_index(index: int) -> None:
    st.session_state.current_index = int(index)
    st.session_state.active_report = None
    st.session_state.negative_type = None
    st.session_state.diff_html = None
    st.session_state.negative_explanation = None
    st.session_state.prediction = None
    st.session_state.text_attr_key = None
    st.session_state.text_attr_table = None
    st.session_state.text_attr_html = None


def current_example(df):
    if "current_index" not in st.session_state:
        set_index(random_index(df))
    return get_example_by_index(df, st.session_state.current_index)


def classify_text(probability: float, threshold: float) -> str:
    return "Согласовано" if probability >= threshold else "Не согласовано"



def result_card(probability: float | None, pair_type: str) -> None:
    threshold = DEFAULT_THRESHOLD
    if probability is None:
        probability_text = "—"
        verdict_text = "Не проверено"
        verdict_class = ""
    else:
        probability_text = f"{probability:.3f}"
        verdict_text = classify_text(probability, threshold)
        verdict_class = "verdict-ok" if probability >= threshold else "verdict-bad"

    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-grid">
            <div class="result-item">
              <div class="result-label">Probability</div>
              <div class="result-value">{probability_text}</div>
            </div>
            <div class="result-item">
              <div class="result-label">Verdict</div>
              <div class="result-value {verdict_class}">{verdict_text}</div>
            </div>
            <div class="result-item">
              <div class="result-label">Pair type</div>
              <div class="result-value">{pair_type}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_current_inference(image_path, report: str) -> None:
    if image_path.is_file():
        try:
            st.session_state.prediction = run_inference(image_path, report)["probability"]
        except Exception as exc:
            st.session_state.prediction = None
            st.error(f"Ошибка инференса: {exc}")
    else:
        st.error("Нельзя выполнить инференс: снимок не найден")


st.title("CXR Consistency Demo")
st.caption("Мультимодальная проверка согласованности рентгеновского снимка грудной клетки и медицинского заключения")

try:
    pairs_df = load_pairs()
except Exception as exc:
    st.error(f"Не удалось загрузить CSV с парами: {exc}")
    st.stop()

st.caption("Positive pairs: 160727")

with st.container():
    col_a, col_b, col_spacer = st.columns([1, 1, 3])
    if col_a.button("Случайный пример", use_container_width=True):
        set_index(random_index(pairs_df))
    if col_b.button("Следующий пример", use_container_width=True):
        set_index(st.session_state.get("current_index", -1) + 1)

example = current_example(pairs_df)
image_path = resolve_image_path(example["image_path"])
original_report = example["report"]
active_report = st.session_state.get("active_report") or original_report
pair_type = st.session_state.get("negative_type") or "positive"

left, right = st.columns([0.95, 1.05], gap="large")
with left:
    st.subheader("Рентген")
    if image_path.is_file():
        st.image(str(image_path), use_container_width=True)
    else:
        st.error(f"Файл снимка не найден: {image_path}")
    meta_a, meta_b = st.columns(2)
    meta_a.metric("patient_id", example["subject_id"])
    meta_b.metric("study_id", example["study_id"])

with right:
    st.subheader("Отчет")
    st.text_area(
        "Текущий отчет",
        active_report,
        height=170,
        disabled=True,
        label_visibility="collapsed",
    )
    if st.session_state.get("negative_type"):
        st.caption(f"negative_type: {st.session_state.negative_type}")
    if st.button("Проверить согласованность", type="primary", use_container_width=True):
        run_current_inference(image_path, active_report)
    result_card(st.session_state.get("prediction"), pair_type)

st.divider()
st.subheader("Создать Hard Negative Pair")
button_cols = st.columns(6)
for idx, (label, kind) in enumerate(BUTTONS):
    if button_cols[idx].button(label, use_container_width=True):
        pair = make_negative(kind, original_report, pairs_df, example["subject_id"])
        st.session_state.active_report = pair.report
        st.session_state.negative_type = pair.kind
        st.session_state.negative_explanation = negative_pair_explanation(pair.kind)
        st.session_state.diff_html = build_report_diff_html(original_report, pair.report)
        run_current_inference(image_path, pair.report)
        st.rerun()

if st.session_state.get("negative_type"):
    st.markdown("### Изменения в отчёте")
    st.info(
        f"negative_type: {st.session_state.negative_type}\n\n"
        f"{st.session_state.get('negative_explanation') or negative_pair_explanation(st.session_state.negative_type)}"
    )
    if st.session_state.get("diff_html"):
        st.markdown(st.session_state.diff_html, unsafe_allow_html=True)
    else:
        st.warning("Изменения не обнаружены или генератор вернул исходный текст.")
    result_card(st.session_state.get("prediction"), st.session_state.negative_type)

st.divider()
st.subheader("Интерпретация решения")
image_interp_col, text_interp_col = st.columns(2, gap="large")

with image_interp_col:
    st.markdown("**Image Grad-CAM**")
    st.caption("Вспомогательная визуализация областей снимка, повлиявших на решение модели. Не является медицинской локализацией патологии.")
    show_cam = st.checkbox("Show Grad-CAM", value=False)
    if show_cam:
        cam_left, cam_right = st.columns(2)
        with st.spinner("Building Grad-CAM..."):
            original_img, heatmap_img, cam_error = build_gradcam(image_path, active_report)
        if original_img is not None:
            cam_left.image(original_img, caption="Original", use_container_width=True)
        if heatmap_img is not None:
            cam_right.image(heatmap_img, caption="Heatmap", use_container_width=True)
        else:
            cam_right.warning(f"Grad-CAM недоступен: {cam_error}")

with text_interp_col:
    st.markdown("**Text attribution**")
    st.caption("Оценка вклада слов через occlusion: слово временно удаляется, затем измеряется изменение probability.")
    if st.session_state.get("prediction") is None:
        st.info("Появится после инференса. Для защиты удобнее смотреть после hard negative pair.")
    else:
        if not st.session_state.get("negative_type"):
            st.caption("Доступно после проверки; наиболее показательно после создания hard negative pair.")
        show_text_attr = st.checkbox("Показать вклад слов отчета", value=False)
        if show_text_attr:
            attr_key = (example["study_id"], pair_type, active_report, round(float(st.session_state.prediction), 6))
            if st.session_state.get("text_attr_key") != attr_key:
                try:
                    progress = st.progress(0.0, text="Text attribution: perturbing report tokens...")
                    table, report_html = compute_text_attribution(
                        image_path=image_path,
                        report=active_report,
                        original_probability=float(st.session_state.prediction),
                        max_words=100,
                        top_k=10,
                        progress_callback=lambda value: progress.progress(value),
                    )
                    progress.empty()
                    st.session_state.text_attr_key = attr_key
                    st.session_state.text_attr_table = table
                    st.session_state.text_attr_html = report_html
                except Exception as exc:
                    st.session_state.text_attr_key = None
                    st.session_state.text_attr_table = None
                    st.session_state.text_attr_html = None
                    st.warning(f"Text attribution недоступен: {exc}")

            if st.session_state.get("text_attr_html"):
                st.markdown(
                    f'<div class="attribution-box">{st.session_state.text_attr_html}</div>',
                    unsafe_allow_html=True,
                )
            if st.session_state.get("text_attr_table") is not None:
                table = st.session_state.text_attr_table
                if table.empty:
                    st.info("Нет значимых слов для анализа после фильтрации стоп-слов.")
                else:
                    st.dataframe(
                        table[["word", "importance", "probability_without_word"]],
                        hide_index=True,
                        use_container_width=True,
                    )

st.divider()
st.markdown(
    """
    <div class="info-card">
    <b>Model:</b> ConvNeXt Tiny + CXR-BERT<br>
    <b>Best ROC-AUC:</b> 0.9673<br>
    <b>Best F1:</b> 0.8337<br>
    <b>Hard negative pairs:</b> negation, laterality, temporal, partial mismatch, pathology swap, random report
    </div>
    """,
    unsafe_allow_html=True,
)
