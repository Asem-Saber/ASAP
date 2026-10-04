import streamlit as st

from asap.config import get_settings
from asap.ui.client import ApiClient, ApiError
from asap.ui.examples import EXAMPLES

TEXT_KEY = "review_text"
RESULT_KEY = "last_result"

_VERDICT_COLORS = {"positive": "#16a34a", "negative": "#dc2626"}
_NEUTRAL_COLOR = "#6b7280"

_RTL_CSS = """
<style>
.stTextArea textarea { direction: rtl; text-align: right; font-size: 1.05rem; }
</style>
"""


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient.from_settings()


def use_example(text: str) -> None:
    st.session_state[TEXT_KEY] = text


def render_result(sentiment: str, confidence: float) -> None:
    color = _VERDICT_COLORS.get(sentiment, _NEUTRAL_COLOR)
    st.markdown(
        f"<p style='font-size:2.25rem;font-weight:700;color:{color};margin:0'>"
        f"{sentiment.capitalize()}</p>",
        unsafe_allow_html=True,
    )
    st.caption(f"Confidence: {confidence:.1%}")
    st.progress(confidence)


cfg = get_settings()

st.set_page_config(
    page_title="Arabic Sentiment Analysis", page_icon="💬", layout="centered"
)
st.markdown(_RTL_CSS, unsafe_allow_html=True)

st.title("Arabic Sentiment Analysis")
st.caption(
    "AraBERTv02 fine-tuned on 247k Arabic product reviews - 0.948 weighted F1 "
    "on a held-out test split, served as ONNX on CPU."
)

st.write("Try an example:")
for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES, strict=True):
    column.button(
        example.label,
        key=f"example_{example.label}",
        on_click=use_example,
        args=(example.text,),
        use_container_width=True,
    )

text = st.text_area(
    "Arabic review",
    key=TEXT_KEY,
    height=160,
    max_chars=cfg.api.max_text_chars,
    placeholder="اكتب مراجعة بالعربية...",
)

if st.button("Analyze", type="primary", disabled=not text.strip()):
    with st.spinner("Classifying..."):
        try:
            prediction = get_client().predict(text)
        except ApiError as exc:
            st.session_state[RESULT_KEY] = {"text": text, "error": str(exc)}
        else:
            st.session_state[RESULT_KEY] = {
                "text": text,
                "sentiment": prediction.sentiment,
                "confidence": prediction.confidence,
            }

result = st.session_state.get(RESULT_KEY)
if result is not None:
    if "error" in result:
        st.error(result["error"])
    else:
        render_result(result["sentiment"], result["confidence"])

    if result["text"] != text:
        st.caption(
            "This is the result for the text analysed earlier, not what is in "
            "the box now. Press Analyze again."
        )
