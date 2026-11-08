"""Chatbot-style Streamlit UI over the Portfolio Agent -- optional, per the README.

    streamlit run streamlit_app.py

Thin wrapper: all the real logic lives in agent.py/sql_tool.py/exposure.py/respond.py.
This file just holds the conversation and renders what they return.
"""
import pandas as pd
import streamlit as st

from agent import PortfolioAgent
from respond import stream_natural_response

st.set_page_config(page_title="Portfolio Analytics Agent", page_icon="\U0001F4C8", layout="centered")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; padding-bottom: 7rem; max-width: 800px; }

    [data-testid="stChatMessage"] {
        border-radius: 18px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: rgba(99, 132, 255, 0.10);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: rgba(120, 120, 130, 0.10);
    }

    [data-testid="stChatInput"] textarea {
        border-radius: 14px;
    }

    .app-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.1rem; }
    .app-header h1 { margin: 0; font-size: 1.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

USER_AVATAR = "\U0001F9D1"
ASSISTANT_AVATAR = "\U0001F4C8"

STARTER_QUESTIONS = [
    "How many portfolios do we have?",
    "What are the sector exposures for the Tech Innovation Fund?",
    "Which portfolio has the highest AUM?",
]


@st.cache_resource
def get_agent() -> PortfolioAgent:
    return PortfolioAgent()


if "messages" not in st.session_state:
    st.session_state.messages = []


def render_query_result(data: dict) -> None:
    st.code(data["sql"], language="sql")
    if not data["rows"]:
        st.write("No results.")
        return

    df = pd.DataFrame(data["rows"], columns=data["columns"])
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    label_cols = [c for c in df.columns if c not in numeric_cols]

    if len(df) == 1 and numeric_cols:
        # A single aggregate answer (COUNT, SUM, AVG, ...) -- a headline number reads
        # better than a one-row table.
        cols = st.columns(len(df.columns))
        for col_widget, col_name in zip(cols, df.columns):
            value = df.iloc[0][col_name]
            display_value = f"{value:,.2f}" if isinstance(value, float) else value
            col_widget.metric(col_name, display_value)
    elif numeric_cols and label_cols:
        # Several numeric columns can be on very different scales (a dollar total next to
        # a percentage) -- charting all of them together would be misleading, so chart
        # only the last one, which by convention is the final/headline computed metric in
        # these generated queries (a ratio or percentage, typically).
        chart_column = numeric_cols[-1]
        st.bar_chart(df.set_index(label_cols[0])[[chart_column]])

    st.dataframe(df, hide_index=True, use_container_width=True)


def render_supporting_data(result: dict) -> None:
    """The raw evidence behind the natural-language answer -- tucked away, not the focus."""
    if len(result["steps"]) > 1:
        st.markdown(f"_Needed {len(result['steps'])} steps:_")

    for step in result["steps"]:
        if step["tool"] == "get_sector_exposure" and step["result"]:
            exposure = step["result"]["sector_exposure"]
            st.code(step["result"]["sql"], language="sql")
            st.bar_chart(pd.Series(exposure, name="Percent"))
            st.dataframe(pd.DataFrame(exposure.items(), columns=["Sector", "Percent"]), hide_index=True)
        elif step["tool"] == "query_database" and step["result"]:
            render_query_result(step["result"])


def render_history() -> None:
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            result = message.get("result")
            if message["role"] == "assistant" and result and not result["error"] and result["tool"]:
                with st.expander("Show the data behind this answer"):
                    render_supporting_data(result)


def handle_question(question: str) -> None:
    """Render the exchange live (user bubble instantly, assistant reply streamed in),
    then save both to history so a future rerun replays them the same way."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Looking that up..."):
            result = get_agent().answer_question(question)
        full_response = st.write_stream(stream_natural_response(question, result))
        if not result["error"] and result["tool"]:
            with st.expander("Show the data behind this answer"):
                render_supporting_data(result)

    st.session_state.messages.append({"role": "assistant", "content": full_response, "result": result})


# --- Layout ---

st.markdown(
    '<div class="app-header"><h1>\U0001F4C8 Portfolio Analytics Agent</h1></div>',
    unsafe_allow_html=True,
)
st.badge("Powered by Gemini", icon="✨", color="blue")
st.caption("Ask me anything about portfolios, holdings, securities, or sector exposure.")
st.write("")

with st.sidebar:
    st.subheader("\U0001F4C8 Portfolio Analytics Agent")
    st.caption("A Gemini-powered agent over a portfolio database -- text-to-SQL and sector exposure, with retrieval-augmented few-shot generation for harder queries.")
    st.divider()
    if st.button("\U0001F5D1️ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(f"{len(st.session_state.messages) // 2} question(s) asked this session.")

render_history()

if not st.session_state.messages:
    st.caption("Try asking:")
    cols = st.columns(len(STARTER_QUESTIONS))
    for col, q in zip(cols, STARTER_QUESTIONS):
        if col.button(q, use_container_width=True):
            handle_question(q)
            st.rerun()

if question := st.chat_input("Ask a question..."):
    handle_question(question)
