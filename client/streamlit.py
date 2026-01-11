import json
from query_llm import make_llm_call_hf
from query_normalizer import normalize
import streamlit as st
import requests

# -----------------------
# Configuration
# -----------------------
BACKEND_URL = "http://localhost:8000/search"

st.set_page_config(
    page_title="ECB Speech Search Engine",
    layout="wide"
)

st.title("🔍 ECB Speech Search Engine")
is_search_mode_llm = st.toggle(
    "LLM enhanced search",
    value=True,
    help="Switch between filter-based search and LLM-enhanced search"
)



def generate_results(data, is_search_mode_llm=False):
    print('Generating results on UI ...')
    results = data["matched_speeches"] if is_search_mode_llm else data["results"]
    st.subheader(f"Results ({len(results)})")

    if not results:
        st.info("No results found.")
    else:
        for item in results:
            with st.expander(
                f"{item['title']} — {item['speaker']} ({item['date']})"
            ):
                if item.get("url"):
                    st.markdown(f"[🔗 Open speech]({item['url']})")
                st.markdown(f"**Title:** {item['title']}")
                st.markdown(f"**Speaker:** {item['speaker']}")
                st.markdown(f"**Date:** {item['date']}")

                if is_search_mode_llm  and item.get("llm_summary"):
                    st.markdown("**LLM Summary:**")
                    st.markdown(f"{item['llm_summary']}")
                

                elif item.get("topics"):
                    st.markdown("**Topics:**")
                    for t in item["topics"]:
                        st.markdown(f"- {t}")
                
                st.markdown("**Why this speech is relevant:**")
                st.markdown(f"{item['provenance']}")
                st.markdown("**LLM Confidence:**")
                st.markdown(f"{item['llm_confidence'] * 100:.2f}%")

        print('Results displayed')


    if data.get("verified"):
        st.success("Results verified ✔️")




def query_suggestions(suggestions, target_key):
    cols = st.columns(len(suggestions))
    for col, text in zip(cols, suggestions):
        with col:
            st.button(
                f"➕ {text}",
                key=f"suggestion_{text}",
                on_click=lambda t=text: st.session_state.update({target_key: t})
            )


#  FILTER SEARCH MODE

if not is_search_mode_llm: 

    st.subheader("🤖 Filter Search")

    # -----------------------
    # Search Controls
    # -----------------------
    with st.form("search_form"):
        query = st.text_input(
            "Search query *",
            placeholder="e.g. Monetary policy, inflation"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            from_date = st.date_input("From date (optional)", value=None)

        with col2:
            to_date = st.date_input("To date (optional)", value=None)

        with col3:
            result_count = st.selectbox(
                "Number of results",
                options=[1, 2, 5, 10, 20],
                index=2
            )

        col4, col5 = st.columns(2)

        with col4:
            speaker = st.selectbox(
                "Speaker (optional)",
                options=[
                    "",
                    "Christine Lagarde",
                    "Frank Elderson",
                    "Philip R. Lane",
                    "Isabel Schnabel",
                    "Piero Cipollone",
                    "Luis de Guindos",
                    "Christine Lagarde,Philip R. Lane",
                    "Fabio Panetta",
                    "Yves Mersch",
                    "Benoît Cœuré",
                    "Sabine Lautenschläger",
                    "Mario Draghi",
                    "Peter Praet",
                    "Vítor Constâncio",
                    "Jörg Asmussen",
                    "José Manuel González-Páramo",
                    "Lorenzo Bini Smaghi",
                    "Jürgen Stark",
                    "Jean-Claude Trichet",
                    "Gertrude Tumpel-Gugerell",
                    "Lucas Papademos",
                    "Otmar Issing",
                    "Tommaso Padoa-Schioppa",
                    "Eugenio Domingo Solans",
                    "Willem F. Duisenberg"
                ],
                index=0
            )

        with col5:
            topic = st.text_input(
                "Topic (optional)",
                placeholder="e.g. Monetary policy"
            )

        submitted = st.form_submit_button("Search")

    # Validation
    if submitted and not query.strip():
        st.error("Search query is required.")

    # Backend Call
    if submitted and query.strip():
        params = {
            "q": query,
            "limit": result_count
        }

        if from_date:
            params["from_date"] = from_date.isoformat()

        if to_date:
            params["to_date"] = to_date.isoformat()

        if speaker.strip():
            params["speaker"] = speaker.strip()

        if topic.strip():
            params["topic"] = topic.strip()

        with st.spinner("Searching..."):
            try:
                response = requests.get(BACKEND_URL, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Backend error: {e}")
                st.stop()

            generate_results(data)      

#  LLM SEARCH MODE
if is_search_mode_llm:
    st.subheader("🤖 LLM Enhanced Search")

    llm_query = st.text_input(
        "Ask a question *",
        placeholder="e.g. what ECB president said recently about inflation risks",
        key="llm_query"
    )

    query_suggestions(
        suggestions=[
            # "What has the ECB president said recently about inflation risks?",
            "What was recently said about inflation risks?",
            "Show speeches where Lagarde talks about climate.",
            "What did ECB board members say after the energy price shock?",
            "Banking regulation talks",
        ],
        target_key="llm_query"
    )


    run_llm_search = st.button("Search")

    if run_llm_search and not llm_query.strip():
        st.error("Query is required.")

    if run_llm_search and llm_query.strip():
        # ---- Intent detection (template stub) ----
        intent_payload = normalize(llm_query)

        params = {
            "q": intent_payload.get("expanded_query") or intent_payload.get("raw_query"),
            "limit": 15
        }

        # ---- speaker from entities ----
        for ent in intent_payload.get("entities", []):
            if ent.get("type") == "speaker" and ent.get("canonical"):
                params["speaker"] = ent["canonical"]
                break

        # ---- date range ----
        date_range = intent_payload.get("date_range")
        if date_range:
            if date_range.get("start"):
                params["from_date"] = date_range["start"]
            if date_range.get("end"):
                params["to_date"] = date_range["end"]

        # ---- topic / indicator ----
        indicators = [
            e["canonical"]
            for e in intent_payload.get("entities", [])
            if e.get("type") == "indicator"
        ]

        if indicators:
            params["topic"] = indicators[0]  # first for now


        # ---- Backend call ----
        with st.spinner("Searching with LLM..."):
            try:
                response = requests.get(BACKEND_URL, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Backend error: {e}")
                st.stop()

            # preprocess data_input from data removing unnecessary fields like provenance and confidence
            for item in data['results']:
                item.pop('provenance')
                item.pop('confidence')
            print('preprocessed data to avoid llm input size error :: ', data['results'])
            
            # make llm call 
            try:
                llm_res = make_llm_call_hf(llm_query, data['results'])

                # Always normalize to dict
                if isinstance(llm_res, str):
                    llm_res = json.loads(llm_res)

                if isinstance(llm_res, list):
                    llm_res = llm_res[0] if llm_res and isinstance(llm_res[0], dict) else {}

                if not isinstance(llm_res, dict):
                    llm_res = {}

                if llm_res.get('matched_speeches'):
                    generate_results(llm_res, is_search_mode_llm)
                else:
                    explaining_text = llm_res.get('explanation', '')
                    st.info(f"No results match found.\n{explaining_text}")
            except Exception as e:
                st.error(f"LLM processing error: {e}")
                generate_results(data)