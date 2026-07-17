import streamlit as st

from mcp_server.client import call_tool_sync
from workflow.graph import support_graph

st.set_page_config(
    page_title="EasySupport AI",
    page_icon="🤖",
    layout="wide",
)

st.title("EasySupport AI")
st.caption("Assistente de suporte com RAG, classificação e ações governadas")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_action" not in st.session_state:
    st.session_state.pending_action = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Fontes utilizadas"):
                for source in message["sources"]:
                    st.write(
                        f"- {source['source']} — {source.get('chunk_id')}"
                    )

question = st.chat_input("Descreva sua dúvida ou problema")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("Analisando solicitação...", expanded=True) as status:
            result = support_graph.invoke({"question": question})
            status.update(label="Análise concluída", state="complete")

        answer = result["answer"]
        st.markdown(answer["answer"])

        st.write(
            "**Urgência sugerida:**",
            answer["urgency"],
        )
        st.write(
            "**Confiança:**",
            f"{answer['classifier_confidence']:.1%}",
        )

        if answer["sources"]:
            with st.expander("Fontes utilizadas"):
                for source in answer["sources"]:
                    st.write(
                        f"- {source['source']} — {source.get('chunk_id')}"
                    )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer["answer"],
            "sources": answer["sources"],
        })

        if result.get("requires_approval") and result.get("proposed_action"):
            st.session_state.pending_action = {
                "tool": result["proposed_action"],
                "arguments": {
                    "title": f"Chamado: {question[:60]}",
                    "description": question,
                    "urgency": result["urgency"],
                },
            }

if st.session_state.pending_action:
    st.warning("O sistema propôs a abertura de um chamado.")

    col_confirm, col_cancel = st.columns(2)

    with col_confirm:
        if st.button("Confirmar abertura", type="primary"):
            action = st.session_state.pending_action
            tool_result = call_tool_sync(
                action["tool"],
                action["arguments"],
            )
            st.success("Ação executada após confirmação.")
            st.json(tool_result)
            st.session_state.pending_action = None

    with col_cancel:
        if st.button("Cancelar"):
            st.session_state.pending_action = None
            st.info("A ação foi cancelada.")