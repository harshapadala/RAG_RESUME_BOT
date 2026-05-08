import streamlit as st
import tempfile
import os
from rag_engine import build_vectorstore, get_qa_chain, ask

# ── Page Setup ─────────────────────────────────────────────
st.set_page_config(page_title="RAG Resume Analyzer", page_icon="🤖", layout="wide")

st.title("📄 RAG Resume Analyzer")
st.caption("Upload a resume (PDF) and analyze or ask questions")

# ── Sidebar Upload ─────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Resume")
    uploaded = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded:
        with st.spinner("Processing PDF... ⏳"):
            try:
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                # Reset session if new file
                if (
                    "last_file" not in st.session_state
                    or st.session_state.last_file != uploaded.name
                ):
                    st.session_state.clear()

                    vectorstore = build_vectorstore(tmp_path)
                    chain = get_qa_chain(vectorstore)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.chain = chain
                    st.session_state.last_file = uploaded.name
                    st.session_state.messages = []

                os.unlink(tmp_path)

                st.success("✅ Resume processed! Ready to analyze.")

            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")

# ── Initialize Chat ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── ATS ANALYZER BUTTON ────────────────────────────────────
if "chain" in st.session_state:
    if st.button("📊 Analyze Resume (ATS)"):
        with st.spinner("Analyzing resume..."):
            prompt = """
            Analyze this resume and provide:

            1. Candidate Name
            2. Skills (bullet points)
            3. Projects (short description)
            4. Experience Summary
            5. Strengths
            6. Weaknesses
            7. ATS Score (out of 100)
            8. Suggestions for improvement

            Format clearly.
            """

            result = ask(st.session_state.chain, prompt)
            analysis = result.get("answer", "No result")

            st.subheader("📊 ATS Analysis")
            st.write(analysis)

            # Store for download
            st.session_state.analysis_result = analysis

# ── QUICK ACTION BUTTONS ───────────────────────────────────
if "chain" in st.session_state:
    st.subheader("⚡ Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🧠 Extract Skills"):
            result = ask(st.session_state.chain, "List all skills from the resume")
            st.write(result.get("answer"))

    with col2:
        if st.button("📁 List Projects"):
            result = ask(st.session_state.chain, "List all projects with description")
            st.write(result.get("answer"))

    with col3:
        if st.button("📄 Summarize Resume"):
            result = ask(st.session_state.chain, "Summarize the resume professionally")
            st.write(result.get("answer"))

# ── DOWNLOAD BUTTON ────────────────────────────────────────
if "analysis_result" in st.session_state:
    st.download_button(
        label="⬇ Download Analysis",
        data=st.session_state.analysis_result,
        file_name="resume_analysis.txt",
        mime="text/plain"
    )

# ── CHAT HISTORY ───────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── CHAT INPUT ─────────────────────────────────────────────
question = st.chat_input("💬 Ask something about the resume...")

if question:
    if "chain" not in st.session_state:
        st.warning("⚠️ Please upload a resume first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤔"):
                result = ask(st.session_state.chain, question)

                answer = result.get("answer", "No answer")
                sources = result.get("sources", [])

            st.write(answer)

            # Show sources
            if sources:
                with st.expander("📚 Source Chunks"):
                    for i, doc in enumerate(sources, 1):
                        page = doc.metadata.get("page", "N/A")
                        content = doc.page_content[:300]

                        st.markdown(f"**Chunk {i} (Page {page})**")
                        st.write(content + "...")
                        st.divider()
            else:
                st.info("No source documents available.")

        st.session_state.messages.append({"role": "assistant", "content": answer})