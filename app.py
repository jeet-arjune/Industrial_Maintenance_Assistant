
import re
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# -------------------- PAGE SETTINGS --------------------

st.set_page_config(
    page_title="Maintena AI",
    page_icon="⚙️",
    layout="wide"
)


# -------------------- WEBSITE DESIGN --------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');

.stApp {
    background: #F4F7FA;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: #10243A;
}

[data-testid="stSidebar"] * {
    color: #F2F7FA;
}

.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    color: #10243A;
}

.hero {
    background: linear-gradient(120deg, #10243A, #17485A, #0E9384);
    padding: 28px 32px;
    border-radius: 18px;
    color: white;
    margin-bottom: 22px;
}

.hero h1 {
    color: white !important;
    font-size: 2rem;
}

.hero p {
    color: #D8E8EF;
}

.tile {
    background: white;
    border: 1px solid #DFE8EF;
    border-radius: 14px;
    padding: 18px;
    min-height: 110px;
}

.tile h3 {
    font-size: 1.05rem;
}

.stButton > button {
    border-radius: 9px;
    font-weight: 500;
    min-height: 38px;
}

.stButton > button[kind="primary"] {
    background: #0E9384;
    color: white;
    border: none;
}

[data-testid="stChatMessage"] {
    background: white;
    border: 1px solid #DFE8EF;
    border-radius: 13px;
    padding: 12px;
}

/* FIX: VISIBLE UPLOAD BUTTON */

[data-testid="stFileUploader"] button {
    background-color: #0E9384 !important;
    color: #FFFFFF !important;
    border: 1px solid #0E9384 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button span {
    color: #FFFFFF !important;
}

[data-testid="stFileUploader"] button:hover {
    background-color: #0B756B !important;
    color: #FFFFFF !important;
    border-color: #0B756B !important;
}

[data-testid="stFileUploader"] section {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploader"] section small {
    color: #526477 !important;
}

</style>
""", unsafe_allow_html=True)


# -------------------- BUILT-IN MAINTENANCE MANUAL --------------------

MANUAL = """
SECTION 1 - MOTOR MAINTENANCE

Motor overheating can be caused by blocked ventilation, a damaged cooling fan,
excessive load, or poor maintenance. Stop equipment if overheating is suspected.
Notify maintenance personnel. Only trained and authorized personnel should
inspect the motor. Isolate energy and follow lockout/tagout procedures.
Never touch exposed electrical parts.

SECTION 2 - CONVEYOR BELT MAINTENANCE

Common problems include belt misalignment, unusual noise, slipping, and wear.
Stop a damaged or unsafe conveyor and report the problem to maintenance
personnel. Stay away from moving parts. Never repair a conveyor while it is
running. Authorized personnel must isolate energy and follow lockout/tagout
procedures before inspection or repair. Record the issue.

SECTION 3 - PUMP MAINTENANCE

Common pump problems include leakage, vibration, overheating, and reduced flow.
Report serious leaks. Keep a safe distance. Only trained personnel should
perform maintenance after energy isolation.

SECTION 4 - MACHINE NOISE AND VIBRATION

Unusual noise or vibration may indicate loose or worn parts or misalignment.
Stop unsafe equipment and report the problem. Do not remove guards or touch
moving parts. Authorized personnel must isolate energy before inspection.

SECTION 5 - PREVENTIVE MAINTENANCE

Follow the approved maintenance schedule, perform required checks, keep
maintenance records, and report problems promptly.
"""


# -------------------- LOAD AI MODELS --------------------

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )
    return embedding_model, generator


# -------------------- READ UPLOADED MANUALS --------------------

def read_file(file):
    if file.name.lower().endswith(".txt"):
        return file.getvalue().decode("utf-8", errors="ignore")

    pdf = PdfReader(file)
    return "\n".join(
        page.extract_text() or ""
        for page in pdf.pages
    )


# -------------------- SPLIT TEXT --------------------

def split_text(text, size=700):
    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n", text)
        if part.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 1 <= size:
            current = (current + "\n" + paragraph).strip()
        else:
            if current:
                chunks.append(current)

            current = paragraph

            while len(current) > size:
                chunks.append(current[:size])
                current = current[size:]

    if current:
        chunks.append(current)

    return chunks or ([text.strip()] if text.strip() else [])


# -------------------- RETRIEVE RELEVANT INFORMATION --------------------

def retrieve(question, chunks, model):
    if not chunks:
        return []

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]

    scores = np.dot(embeddings, query_embedding)
    indexes = np.argsort(scores)[::-1][:3]

    return [
        (chunks[i], float(scores[i]))
        for i in indexes
    ]


# -------------------- SAFETY-FOCUSED ANSWERS --------------------

def safety_answer(question):
    q = question.lower()

    if any(term in q for term in [
        "conveyor", "belt damage", "belt damaged",
        "belt slipping", "belt misalignment"
    ]):
        return (
            "### Conveyor Belt Maintenance\n\n"
            "**Possible problems**\n"
            "- Belt misalignment\n"
            "- Slipping or wear\n"
            "- Unusual noise\n\n"
            "**Recommended actions**\n"
            "1. Stop the conveyor if it is damaged or unsafe.\n"
            "2. Keep away from moving belts and rollers.\n"
            "3. Report the problem to maintenance personnel.\n"
            "4. Never inspect or repair a running conveyor.\n"
            "5. Authorized personnel must isolate energy and follow "
            "lockout/tagout procedures before inspection or repair.\n"
            "6. Record the problem in the maintenance log.\n\n"
            "**Before restarting:** Authorized personnel should confirm "
            "that the equipment is safe."
        )

    if any(term in q for term in [
        "motor", "overheat", "overheating", "cooling fan"
    ]):
        return (
            "### Motor Overheating\n\n"
            "**Possible causes listed in the manual**\n"
            "- Blocked ventilation\n"
            "- Damaged cooling fan\n"
            "- Excessive load\n"
            "- Poor maintenance\n\n"
            "**Recommended actions**\n"
            "1. Stop the equipment if overheating is suspected.\n"
            "2. Notify maintenance personnel.\n"
            "3. Never touch exposed electrical parts.\n"
            "4. Only trained, authorized personnel should inspect the "
            "motor after energy isolation and lockout/tagout.\n"
            "5. Follow the approved maintenance procedure before restarting.\n\n"
            "**Important:** Do not assume the cause without an inspection."
        )

    if any(term in q for term in [
        "pump", "leakage", "leaking", "reduced flow"
    ]):
        return (
            "### Pump Maintenance\n\n"
            "**Problems covered in the manual**\n"
            "- Leakage\n"
            "- Vibration\n"
            "- Overheating\n"
            "- Reduced flow\n\n"
            "**Recommended actions**\n"
            "1. Keep a safe distance from leaking or unsafe equipment.\n"
            "2. Report serious leaks to maintenance personnel.\n"
            "3. Do not attempt repairs while the pump is operating.\n"
            "4. Only trained personnel should perform maintenance after "
            "energy isolation.\n"
            "5. Follow the approved site procedure.\n\n"
            "The available manual does not provide detailed repair "
            "instructions for individual pump faults."
        )

    if any(term in q for term in [
        "vibration", "unusual noise", "machine noise",
        "strange sound", "rattling"
    ]):
        return (
            "### Unusual Machine Noise or Vibration\n\n"
            "**Possible causes listed in the manual**\n"
            "- Loose parts\n"
            "- Worn parts\n"
            "- Misalignment\n\n"
            "**Recommended actions**\n"
            "1. Stop the equipment if it is unsafe.\n"
            "2. Report the issue to maintenance personnel.\n"
            "3. Do not remove guards or touch moving parts.\n"
            "4. Authorized personnel must isolate energy before inspection.\n"
            "5. Record the problem and follow the approved procedure.\n\n"
            "A proper inspection is needed to identify the actual cause."
        )

    if any(term in q for term in [
        "preventive maintenance", "maintenance schedule",
        "prevent breakdown", "routine maintenance"
    ]):
        return (
            "### Preventive Maintenance\n\n"
            "1. Follow the approved maintenance schedule.\n"
            "2. Perform the required equipment checks.\n"
            "3. Keep accurate maintenance records.\n"
            "4. Report equipment problems promptly.\n"
            "5. Follow approved safety procedures.\n\n"
            "**Purpose:** Identify problems early and help reduce "
            "avoidable equipment breakdowns."
        )

    return None


# -------------------- GENERATE ANSWERS --------------------

def answer_question(question, chunks, model, generator):
    question = question.strip()

    if not question:
        return "Please enter a maintenance question.", []

    sources = retrieve(question, chunks, model)
    fixed_answer = safety_answer(question)

    if fixed_answer:
        return fixed_answer, sources

    relevant_sources = [
        (text, score)
        for text, score in sources
        if score >= 0.25
    ]

    if not relevant_sources:
        return (
            "### I couldn't find enough relevant information\n\n"
            "The available maintenance manuals do not contain enough "
            "relevant information to answer confidently.\n\n"
            "**Next steps**\n"
            "- Check the equipment manufacturer's manual.\n"
            "- Consult authorized maintenance personnel.\n"
            "- Follow your site's approved safety procedures.\n\n"
            "You can upload a relevant PDF or TXT manual and ask again."
        ), sources

    context = "\n\n".join(
        text for text, score in relevant_sources
    )[:2500]

    prompt = f"""
You are Maintena AI, an industrial maintenance assistant.

Answer the user's question using only the supplied maintenance information.
Use simple professional English and clear steps.
Include relevant safety precautions.
Never invent repair procedures or technical specifications.
Recommend trained, authorized personnel for inspections and repairs.
If information is insufficient, say so clearly.

MAINTENANCE INFORMATION:
{context}

QUESTION:
{question}

ANSWER:
"""

    try:
        result = generator(
            prompt,
            max_new_tokens=180,
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.15,
            truncation=True
        )

        answer = result[0]["generated_text"].strip()

        if len(answer) < 20:
            answer = (
                "The available manual does not provide enough information "
                "to answer confidently. Consult the approved manual and "
                "authorized maintenance personnel."
            )

        return answer, relevant_sources

    except Exception:
        return (
            "I couldn't generate an AI answer right now. Please consult "
            "the available maintenance manual and authorized personnel."
        ), relevant_sources


# -------------------- SESSION STATE --------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "questions" not in st.session_state:
    st.session_state.questions = 0

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

if "navigate_to_chat" not in st.session_state:
    st.session_state.navigate_to_chat = False


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("## ⚙️ Maintena AI")
    st.caption("INDUSTRIAL MAINTENANCE ASSISTANT")

    # Set navigation state BEFORE the radio widget is created.
    if st.session_state.navigate_to_chat:
        st.session_state.main_page = "Maintenance Chat"
        st.session_state.navigate_to_chat = False

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Maintenance Chat",
            "Manual Library",
            "About Project"
        ],
        key="main_page"
    )

    st.divider()

    uploads = st.file_uploader(
        "Upload PDF or TXT manuals",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    st.caption("Follow approved industrial safety procedures.")


# -------------------- PREPARE MANUAL LIBRARY --------------------

manual_parts = [("Built-in Maintenance Guide", MANUAL)]

if uploads:
    for file in uploads:
        try:
            content = read_file(file)

            if content.strip():
                manual_parts.append((file.name, content))

        except Exception as error:
            st.sidebar.error(f"Cannot read {file.name}: {error}")

all_chunks = []

for manual_name, manual_text in manual_parts:
    for chunk in split_text(manual_text):
        all_chunks.append(f"Manual: {manual_name}\n{chunk}")


# -------------------- HEADER --------------------

def hero():
    st.markdown("""
    <div class="hero">
        <p>SMARTER MAINTENANCE · SAFER OPERATIONS</p>
        <h1>Keep Your Operations Moving.</h1>
        <p>Find maintenance guidance, explore manuals, and get safety-focused answers.</p>
    </div>
    """, unsafe_allow_html=True)


# -------------------- OVERVIEW --------------------

if page == "Overview":
    hero()

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="tile">
            <h3>📚 Knowledge Base</h3>
            <p>Built-in guidance and uploaded manuals.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="tile">
            <h3>🤖 AI Assistant</h3>
            <p>Ask about equipment and maintenance issues.</p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown(f"""
        <div class="tile">
            <h3>💬 Questions</h3>
            <p>{st.session_state.questions} questions this session.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Try a sample question")

    samples = [
        "How should conveyor belt damage be handled?",
        "What can cause a motor to overheat?",
        "What should I do if a pump is leaking?",
        "What causes unusual machine vibration?"
    ]

    for sample in samples:
        if st.button(
            sample,
            key=f"sample_{sample}",
            use_container_width=True
        ):
            st.session_state.selected_question = sample
            st.session_state.navigate_to_chat = True
            st.rerun()


# -------------------- MAINTENANCE CHAT --------------------

elif page == "Maintenance Chat":
    hero()

    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown("### Ask Maintena AI")

    with col2:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View source sections"):
                    for i, (source, score) in enumerate(
                        message["sources"], 1
                    ):
                        st.markdown(
                            f"**Source {i} · relevance {score:.2f}**"
                        )
                        st.write(source[:1200])

    question = st.chat_input(
        "Ask about a machine or maintenance issue..."
    )

    if not question and st.session_state.selected_question:
        question = st.session_state.selected_question
        st.session_state.selected_question = None

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Checking maintenance guidance..."):
                    model, generator = load_models()

                    response, sources = answer_question(
                        question,
                        all_chunks,
                        model,
                        generator
                    )

                st.markdown(response)

                if sources:
                    with st.expander("View source sections"):
                        for i, (source, score) in enumerate(
                            sources, 1
                        ):
                            st.markdown(
                                f"**Source {i} · relevance {score:.2f}**"
                            )
                            st.write(source[:1200])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })

                st.session_state.questions += 1

            except Exception as error:
                st.error(
                    "The AI could not start. Check package installation "
                    "and internet access for the first model download."
                )
                st.caption(str(error))


# -------------------- MANUAL LIBRARY --------------------

elif page == "Manual Library":
    hero()
    st.markdown("### Available maintenance manuals")

    for name, content in manual_parts:
        with st.expander("📄 " + name):
            st.write(content[:12000])


# -------------------- ABOUT PROJECT --------------------

elif page == "About Project":
    hero()
    st.markdown("### About Maintena AI")

    st.write(
        "Maintena AI is an educational industrial maintenance assistant "
        "that helps users find guidance in maintenance manuals."
    )

    st.markdown("""
    **Features**
    - AI-powered maintenance chat
    - PDF and TXT manual uploads
    - Relevant source sections
    - Safety-focused maintenance guidance
    - Motor, conveyor, pump, and vibration guidance
    """)

    st.warning(
        "This application does not replace approved site procedures, "
        "manufacturer instructions, or qualified maintenance personnel. "
        "Never bypass guards or work on energized or moving equipment."
    )
