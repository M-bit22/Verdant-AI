"""Verdant AI: chat with your documents (Streamlit + LangChain + Gemini)."""

import base64

import csv

import hashlib

import html

import io

import json

import os

import tempfile
import time

import uuid

from datetime import datetime

from pathlib import Path

from urllib.parse import quote

import streamlit as st

import streamlit.components.v1 as components

from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

try:

    from dotenv import load_dotenv

    load_dotenv()

except ImportError:

    pass

st.set_page_config(

    page_title="Verdant AI",

    page_icon="🌿",

    layout="wide",

    initial_sidebar_state="expanded",

)

# ---------------------------------------------------------------------------

# Constants

# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent

HISTORY_DIR = APP_DIR / "verdant_history"

HISTORY_DIR.mkdir(exist_ok=True)

# Primary model first; the others are used automatically if quota runs out.

MODEL_ORDER = [

    "gemini-3.5-flash-lite",  # confirmed working with this API key

    "gemini-3.8-flash",

    "gemini-3.6-flash",

    "gemini-3.1-flash-lite",

]

EMBEDDING_MODEL = "models/gemini-embedding-001"

IMAGE_MODEL = "gemini-2.5-flash-image"  # change this one line if your key uses another image model

DOC_TYPES = ["pdf", "docx", "xlsx", "xlsm", "csv", "pptx"]

SWITCH_MARKERS = (

    "quota", "429", "resource_exhausted", "rate limit", "404", "not_found",

    "not found", "no longer available", "503", "unavailable", "overloaded",
    "timeout", "timed out", "deadline exceeded",

)

THEMES = {

    "Dark": {

        "bg": "#080b09", "sidebar": "#0b100d", "panel": "#0e1410", "panel2": "#121a15",

        "border": "#25372b", "text": "#edf5ef", "muted": "#91a296", "green": "#43d17d",

        "green_dark": "#143d27", "green_soft": "#a0f0bc",

    },

    "Light": {

        "bg": "#f4f8f4", "sidebar": "#eaf1eb", "panel": "#ffffff", "panel2": "#f0f6f1",

        "border": "#d4e2d6", "text": "#14251a", "muted": "#63786a", "green": "#178447",

        "green_dark": "#dff4e6", "green_soft": "#176b3c",

    },

}

def secret_key() -> str:

    try:

        value = st.secrets.get("GOOGLE_API_KEY", "")

    except Exception:

        value = ""

    return str(value or os.getenv("GOOGLE_API_KEY", "")).strip()

DEFAULTS = {

    "theme": "Dark",

    "api_key_input": secret_key(),

    "chat_model": MODEL_ORDER[0],

    "auto_switch": True,

    "max_output_tokens": 384,

    "temperature": 0.0,

    "chunk_size": 700,

    "chunk_overlap": 60,

    "top_k": 2,

    "save_history": True,

    "enable_image_generation": False,

    "view": "chat",

    "active_chat_id": None,

    "messages": [],

    "docs": [],

    "doc_source": None,

    "file_sig": None,

    "vectorstore": None,

    "index_fail": None,

    "uploader_nonce": 0,

    "pending_question": None,

    "gallery": [],

}

for _key, _value in DEFAULTS.items():

    st.session_state.setdefault(_key, _value)

def esc(value) -> str:

    return html.escape(str(value), quote=True)

# ---------------------------------------------------------------------------

# Chat history

# ---------------------------------------------------------------------------

def chat_file(chat_id: str) -> Path:

    return HISTORY_DIR / f"{chat_id}.json"

def load_history_index():

    items = []

    for path in HISTORY_DIR.glob("*.json"):

        try:

            data = json.loads(path.read_text(encoding="utf-8"))

        except (OSError, json.JSONDecodeError):

            continue

        items.append({

            "id": data.get("id", path.stem),

            "title": data.get("title", "Untitled conversation"),

            "updated": data.get("updated", ""),

        })

    return sorted(items, key=lambda item: item["updated"], reverse=True)

def ensure_chat_id():

    if not st.session_state.active_chat_id:

        st.session_state.active_chat_id = uuid.uuid4().hex[:12]

def save_chat():

    if not st.session_state.save_history or not st.session_state.messages:

        return

    ensure_chat_id()

    first = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), "")

    title = " ".join(first.split())[:60] or "Untitled conversation"

    record = {

        "id": st.session_state.active_chat_id,

        "title": title,

        "updated": datetime.now().isoformat(timespec="seconds"),

        "messages": st.session_state.messages,

        "docs": [

            {"name": name, "b64": base64.b64encode(data).decode("ascii")}

            for name, data in st.session_state.docs

        ],

    }

    chat_file(st.session_state.active_chat_id).write_text(

        json.dumps(record, ensure_ascii=False), encoding="utf-8"

    )

def open_chat(chat_id: str):

    try:

        record = json.loads(chat_file(chat_id).read_text(encoding="utf-8"))

    except (OSError, json.JSONDecodeError):

        return

    st.session_state.active_chat_id = chat_id

    st.session_state.messages = record.get("messages", [])

    st.session_state.docs = [(d["name"], base64.b64decode(d["b64"])) for d in record.get("docs", [])]

    st.session_state.doc_source = "restored"

    st.session_state.file_sig = None

    st.session_state.vectorstore = None

    st.session_state.index_fail = None

    st.session_state.uploader_nonce += 1  # empties the uploader so it doesn't override the restored files

    st.session_state.view = "chat"

def new_chat():

    st.session_state.update(

        active_chat_id=None, messages=[], docs=[], doc_source=None, file_sig=None,

        vectorstore=None, index_fail=None, view="chat", pending_question=None,

    )

    st.session_state.uploader_nonce += 1

def remove_docs():

    st.session_state.update(docs=[], doc_source=None, file_sig=None, vectorstore=None, index_fail=None)

    st.session_state.uploader_nonce += 1

def clear_chat():

    if st.session_state.active_chat_id:

        chat_file(st.session_state.active_chat_id).unlink(missing_ok=True)

    st.session_state.messages = []

def go_images():

    st.session_state.view = "images"

def go_chat():

    st.session_state.view = "chat"

def retry_index():

    st.session_state.index_fail = None

    st.session_state.file_sig = None

# ---------------------------------------------------------------------------

# Document extraction

# ---------------------------------------------------------------------------

def extract_documents(file_name: str, file_bytes: bytes):

    ext = Path(file_name).suffix.lower()

    docs: list[Document] = []

    if ext == ".pdf":

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:

            tmp.write(file_bytes)

            tmp_path = tmp.name

        try:

            docs = PyPDFLoader(tmp_path).load()

        finally:

            if os.path.exists(tmp_path):

                os.unlink(tmp_path)

        for doc in docs:

            page = doc.metadata.get("page")

            doc.metadata["source"] = file_name

            doc.metadata["page_label"] = f"p. {page + 1}" if isinstance(page, int) else "p. ?"

    elif ext == ".docx":

        try:

            from docx import Document as WordDocument

        except ImportError as exc:

            raise RuntimeError("Word support needs python-docx: pip install python-docx") from exc

        word = WordDocument(io.BytesIO(file_bytes))

        text = "\n".join(p.text for p in word.paragraphs if p.text.strip())

        for table in word.tables:

            text += "\n" + "\n".join(" | ".join(cell.text for cell in row.cells) for row in table.rows)

        if text.strip():

            docs = [Document(page_content=text, metadata={"source": file_name, "page_label": "Word document"})]

    elif ext in {".xlsx", ".xlsm"}:

        try:

            from openpyxl import load_workbook

        except ImportError as exc:

            raise RuntimeError("Excel support needs openpyxl: pip install openpyxl") from exc

        workbook = load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)

        for sheet in workbook.worksheets:

            rows = []

            for row in sheet.iter_rows(values_only=True):

                values = [str(v) if v is not None else "" for v in row]

                if any(values):

                    rows.append(" | ".join(values))

            if rows:

                docs.append(Document(

                    page_content=f"Sheet: {sheet.title}\n" + "\n".join(rows),

                    metadata={"source": file_name, "page_label": f"sheet {sheet.title}"},

                ))

        workbook.close()

    elif ext == ".csv":

        reader = csv.reader(io.StringIO(file_bytes.decode("utf-8-sig", errors="replace")))

        text = "\n".join(" | ".join(row) for row in reader if any(c.strip() for c in row))

        if text.strip():

            docs = [Document(page_content=text, metadata={"source": file_name, "page_label": "CSV data"})]

    elif ext == ".pptx":

        try:

            from pptx import Presentation

        except ImportError as exc:

            raise RuntimeError("PowerPoint support needs python-pptx: pip install python-pptx") from exc

        presentation = Presentation(io.BytesIO(file_bytes))

        for index, slide in enumerate(presentation.slides, start=1):

            bits = []

            for shape in slide.shapes:

                if getattr(shape, "has_text_frame", False) and shape.text.strip():

                    bits.append(shape.text.strip())

                if getattr(shape, "has_table", False):

                    for row in shape.table.rows:

                        bits.append(" | ".join(cell.text for cell in row.cells))

            if bits:

                docs.append(Document(

                    page_content=f"Slide {index}\n" + "\n".join(bits),

                    metadata={"source": file_name, "page_label": f"slide {index}"},

                ))

    else:

        raise ValueError(f"Unsupported file type: {ext}")

    if not any(doc.page_content.strip() for doc in docs):

        raise ValueError(f"No readable text was found in {file_name}. Scanned files need OCR first.")

    return docs

@st.cache_resource(show_spinner=False, max_entries=4)
def build_vectorstore(files: tuple, size: int, overlap: int, api_key: str) -> FAISS:
    """Build or reuse a local FAISS index, embedding in paced batches with 429 retries."""
    all_docs = []
    for file_name, file_bytes in files:
        all_docs.extend(extract_documents(file_name, file_bytes))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=min(overlap, max(0, size - 1)),
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    if not chunks:
        raise ValueError("No text chunks could be created from the uploaded files.")

    files_fingerprint = tuple(
        (name, hashlib.sha256(data).hexdigest()) for name, data in files
    )
    key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    index_fingerprint = hashlib.sha256(
        repr((files_fingerprint, size, overlap, EMBEDDING_MODEL, key_fingerprint)).encode("utf-8")
    ).hexdigest()
    index_dir = APP_DIR / ".verdant_indexes" / index_fingerprint
    index_file = index_dir / "index.faiss"
    metadata_file = index_dir / "index.pkl"

    if index_file.is_file() and metadata_file.is_file():
        try:
            return FAISS.load_local(
                str(index_dir),
                GoogleGenerativeAIEmbeddings(
                    model=EMBEDDING_MODEL, google_api_key=api_key, transport="rest"
                ),
                allow_dangerous_deserialization=True,
            )
        except Exception:
            pass  # Rebuild an incompatible or damaged local index.

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, google_api_key=api_key, transport="rest"
    )
    batch_size = 20
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    all_vectors = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        for attempt in range(4):
            try:
                vectors = embeddings.embed_documents(batch)
                all_vectors.extend(vectors)
                break
            except Exception as exc:
                error_text = str(exc).lower()
                is_rate_limit = any(
                    marker in error_text
                    for marker in ("429", "quota", "rate limit", "resource_exhausted", "too many requests")
                )
                if not is_rate_limit or attempt == 3:
                    raise
                wait_seconds = min(60, 30 * (attempt + 1))
                st.warning(
                    f"Embedding rate limit reached. Retrying this batch in {wait_seconds} seconds "
                    f"(attempt {attempt + 2}/4)."
                )
                time.sleep(wait_seconds)

        if start + batch_size < len(texts):
            time.sleep(1.2)

    if len(all_vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding returned {len(all_vectors)} vectors for {len(chunks)} chunks."
        )

    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, all_vectors)),
        embedding=embeddings,
        metadatas=metadatas,
    )
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    return vectorstore




def doc_signature(docs, size, overlap, api_key):

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

    return (

        tuple((name, hashlib.sha256(data).hexdigest()) for name, data in docs),

        size, overlap, key_hash,

    )

# ---------------------------------------------------------------------------

# Answering with automatic model switching

# ---------------------------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_messages([

    (

        "system",

        "You are Verdant, a precise document research assistant. Answer ONLY from the provided "

        "context. If the answer is not supported, say: 'I couldn't find that in the documents.' "

        "Cite the source in brackets, for example [report.pdf, p. 3]. Keep answers concise by "

        "default, use bullets when useful, and never invent facts or citations.\n\nCONTEXT:\n{context}",

    ),

    ("human", "{question}"),

])

def format_docs(docs) -> str:

    parts = []

    for doc in docs:

        label = doc.metadata.get("page_label", "?")

        source = doc.metadata.get("source", "Source")

        parts.append(f"[{source} · {label}]\n{doc.page_content[:900]}")

    return "\n\n".join(parts)

def chunk_text(content) -> str:

    """Stream chunks can be plain strings or lists of parts depending on the SDK version."""

    if isinstance(content, str):

        return content

    if isinstance(content, list):

        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)

    return ""

def should_switch_model(exc: Exception) -> bool:

    text = str(exc).lower()

    return any(marker in text for marker in SWITCH_MARKERS)

def stream_answer(question, docs, models, api_key, temperature, max_tokens, meta):

    """Yield answer text. If a model hits quota or is unavailable before it writes anything,

    the next model in the list takes over."""

    prompt_value = RAG_PROMPT.invoke({"context": format_docs(docs), "question": question})

    last_error: Exception | None = None

    for position, model in enumerate(models):

        emitted = False

        meta["model"] = model

        try:

            llm = ChatGoogleGenerativeAI(

                model=model,

                google_api_key=api_key,

                temperature=temperature,

                max_output_tokens=max_tokens,

                max_retries=1,   # fail fast so the model fallback runs instead of long internal retries

                timeout=60,

                transport="rest",  # REST is more reliable than gRPC on many Windows networks

            )

            for chunk in llm.stream(prompt_value):

                text = chunk_text(chunk.content)

                if text:

                    emitted = True

                    yield text

            return

        except Exception as exc:

            last_error = exc

            is_last = position == len(models) - 1

            if emitted or is_last or not should_switch_model(exc):

                raise

            meta.setdefault("switched_from", model)

    if last_error is not None:

        raise last_error

    raise RuntimeError("No Gemini model is available right now.")

def loader_html(title: str, detail: str) -> str:

    return (

        '<div class="loader-card">'

        '<div class="ring"></div>'

        '<div><div class="loader-title">' + esc(title) + '<span class="dots"></span></div>'

        '<div class="loader-detail">' + esc(detail) + '</div></div>'

        '</div>'

    )

def autoscroll():

    """Slide the page down so the newest message and loader stay in view."""

    components.html(

        """<script>

        (function () {

          try {

            const doc = window.parent.document;

            const scrollers = doc.querySelectorAll('[data-testid="stMain"], section.main, [data-testid="stAppViewContainer"]');

            setTimeout(() => scrollers.forEach(el => el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })), 150);

          } catch (e) {}

        })();

        </script>""",

        height=0,

    )

def run_question(question: str):

    """Answer one question inside the chat box and save the exchange."""

    ensure_chat_id()

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user", avatar="🧑"):

        st.markdown(question)

    models = [st.session_state.chat_model]

    if st.session_state.auto_switch:

        models += [m for m in MODEL_ORDER if m not in models]

    with st.chat_message("assistant", avatar="🌿"):

        loader = st.empty()

        answer_slot = st.empty()

        sources_slot = st.empty()

        loader.markdown(loader_html("Analyzing", "Loading your request…"), unsafe_allow_html=True)

        autoscroll()

        meta: dict = {}

        text = ""

        sources = []

        try:

            retriever = st.session_state.vectorstore.as_retriever(

                search_kwargs={"k": st.session_state.top_k}

            )

            docs = retriever.invoke(question)

            loader.markdown(loader_html("Analyzing", "Finding the most relevant passages…"), unsafe_allow_html=True)

            for piece in stream_answer(

                question, docs, models, st.session_state.api_key_input.strip(),

                0.0, 512, meta,

            ):

                if not text:

                    loader.empty()

                text += piece

                answer_slot.markdown(text + " ▌")

            text = text.strip() or "I couldn't find that in the documents."

            answer_slot.markdown(text)

            if meta.get("switched_from"):

                st.caption(f"{meta['switched_from']} hit its limit, so this answer came from {meta['model']}.")

            sources = [

                {

                    "file": doc.metadata.get("source", "Source"),

                    "page": doc.metadata.get("page_label", "?"),

                    "text": doc.page_content[:900],

                }

                for doc in docs

            ]

            if sources:

                with sources_slot.container():

                    with st.expander(f"Sources · {len(sources)} passages"):

                        for source in sources:

                            st.markdown(f"**{esc(source['file'])} · {esc(source['page'])}**")

                            st.write(source["text"])

                            st.divider()

        except Exception as exc:

            loader.empty()

            answer_slot.empty()

            st.session_state.messages.pop()  # drop the unanswered question

            st.error("I couldn't generate an answer. Check your API key, quota and model access.")

            with st.expander("Technical details"):

                st.code(str(exc))

            return

    loader.empty()

    st.session_state.messages.append({"role": "assistant", "content": text, "sources": sources})

    save_chat()

    autoscroll()

def share_text(messages, title) -> str:

    lines = [f"Verdant AI conversation: {title}", ""]

    for msg in messages:

        role = "You" if msg["role"] == "user" else "Verdant AI"

        lines.extend([f"{role}:", msg.get("content", ""), ""])

    return "\n".join(lines).strip()

# ---------------------------------------------------------------------------

# Image generation

# ---------------------------------------------------------------------------

def generate_illustration(prompt: str, api_key: str) -> bytes:

    from google import genai

    from google.genai import types

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(

        model=IMAGE_MODEL,

        contents=(

            "Create a clear, educational illustration that explains this concept. Polished editorial "

            "infographic style, legible hierarchy, no fake quotations, minimal text. Concept: " + prompt

        ),

        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),

    )

    for part in response.candidates[0].content.parts:

        if getattr(part, "inline_data", None) and part.inline_data.data:

            return part.inline_data.data

    raise RuntimeError("The image model returned no image for this request.")

# ---------------------------------------------------------------------------

# Sidebar: navigation, chat list, settings

# ---------------------------------------------------------------------------

with st.sidebar:

    st.markdown(

        '<div class="brand"><span class="brand-mark">✳</span><span>verdant<span class="dot-accent">.</span></span></div>',

        unsafe_allow_html=True,

    )

    st.button("＋  New chat", key="nav_new", on_click=new_chat, use_container_width=True)

    if st.session_state.view == "images":

        st.button("💬  Back to chat", key="nav_chat", on_click=go_chat, use_container_width=True)

    else:

        st.button("🖼  Generate images", key="nav_images", on_click=go_images, use_container_width=True)

    st.markdown('<div class="sidebar-section">Chats and tasks</div>', unsafe_allow_html=True)

    search_text = st.text_input(

        "Search chats", key="chat_search", label_visibility="collapsed", placeholder="Search chats…"

    )

    history = load_history_index() if st.session_state.save_history else []

    matches = [h for h in history if search_text.lower() in h["title"].lower()]

    if not st.session_state.save_history:

        st.caption("History is off. Turn it on in Settings to keep chats.")

    elif not history:

        st.caption("Your conversations will appear here.")

    for item in matches[:25]:

        label = item["title"] if len(item["title"]) <= 32 else item["title"][:31] + "…"

        st.button(

            label,

            key=f"open_{item['id']}",

            on_click=open_chat,

            args=(item["id"],),

            use_container_width=True,

            type="primary" if item["id"] == st.session_state.active_chat_id else "secondary",

        )

    st.divider()

    with st.expander("⚙  Settings", expanded=False):

        general_tab, model_tab, docs_tab, privacy_tab = st.tabs(

            ["General", "Model", "Documents", "Privacy"]

        )

        with general_tab:

            st.radio("Appearance", ["Dark", "Light"], horizontal=True, key="theme")

            st.caption("Choose the look of Verdant AI. This does not change answer generation.")

            st.divider()

            st.markdown("**Answer settings**")

            st.caption("Answer length is fixed at 512 tokens and creativity is fixed at 0.0 for concise, consistent, document-grounded replies.")

        with model_tab:

            st.markdown("**Current chat model**")

            st.code(st.session_state.chat_model, language=None)

            st.caption("This is the primary model selected for answers. Automatic switching can use another configured model if needed.")

            st.toggle(

                "Switch models when quota runs out",

                key="auto_switch",

                help="If the primary model hits its limit or is unavailable, the next configured Gemini model answers instead.",

            )

            st.markdown("**Embedding model**")

            st.code(EMBEDDING_MODEL, language=None)

            st.caption("Used to index uploaded documents. Embedding quota is separate from chat-model quota.")

        with docs_tab:

            st.markdown("**Document retrieval**")

            st.caption(f"Passages retrieved per question: {st.session_state.top_k}")

            st.caption(f"Chunk size: {st.session_state.chunk_size} characters")

            st.caption(f"Chunk overlap: {st.session_state.chunk_overlap} characters")

            st.info("These values are managed automatically to keep document indexing and retrieval consistent.")

        with privacy_tab:

            st.toggle("Save conversation history", key="save_history",

                      help="Chats and document copies are stored in the verdant_history folder on this machine.")

            st.toggle("Enable image generation", key="enable_image_generation")

            st.text_input("Gemini API key", type="password", key="api_key_input",

                          placeholder="Paste your key or set GOOGLE_API_KEY")

        st.caption("Settings apply immediately.")

    st.caption("🔐 Keys are never saved in chat history. Don't use shared computers for sensitive documents.")

# Read the settings used by the main area.

theme_name = st.session_state.theme

C = THEMES[theme_name]

api_key = st.session_state.api_key_input.strip()

chunk_size = st.session_state.chunk_size

chunk_overlap = st.session_state.chunk_overlap

THEME_VARS = (

    "<style>:root{"

    f"--bg:{C['bg']};--sidebar:{C['sidebar']};--panel:{C['panel']};--panel-2:{C['panel2']};"

    f"--border:{C['border']};--text:{C['text']};--muted:{C['muted']};--green:{C['green']};"

    f"--green-dark:{C['green_dark']};--green-soft:{C['green_soft']};"

    "}</style>"

)

STATIC_CSS = """

<style>

@import url('https\://fonts.googleapis.com/css2?family=DM+Sans:wght\@400;500;600;700&family=Manrope:wght\@400;500;600;700;800&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: radial-gradient(ellipse at 50% -10%, color-mix(in srgb, var(--green) 10%, transparent), transparent 40%), var(--bg); color: var(--text); }

[data-testid="stHeader"] { background: transparent; }

#MainMenu, footer { visibility: hidden; }

.block-container { max-width: 880px; padding-top: 1.2rem; padding-bottom: 6rem; }

h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.035em; color: var(--text); }

p, li, label { color: var(--text); }

/* sidebar */

[data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--border); }

[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }

[data-testid="stSidebar"] .stButton > button {

  justify-content: flex-start; text-align: left; border: none; background: transparent; color: var(--text);

  font-weight: 500; font-size: 14px; padding: .5rem .7rem; border-radius: 10px; box-shadow: none;

  transition: background .18s ease, transform .18s ease; }

[data-testid="stSidebar"] .stButton > button:hover { background: var(--panel); transform: translateX(2px); }

[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: var(--panel-2); border: none; color: var(--text); }

[data-testid="stSidebar"] .stButton > button p { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%; }

[data-testid="stSidebar"] [data-testid="stTextInput"] input { background: var(--panel) !important; border-radius: 10px !important; font-size: 13px; }

.brand { font-family: 'Manrope', sans-serif; font-size: 18px; font-weight: 800; letter-spacing: -.02em;

  display: flex; align-items: center; gap: 10px; margin: 2px 4px 14px 4px; color: var(--text); }

.brand-mark { width: 26px; height: 26px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center;

  background: var(--green); color: var(--bg); font-size: 15px; }

.dot-accent { color: var(--green); }

.sidebar-section { font-size: 12px; font-weight: 600; color: var(--muted); margin: 18px 6px 6px 6px; }

/* header + surfaces */

.eyebrow { text-transform: uppercase; letter-spacing: .16em; font-size: 11px; color: var(--green); font-weight: 700; margin-bottom: 6px; }

.hero-title { font-family: 'Manrope', sans-serif; font-size: 2.1rem; line-height: 1.1; letter-spacing: -.05em; font-weight: 800; margin: .1rem 0 .4rem 0; }

.hero-title span { color: var(--green); }

.hero-copy { color: var(--muted); font-size: .98rem; line-height: 1.65; }

.card-kicker { font-size: 11px; font-weight: 700; letter-spacing: .14em; color: var(--muted); margin-bottom: 4px; }

[data-testid="stVerticalBlockBorderWrapper"] > div > div { background: var(--panel); border-radius: 16px; }

[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--border) !important; border-radius: 16px; transition: border-color .25s ease, box-shadow .25s ease; }

[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: color-mix(in srgb, var(--green) 45%, var(--border)) !important; }

.chip { display: inline-block; background: var(--panel-2); border: 1px solid var(--border); padding: 4px 10px; border-radius: 999px;

  font-size: 12px; margin: 3px 6px 3px 0; animation: pop-in .35s ease both; }

.pill { display: inline-flex; align-items: center; gap: 7px; font-size: 12px; font-weight: 600; color: var(--green-soft);

  background: var(--green-dark); border-radius: 999px; padding: 4px 11px; }

.pill-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); animation: pulse-dot 1.6s infinite; }

/* chat */

[data-testid="stChatMessage"] { background: transparent; border: none; padding: 4px 0; animation: slide-up .35s ease both; }

[data-testid="stChatMessage"] p { font-size: 15px; line-height: 1.7; }

[data-testid="stChatInput"] { border: 1px solid color-mix(in srgb, var(--green) 45%, var(--border)); background: var(--panel);

  border-radius: 18px; animation: input-aura 4s ease-in-out infinite; transition: border-color .25s ease, transform .25s ease; }

[data-testid="stChatInput"]:focus-within { border-color: var(--green); transform: translateY(-1px); }

[data-testid="stChatInput"] textarea { background: transparent !important; }

[data-testid="stChatInput"] button { background: var(--green) !important; color: var(--bg) !important; border-radius: 10px !important; }

/* premium loader */

.loader-card { display: flex; align-items: center; gap: 14px; padding: 15px 18px; border-radius: 16px; background: var(--panel);

  border: 1px solid color-mix(in srgb, var(--green) 40%, var(--border)); animation: aura-pulse 2.2s ease-in-out infinite,

  slide-up .3s ease both; }

.ring { width: 26px; height: 26px; border-radius: 50%; flex-shrink: 0; border: 3px solid var(--border);

  border-top-color: var(--green); border-right-color: var(--green); animation: spin .8s linear infinite; }

.loader-title { font-weight: 700; font-size: 15px; background: linear-gradient(90deg, var(--text) 0%, var(--green) 50%, var(--text) 100%);

  background-size: 200% auto; -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;

  animation: shimmer 2s linear infinite; }

.loader-detail { font-size: 12px; color: var(--muted); margin-top: 3px; }

.dots::after { content: ""; animation: dots 1.4s steps(4, end) infinite; }

/* buttons, inputs, popovers */

.stButton > button, .stDownloadButton > button, [data-testid="stLinkButton"] a { border-radius: 11px; border: 1px solid var(--border);

  background: var(--panel-2); color: var(--text); font-weight: 600; transition: transform .18s ease, border-color .2s ease, box-shadow .2s ease; }

.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stLinkButton"] a:hover { border-color: var(--green); transform: translateY(-2px); }

.stButton > button:active { transform: translateY(0) scale(.98); }

.stButton > button[kind="primary"] { background: var(--green); color: var(--bg); border: 1px solid var(--green); }

[data-testid="stFileUploader"] { background: var(--panel-2); border: 1px dashed color-mix(in srgb, var(--green) 50%, var(--border)); border-radius: 14px; padding: .6rem; }

[data-testid="stFileUploader"] button { background: var(--green-dark) !important; color: var(--text) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }

[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div { background: var(--panel) !important; border-color: var(--border) !important; border-radius: 10px !important; }

[data-baseweb="tab-highlight"] { background: var(--green) !important; }

[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 13px; background: var(--panel); }

[data-testid="stPopoverBody"] { background: var(--panel) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; }

hr { border-color: var(--border); }

/* motion */

@keyframes spin { to { transform: rotate(360deg); } }

@keyframes shimmer { to { background-position: 200% center; } }

@keyframes dots { 0% { content: ""; } 25% { content: "."; } 50% { content: ".."; } 75% { content: "..."; } }

@keyframes slide-up { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }

@keyframes pop-in { from { opacity: 0; transform: scale(.96); } to { opacity: 1; transform: scale(1); } }

@keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: .35; } }

@keyframes aura-pulse { 0%, 100% { box-shadow: 0 0 0 1px color-mix(in srgb, var(--green) 8%, transparent), 0 0 18px color-mix(in srgb, var(--green) 10%, transparent); }

  50% { box-shadow: 0 0 0 1px color-mix(in srgb, var(--green) 20%, transparent), 0 0 32px color-mix(in srgb, var(--green) 20%, transparent); } }

@keyframes input-aura { 0%, 100% { box-shadow: 0 0 14px color-mix(in srgb, var(--green) 8%, transparent); }

  50% { box-shadow: 0 0 26px color-mix(in srgb, var(--green) 16%, transparent); } }

@media (prefers-reduced-motion: reduce) {

  *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; }

}

@media (max-width: 760px) { .hero-title { font-size: 1.6rem; } }

</style>

"""

st.markdown(THEME_VARS + STATIC_CSS, unsafe_allow_html=True)

if not api_key:

    st.markdown('<div class="eyebrow">AI DOCUMENT WORKSPACE</div>', unsafe_allow_html=True)

    st.markdown('<div class="hero-title">Your documents have<br/>more to <span>say.</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="hero-copy">Add your Gemini API key in Settings → Privacy to begin.</div>', unsafe_allow_html=True)

    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# ---------------------------------------------------------------------------

# Image studio view

# ---------------------------------------------------------------------------

if st.session_state.view == "images":

    st.markdown('<div class="eyebrow">IMAGE STUDIO</div>', unsafe_allow_html=True)

    st.markdown('<div class="hero-title">Turn an idea into a <span>picture.</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="hero-copy">Describe a concept and get an educational illustration.</div>', unsafe_allow_html=True)

    prompt = st.text_area(

        "Describe the image", key="img_prompt", height=110,

        placeholder="e.g. An infographic showing the stages of the water cycle",

    )

    if not st.session_state.enable_image_generation:

        st.info("Image generation is off. Turn it on in Settings → Privacy.")

    if st.button("✳  Generate image", type="primary",

                 disabled=not (prompt.strip() and st.session_state.enable_image_generation)):

        slot = st.empty()

        slot.markdown(loader_html("Creating your illustration", "This can take a moment…"), unsafe_allow_html=True)

        try:

            data = generate_illustration(prompt.strip(), api_key)

            st.session_state.gallery = ([{"bytes": data, "prompt": prompt.strip()}] + st.session_state.gallery)[:12]

            slot.empty()

        except Exception as exc:

            slot.empty()

            st.error("Image generation failed. Your key may not have access to this image model.")

            with st.expander("Technical details"):

                st.code(str(exc))

    if st.session_state.gallery:

        columns = st.columns(3, gap="medium")

        for index, item in enumerate(st.session_state.gallery):

            with columns[index % 3]:

                st.image(item["bytes"], caption=item["prompt"], use_container_width=True)

                st.download_button("↓ Download", data=item["bytes"], file_name=f"verdant_image_{index + 1}.png",

                                   mime="image/png", key=f"dl_img_{index}", use_container_width=True)

    st.stop()

# ---------------------------------------------------------------------------

# Chat view: header

# ---------------------------------------------------------------------------

head_left, head_right = st.columns([5, 1])

with head_left:

    st.markdown('<div class="eyebrow">YOUR AI RESEARCH DESK</div>', unsafe_allow_html=True)

    st.markdown('<div class="hero-title">Your AI <span>workspace</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="hero-copy">Ask questions across your uploaded documents.</div>', unsafe_allow_html=True)

with head_right:

    with st.popover("⋯  Actions"):

        names = [name for name, _ in st.session_state.docs]

        title = names[0] if len(names) == 1 else (f"{len(names)} documents" if names else "Chat")

        share = share_text(st.session_state.messages, title)

        st.download_button("↓ Export chat", data=share, file_name="verdant_conversation.txt", mime="text/plain",

                           key="export_chat", use_container_width=True, disabled=not st.session_state.messages)

        st.link_button("WhatsApp ↗", "https\://wa.me/?text=" + quote(share[:4000]), use_container_width=True)

        st.link_button("Telegram ↗", "https\://t.me/share/url?url=&text=" + quote(share[:4000]), use_container_width=True)

        components.html(

            f"""<button id="copyBtn" style="width:100%;height:40px;border-radius:11px;cursor:pointer;

            border:1px solid {C['border']};background:{C['panel2']};color:{C['text']};font-weight:600;

            font-family:'DM Sans',sans-serif;">Copy conversation</button>

            <script>

            const shareText = {json.dumps(share[:12000]).replace('</', '<\\\\\\/')};

            const btn = document.getElementById('copyBtn');

            btn.addEventListener('click', async () => {{

              try {{ await navigator.clipboard.writeText(shareText); }}

              catch (e) {{

                const area = document.createElement('textarea');

                area.value = shareText; document.body.appendChild(area);

                area.select(); document.execCommand('copy'); area.remove();

              }}

              btn.innerText = 'Copied ✓';

              setTimeout(() => btn.innerText = 'Copy conversation', 1800);

            }});

            </script>""",

            height=46,

        )

        st.button("Clear chat", key="clear_chat_btn", on_click=clear_chat, use_container_width=True,

                  disabled=not st.session_state.messages)

# ---------------------------------------------------------------------------

# Documents card (uploader accepts PDF, Word, Excel, CSV, PowerPoint)

# ---------------------------------------------------------------------------

with st.container(border=True):

    st.markdown('<div class="card-kicker">DOCUMENTS</div>', unsafe_allow_html=True)

    uploads = st.file_uploader(

        "Upload files",

        type=DOC_TYPES,

        accept_multiple_files=True,

        key=f"uploader_{st.session_state.uploader_nonce}",

        help="PDF, Word (.docx), Excel (.xlsx/.xlsm), CSV and PowerPoint (.pptx). Several files can be added together.",

    )

    if uploads:

        st.session_state.docs = [(f.name, f.getvalue()) for f in uploads]

        st.session_state.doc_source = "upload"

    status_slot = st.empty()

    if st.session_state.docs:

        st.markdown("".join(f'<span class="chip">{esc(n)}</span>' for n, _ in st.session_state.docs),

                    unsafe_allow_html=True)

        st.button("Remove all documents", key="remove_docs", on_click=remove_docs)

    else:

        st.caption("PDF · Word · Excel · CSV · PowerPoint")

# ---------------------------------------------------------------------------

# Index documents when they or the chunk settings change

# ---------------------------------------------------------------------------

if st.session_state.docs:

    sig = doc_signature(st.session_state.docs, chunk_size, chunk_overlap, api_key)

    if st.session_state.file_sig != sig:

        if st.session_state.index_fail and st.session_state.index_fail[0] == sig:

            status_slot.error("I couldn't read one or more files. Check the file format and your API access.")

            with status_slot.container():

                with st.expander("Technical details"):

                    st.code(st.session_state.index_fail[1])

                st.button("Try again", key="retry_index", on_click=retry_index)

        else:

            status_slot.markdown(

                loader_html("Preparing your documents", f"Extracting and indexing {len(st.session_state.docs)} file(s)…"),

                unsafe_allow_html=True,

            )

            try:

                st.session_state.vectorstore = build_vectorstore(

                    tuple(st.session_state.docs), chunk_size, min(chunk_overlap, chunk_size - 1), api_key,

                )

                st.session_state.file_sig = sig

                st.session_state.index_fail = None

                status_slot.markdown('<span class="pill"><span class="pill-dot"></span>Ready to answer</span>',

                                     unsafe_allow_html=True)

            except Exception as exc:

                st.session_state.vectorstore = None

                st.session_state.index_fail = (sig, str(exc))

                status_slot.error("I couldn't read one or more files.")

                with status_slot.container():

                    with st.expander("Technical details"):

                        st.code(str(exc))

                    st.button("Try again", key="retry_index_now", on_click=retry_index)

    else:

        status_slot.markdown('<span class="pill"><span class="pill-dot"></span>Ready to answer</span>',

                             unsafe_allow_html=True)

ready = st.session_state.vectorstore is not None

# ---------------------------------------------------------------------------

# Chat messages (centered)

# ---------------------------------------------------------------------------

chat_box = st.container()

with chat_box:

    if not st.session_state.messages:

        if not ready:

            st.markdown('<div class="card-kicker" style="margin-top:14px">GET STARTED</div>', unsafe_allow_html=True)

            st.markdown("Upload one or more documents above. Once they are indexed, ask your first question here.")

        else:

            st.markdown('<div class="card-kicker" style="margin-top:14px">SUGGESTIONS</div>', unsafe_allow_html=True)

            for index, suggestion in enumerate([

                "Summarize these documents in 5 bullets",

                "What are the most important findings?",

                "Explain the main conclusion simply",

            ]):

                st.button(suggestion, key=f"sugg_{index}", use_container_width=True,

                          on_click=lambda s=suggestion: st.session_state.update(pending_question=s))

    for message in st.session_state.messages:

        avatar = "🧑" if message["role"] == "user" else "🌿"

        with st.chat_message(message["role"], avatar=avatar):

            st.markdown(message.get("content", ""))

            if message["role"] == "assistant" and message.get("sources"):

                with st.expander(f"Sources · {len(message['sources'])} passages"):

                    for source in message["sources"]:

                        st.markdown(f"**{esc(source['file'])} · {esc(source['page'])}**")

                        st.write(source["text"])

                        st.divider()

# ---------------------------------------------------------------------------

# Input pinned to the bottom

# ---------------------------------------------------------------------------

question = st.chat_input("Ask anything about your documents…", disabled=not ready)

if not question:

    question = st.session_state.pop("pending_question", None)

if question and ready:

    with chat_box:

        run_question(question)