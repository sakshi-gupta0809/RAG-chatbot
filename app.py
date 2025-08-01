
# import asyncio
# import sys
# import os
# from datetime import datetime
# import logging
# import streamlit as st
# from dotenv import load_dotenv
# from together import Together
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain_community.document_loaders import PyPDFLoader

# # Fix async loop issue on Windows
# if sys.platform.startswith('win'):
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# # Load API keys
# load_dotenv()
# TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # Logging setup
# logging.basicConfig(
#     level=logging.INFO,
#     filename="app.log",
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )

# def show_chat_messages():
#     for user_msg, bot_msg, model, timestamp in st.session_state.conversation_history:
#         with st.chat_message("user"):
#             st.markdown(f"**You** ({timestamp}): {user_msg}")
#         with st.chat_message("assistant"):
#             st.markdown(f"**{model}**: {bot_msg}")

# def fetch_together_chat(prompt_messages):
#     client = Together().chat.completions.create(
#         model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
#         messages=prompt_messages
#     )
#     return client.choices[0].message.content

# def ensure_async_loop():
#     try:
#         asyncio.get_event_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)

# def embed_and_store(chunks):
#     ensure_async_loop()
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
#     return FAISS.from_texts(chunks, embeddings)

# def handle_user_query(user_question):
#     with st.spinner("Thinking..."):
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         if st.session_state.vector_store:
#             docs = st.session_state.vector_store.similarity_search(user_question, k=3)
#             context_texts = [doc.page_content for doc in docs]

#             # Combine context into prompt
#             prompt = [{"role": "system", "content": "Use the context to answer carefully."}]
#             for ctxt in context_texts:
#                 prompt.append({"role": "system", "content": ctxt})
#             prompt.append({"role": "user", "content": user_question})

#             answer = fetch_together_chat(prompt)
#             source = "Llama 3.3‑RAG"
#         else:
#             prompt = [{"role": "user", "content": user_question}]
#             answer = fetch_together_chat(prompt)
#             source = "Llama 3.3‑Chat"

#         st.session_state.conversation_history.append((user_question, answer, source, timestamp))
#         show_chat_messages()


# def process_pdf(pdf_file):
#     path = "temp.pdf"
#     with open(path, "wb") as f:
#         f.write(pdf_file.read())
#     pages = PyPDFLoader(path).load()
#     os.remove(path)
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     return splitter.split_documents(pages)

# def main():
#     st.set_page_config(page_title="Llama 3.3 Chatbot", layout="wide")
#     st.title("🤖 Llama 3.3‑70B Instruct Turbo Chatbot")
#     st.caption("General chat or PDF‑powered RAG using Llama 3.3 Free model")

#     if "conversation_history" not in st.session_state:
#         st.session_state.conversation_history = []
#     if "vector_store" not in st.session_state:
#         st.session_state.vector_store = None

#     with st.sidebar:
#         st.header("Optional PDF RAG")
#         pdf_file = st.file_uploader("Upload a PDF for RAG", type=["pdf"])
#         if pdf_file:
#             if st.button("Process PDF"):
#                 with st.spinner("Indexing PDF..."):
#                     docs = process_pdf(pdf_file)
#                     chunks = [doc.page_content for doc in docs]
#                     st.session_state.vector_store = embed_and_store(chunks)
#                     st.success("✅ PDF processed and indexed")

#     show_chat_messages()
#     user_question = st.chat_input("Ask a question...")

#     if user_question:
#         handle_user_query(user_question)

#     if st.button("Clear Chat"):
#         st.session_state.conversation_history = []
#         st.session_state.vector_store = None
#         st.rerun()

# if __name__ == "__main__":
#     main()








# import asyncio
# import sys
# import os
# import json
# import pickle
# import uuid
# from datetime import datetime
# import logging
# import streamlit as st
# from pathlib import Path
# from dotenv import load_dotenv
# from together import Together
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader

# # Fix async loop issue on Windows
# if sys.platform.startswith('win'):
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# # Load API keys
# load_dotenv()
# TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # Logging setup
# logging.basicConfig(
#     level=logging.INFO,
#     filename="app.log",
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )

# CHAT_DIR = Path("chat_history")
# CHAT_DIR.mkdir(exist_ok=True)

# # ================== Custom Styling & Animations ==================
# def inject_css_and_animation():
#     dark_mode = st.session_state.get("dark_mode", False)
#     background_color = "#1e1e1e" if dark_mode else "#f5f7fa"
#     user_bubble_color = "#343541" if dark_mode else "#d1e7dd"
#     bot_bubble_color = "#2c2f33" if dark_mode else "#ffffff"
#     bot_border_color = "#444" if dark_mode else "#dee2e6"
#     text_color = "#f0f0f0" if dark_mode else "#000000"

#     st.markdown(f"""
#     <style>
#         .stApp {{
#             background-color: {background_color};
#             font-family: 'Segoe UI', sans-serif;
#             color: {text_color};
#         }}
#         .chat-container {{
#             display: flex;
#             flex-direction: column;
#             gap: 10px;
#         }}
#         .chat-bubble {{
#             max-width: 80%;
#             padding: 12px 20px;
#             border-radius: 18px;
#             margin-bottom: 10px;
#             font-size: 15px;
#             line-height: 1.5;
#             animation: slideFadeIn 0.5s ease-in-out;
#         }}
#         .user {{
#             background-color: {user_bubble_color};
#             align-self: flex-end;
#             color: #000;
#         }}
#         .bot {{
#             background-color: {bot_bubble_color};
#             border: 1px solid {bot_border_color};
#             align-self: flex-start;
#         }}
#         @keyframes slideFadeIn {{
#             0% {{ opacity: 0; transform: translateY(15px); }}
#             100% {{ opacity: 1; transform: translateY(0); }}
#         }}
#         .timestamp {{
#             font-size: 11px;
#             color: #6c757d;
#             margin-top: -8px;
#             margin-bottom: 6px;
#         }}
#     </style>
#     """, unsafe_allow_html=True)

# # ================== Chat Functions ==================
# def show_chat_messages():
#     st.markdown('<div class="chat-container">', unsafe_allow_html=True)
#     for user_msg, bot_msg, model, timestamp in st.session_state.conversation_history:
#         st.markdown(f'<div class="chat-bubble user">{user_msg}</div>', unsafe_allow_html=True)
#         st.markdown(f'<div class="timestamp">🕒 {timestamp}</div>', unsafe_allow_html=True)
#         st.markdown(f'<div class="chat-bubble bot"><strong>{model}</strong>: {bot_msg}</div>', unsafe_allow_html=True)
#     st.markdown('</div>', unsafe_allow_html=True)

# def fetch_together_chat(prompt_messages):
#     client = Together().chat.completions.create(
#         model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
#         messages=prompt_messages
#     )
#     return client.choices[0].message.content

# def ensure_async_loop():
#     try:
#         asyncio.get_event_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)

# def embed_and_store(chunks):
#     ensure_async_loop()
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
#     return FAISS.from_texts(chunks, embeddings)

# def handle_user_query(user_question):
#     with st.spinner("🤔 Thinking..."):
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         if st.session_state.vector_store:
#             docs = st.session_state.vector_store.similarity_search(user_question, k=5)
#             context_texts = [doc.page_content for doc in docs]

#             if "similarity between all" in user_question.lower():
#                 merged_context = "\n".join(context_texts)
#                 prompt = [
#                     {"role": "system", "content": "You are an expert at comparing documents. Analyze the combined content and describe common patterns, topics, or repeated statements."},
#                     {"role": "system", "content": merged_context},
#                     {"role": "user", "content": user_question}
#                 ]
#             else:
#                 prompt = [{"role": "system", "content": "Use the context to answer carefully."}]
#                 for ctxt in context_texts:
#                     prompt.append({"role": "system", "content": ctxt})
#                 prompt.append({"role": "user", "content": user_question})

#             answer = fetch_together_chat(prompt)
#             source = "📘 Llama 3.3‑RAG"
#         else:
#             prompt = [{"role": "user", "content": user_question}]
#             answer = fetch_together_chat(prompt)
#             source = "🧠 Llama 3.3‑Chat"

#         st.session_state.conversation_history.append((user_question, answer, source, timestamp))
#         show_chat_messages()

# def process_multiple_pdfs(pdf_files):
#     all_chunks = []
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     for pdf_file in pdf_files:
#         path = f"temp_{pdf_file.name}"
#         with open(path, "wb") as f:
#             f.write(pdf_file.read())
#         pages = PyPDFLoader(path).load()
#         os.remove(path)
#         chunks = splitter.split_documents(pages)
#         all_chunks.extend(chunks)
#     return all_chunks

# # =============== Chat Save/Load Functions ===============
# def save_chat():
#     if st.session_state.conversation_history:
#         first_question = st.session_state.conversation_history[0][0].strip()
#         title_words = first_question.split()[:6]
#         title = "_".join(title_words).lower() if title_words else "chat"
#         chat_id = f"{title}.pkl"
#         with open(CHAT_DIR / chat_id, "wb") as f:
#             pickle.dump(st.session_state.conversation_history, f)
#         st.success(f"💾 Chat saved as '{chat_id}'")

# def list_chat_files():
#     return sorted(CHAT_DIR.glob("*.pkl"))

# def load_chat(chat_file):
#     with open(chat_file, "rb") as f:
#         st.session_state.conversation_history = pickle.load(f)
#     st.success(f"✅ Loaded chat: {chat_file.stem}")

# def delete_chat(chat_file):
#     os.remove(chat_file)
#     st.success(f"🗑️ Deleted chat: {chat_file.stem}")

# # ================== Main ==================
# def main():
#     st.set_page_config(page_title="💬 Llama 3.3 Chatbot", layout="wide")

#     if "conversation_history" not in st.session_state:
#         st.session_state.conversation_history = []
#     if "vector_store" not in st.session_state:
#         st.session_state.vector_store = None
#     if "dark_mode" not in st.session_state:
#         st.session_state.dark_mode = False

#     inject_css_and_animation()

#     st.title("🤖 Llama 3.3‑70B Instruct Turbo Chatbot")
#     st.caption("Chat freely or use PDF-powered RAG. Save and revisit your chats anytime!")

#     with st.sidebar:
#         if st.button("🌓 Toggle Dark/Light Mode"):
#             st.session_state.dark_mode = not st.session_state.dark_mode
#             st.rerun()

#         st.header("📄 Upload Multiple PDFs")
#         pdf_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
#         if pdf_files:
#             if st.button("⚙️ Process PDFs"):
#                 with st.spinner("🔍 Indexing PDFs..."):
#                     docs = process_multiple_pdfs(pdf_files)
#                     chunks = [doc.page_content for doc in docs]
#                     st.session_state.vector_store = embed_and_store(chunks)
#                     st.success("✅ PDFs processed and indexed")

#         st.divider()

#         st.header("💾 Chat History")
#         if st.button("💬 Save Chat"):
#             save_chat()

#         chat_files = list_chat_files()
#         selected_chat = st.selectbox("📂 Load Previous Conversation", ["-- Select --"] + [f.stem for f in chat_files])
#         if selected_chat != "-- Select --":
#             chat_path = CHAT_DIR / f"{selected_chat}.pkl"
#             if st.button("🗂 Load Selected Chat"):
#                 load_chat(chat_path)
#             if st.button("🗑 Delete Selected Chat"):
#                 delete_chat(chat_path)
#                 st.rerun()

#         new_chat_btn = st.button("🆕 New Chat")
#         if new_chat_btn:
#             st.session_state.conversation_history = []
#             st.session_state.vector_store = None
#             st.rerun()

#         st.divider()
#         if st.button("🧹 Clear Chat"):
#             st.session_state.conversation_history = []
#             st.session_state.vector_store = None
#             st.rerun()

#     show_chat_messages()

#     user_question = st.chat_input("Type your question here...")
#     if user_question:
#         handle_user_query(user_question)

# if __name__ == "__main__":
#     main()


# import asyncio
# import sys
# import os
# import json
# import pickle
# import uuid
# import shutil
# from datetime import datetime
# import logging
# import streamlit as st
# from pathlib import Path
# from dotenv import load_dotenv
# from together import Together
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader

# # Fix async loop issue on Windows
# if sys.platform.startswith('win'):
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# # Load API keys
# load_dotenv()
# TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # Logging setup
# logging.basicConfig(
#     level=logging.INFO,
#     filename="app.log",
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )

# CHAT_DIR = Path("chat_history")
# CHAT_DIR.mkdir(exist_ok=True)

# # ================== Styling ==================
# def inject_css_and_animation():
#     st.markdown("""
#     <style>
#         .stApp {
#             transition: background-color 0.3s ease;
#         }
#         .chat-container {
#             display: flex;
#             flex-direction: column;
#             gap: 10px;
#         }
#         .chat-bubble {
#             max-width: 80%;
#             padding: 12px 20px;
#             border-radius: 18px;
#             margin-bottom: 10px;
#             font-size: 15px;
#             line-height: 1.5;
#             animation: slideFadeIn 0.4s ease-in-out;
#         }
#         .user {
#             background-color: #d1e7dd;
#             align-self: flex-end;
#         }
#         .bot {
#             background-color: #ffffff;
#             border: 1px solid #dee2e6;
#             align-self: flex-start;
#         }
#         .dark .user {
#             background-color: #455a64;
#             color: white;
#         }
#         .dark .bot {
#             background-color: #263238;
#             color: white;
#         }
#         @keyframes slideFadeIn {
#             0% {opacity: 0; transform: translateY(10px);}
#             100% {opacity: 1; transform: translateY(0);}
#         }
#         .timestamp {
#             font-size: 11px;
#             color: #6c757d;
#             margin-top: -8px;
#             margin-bottom: 6px;
#         }
#     </style>
#     """, unsafe_allow_html=True)

# # ================== Chat ==================
# def show_chat_messages():
#     mode = st.session_state.get("theme", "light")
#     css_class = "dark" if mode == "dark" else ""
#     st.markdown(f'<div class="chat-container {css_class}">', unsafe_allow_html=True)
#     for user_msg, bot_msg, model, timestamp in st.session_state.conversation_history:
#         st.markdown(f'<div class="chat-bubble user">{user_msg}</div>', unsafe_allow_html=True)
#         st.markdown(f'<div class="timestamp">🕒 {timestamp}</div>', unsafe_allow_html=True)
#         st.markdown(f'<div class="chat-bubble bot"><strong>{model}</strong>: {bot_msg}</div>', unsafe_allow_html=True)
#     st.markdown('</div>', unsafe_allow_html=True)

# def fetch_together_chat(prompt_messages):
#     client = Together().chat.completions.create(
#         model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
#         messages=prompt_messages
#     )
#     return client.choices[0].message.content

# def ensure_async_loop():
#     try:
#         asyncio.get_event_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)

# def embed_and_store(chunks):
#     ensure_async_loop()
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
#     return FAISS.from_texts(chunks, embeddings)

# def handle_user_query(user_question):
#     with st.spinner("🤔 Thinking..."):
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         if st.session_state.vector_store:
#             docs = st.session_state.vector_store.similarity_search(user_question, k=3)
#             context_texts = [doc.page_content for doc in docs]
#             prompt = [{"role": "system", "content": "Use the context to answer carefully."}]
#             for ctxt in context_texts:
#                 prompt.append({"role": "system", "content": ctxt})
#             prompt.append({"role": "user", "content": user_question})
#             answer = fetch_together_chat(prompt)
#             source = "📘 Llama 3.3‑RAG"
#         else:
#             prompt = [{"role": "user", "content": user_question}]
#             answer = fetch_together_chat(prompt)
#             source = "🧠 Llama 3.3‑Chat"
#         st.session_state.conversation_history.append((user_question, answer, source, timestamp))
#         auto_save_chat()
#         show_chat_messages()

# def process_multiple_pdfs(pdf_files):
#     all_chunks = []
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     for pdf_file in pdf_files:
#         path = f"temp_{pdf_file.name}"
#         with open(path, "wb") as f:
#             f.write(pdf_file.read())
#         pages = PyPDFLoader(path).load()
#         os.remove(path)
#         chunks = splitter.split_documents(pages)
#         all_chunks.extend(chunks)
#     return all_chunks

# # =============== Chat Save/Load ===============
# def generate_chat_title():
#     if not st.session_state.conversation_history:
#         return "chat"
#     first_user_msg = st.session_state.conversation_history[0][0][:30].replace(" ", "_").strip("?.,:")
#     return f"{first_user_msg}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# def auto_save_chat():
#     title = st.session_state.get("current_chat_title")
#     if not title:
#         title = generate_chat_title()
#         st.session_state.current_chat_title = title
#     with open(CHAT_DIR / f"{title}.pkl", "wb") as f:
#         pickle.dump(st.session_state.conversation_history, f)

# def list_chat_files():
#     return sorted(CHAT_DIR.glob("*.pkl"))

# def load_chat(chat_file):
#     with open(chat_file, "rb") as f:
#         st.session_state.conversation_history = pickle.load(f)
#     st.session_state.current_chat_title = chat_file.stem
#     st.success(f"✅ Loaded chat: {chat_file.stem}")

# def rename_chat(old_name, new_name):
#     old_path = CHAT_DIR / f"{old_name}.pkl"
#     new_path = CHAT_DIR / f"{new_name}.pkl"
#     if new_path.exists():
#         st.warning("⚠️ File with this name already exists.")
#         return
#     shutil.move(old_path, new_path)
#     st.session_state.current_chat_title = new_name
#     st.success("✅ Renamed chat.")

# def delete_chat(filename):
#     os.remove(CHAT_DIR / f"{filename}.pkl")
#     st.success("🗑️ Chat deleted.")

# # ================== Main ==================
# def main():
#     st.set_page_config(page_title="💬 Llama 3.3 Chatbot", layout="wide")
#     inject_css_and_animation()

#     st.title("🤖 Llama 3.3‑70B Instruct Turbo Chatbot")
#     st.caption("Chat freely or use PDF-powered RAG. Automatically save and revisit chats.")

#     if "conversation_history" not in st.session_state:
#         st.session_state.conversation_history = []
#     if "vector_store" not in st.session_state:
#         st.session_state.vector_store = None
#     if "theme" not in st.session_state:
#         st.session_state.theme = "light"
#     if "current_chat_title" not in st.session_state:
#         st.session_state.current_chat_title = None

#     with st.sidebar:
#         st.header("📄 PDF Upload")
#         pdf_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
#         if pdf_files and st.button("⚙️ Process PDFs"):
#             with st.spinner("🔍 Indexing PDFs..."):
#                 docs = process_multiple_pdfs(pdf_files)
#                 chunks = [doc.page_content for doc in docs]
#                 st.session_state.vector_store = embed_and_store(chunks)
#                 st.success("✅ PDFs indexed")

#         st.divider()

#         st.header("💾 Chat History")
#         chat_options = list_chat_files()
#         chat_names = [f.stem for f in chat_options]
#         if chat_names:
#             selected = st.selectbox("📂 Your Chats", chat_names, index=0, key="chat_select")
#             if selected and st.session_state.current_chat_title != selected:
#                 load_chat(CHAT_DIR / f"{selected}.pkl")

#             col1, col2 = st.columns(2)
#             with col1:
#                 new_name = st.text_input("✏️ Rename to", key="rename_input")
#                 if st.button("Rename"):
#                     rename_chat(selected, new_name)
#             with col2:
#                 if st.button("Delete"):
#                     delete_chat(selected)
#                     st.rerun()

#         st.divider()

#         theme_toggle = st.toggle("🌗 Dark Mode", value=st.session_state.theme == "dark")
#         st.session_state.theme = "dark" if theme_toggle else "light"

#         if st.button("🆕 New Chat"):
#             st.session_state.conversation_history = []
#             st.session_state.vector_store = None
#             st.session_state.current_chat_title = None
#             st.rerun()

#     # Set background color dynamically
#     if st.session_state.theme == "dark":
#         st.markdown("<style>body { background-color: #121212; color: white; }</style>", unsafe_allow_html=True)

#     show_chat_messages()
#     user_question = st.chat_input("Type your message here...")
#     if user_question:
#         handle_user_query(user_question)

# if __name__ == "__main__":
#     main()




import asyncio
import sys
import os
import json
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from together import Together

# ===== Windows Event Loop Fix =====
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ===== Env Load =====
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ===== Chat History Folder =====
CHAT_DIR = Path("chat_history")
CHAT_DIR.mkdir(exist_ok=True)

# ===== CSS & Theme Toggle =====
def inject_css_and_theme():
    theme = st.session_state.get("theme", "light")
    bg_color = "#121212" if theme == "dark" else "#f5f7fa"
    text_color = "#f0f0f0" if theme == "dark" else "#000000"

    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
            font-family: 'Segoe UI', sans-serif;
        }}
        .chat-container {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .chat-bubble {{
            max-width: 80%;
            padding: 12px 20px;
            border-radius: 18px;
            margin-bottom: 10px;
            font-size: 15px;
            line-height: 1.5;
            animation: slideFadeIn 0.5s ease-in-out;
        }}
        .user {{
            background-color: #d1e7dd;
            align-self: flex-end;
        }}
        .bot {{
            background-color: #ffffff;
            border: 1px solid #dee2e6;
            align-self: flex-start;
        }}
        @keyframes slideFadeIn {{
            0% {{ opacity: 0; transform: translateY(15px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
        .timestamp {{
            font-size: 11px;
            color: #6c757d;
            margin-top: -8px;
            margin-bottom: 6px;
        }}
        </style>
    """, unsafe_allow_html=True)

# ===== Chat UI =====
def show_chat_messages():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for user_msg, bot_msg, model, timestamp in st.session_state.conversation_history:
        st.markdown(f'<div class="chat-bubble user">{user_msg}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="timestamp">🕒 {timestamp}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bubble bot"><strong>{model}</strong>: {bot_msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def fetch_together_chat(prompt_messages):
    client = Together().chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=prompt_messages
    )
    return client.choices[0].message.content

def ensure_async_loop():
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

def embed_and_store(chunks):
    ensure_async_loop()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    return FAISS.from_texts(chunks, embeddings)

def process_multiple_pdfs(pdf_files):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for pdf_file in pdf_files:
        path = f"temp_{pdf_file.name}"
        with open(path, "wb") as f:
            f.write(pdf_file.read())
        pages = PyPDFLoader(path).load()
        os.remove(path)
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
    return all_chunks

# ===== Chat Save/Load Logic =====
def get_chat_name_from_message(msg):
    return msg.strip().split(" ")[0][:10] or uuid.uuid4().hex[:6]

def save_chat_auto():
    if st.session_state.conversation_history:
        base_name = get_chat_name_from_message(st.session_state.conversation_history[0][0])
        filename = CHAT_DIR / f"{base_name}_{datetime.now().strftime('%H%M%S')}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(st.session_state.conversation_history, f)
        st.session_state.current_chat_title = filename.name

def load_chat(chat_file):
    with open(chat_file, "rb") as f:
        st.session_state.conversation_history = pickle.load(f)

def rename_chat_file(old_name, new_title):
    new_file = CHAT_DIR / f"{new_title}.pkl"
    os.rename(CHAT_DIR / old_name, new_file)
    st.session_state.current_chat_title = new_file.name
    st.rerun()

def delete_chat_file(file_name):
    os.remove(CHAT_DIR / file_name)
    if st.session_state.current_chat_title == file_name:
        st.session_state.conversation_history = []
        st.session_state.current_chat_title = None
    st.rerun()

# ===== Main Handler =====
def handle_user_query(user_question):
    with st.spinner("🤔 Thinking..."):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if st.session_state.vector_store:
            docs = st.session_state.vector_store.similarity_search(user_question, k=3)
            context_texts = [doc.page_content for doc in docs]

            prompt = [{"role": "system", "content": "Use the context to answer carefully."}]
            for ctxt in context_texts:
                prompt.append({"role": "system", "content": ctxt})
            prompt.append({"role": "user", "content": user_question})

            answer = fetch_together_chat(prompt)
            source = "📘 Llama 3.3-RAG"
        else:
            prompt = [{"role": "user", "content": user_question}]
            answer = fetch_together_chat(prompt)
            source = "🧠 Llama 3.3-Chat"

        st.session_state.conversation_history.append((user_question, answer, source, timestamp))
        save_chat_auto()
        show_chat_messages()

# ===== Main App =====
def main():
    st.set_page_config(page_title="💬 Llama Chat", layout="wide")
    inject_css_and_theme()

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "current_chat_title" not in st.session_state:
        st.session_state.current_chat_title = None
    if "theme" not in st.session_state:
        st.session_state.theme = "light"

    st.title("🤖 Llama 3.3‑Chat + PDF RAG")

    with st.sidebar:
        st.header("📁 Upload PDFs")
        pdf_files = st.file_uploader("Upload Multiple PDFs", type=["pdf"], accept_multiple_files=True)
        if pdf_files and st.button("⚙️ Process PDFs"):
            docs = process_multiple_pdfs(pdf_files)
            chunks = [doc.page_content for doc in docs]
            st.session_state.vector_store = embed_and_store(chunks)
            st.success("✅ PDFs Indexed")

        st.divider()
        st.header("🧠 Chat History")

        options = ["-- Select --"] + [f.name for f in CHAT_DIR.glob("*.pkl")]
        selected = st.selectbox("📂 Choose Chat", options, key="chat_select")

        if selected != "-- Select --":
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                if st.button("Load", key="load_button"):
                    load_chat(CHAT_DIR / selected)
                    st.session_state.current_chat_title = selected
            with col2:
                if st.button("🗑️", key="del_btn"):
                    delete_chat_file(selected)

            new_title = st.text_input("✏️ Rename Chat", value=selected.replace(".pkl", ""), key="rename_input")
            if st.button("✅ Rename"):
                rename_chat_file(selected, new_title)

        if st.button("🆕 New Chat"):
            st.session_state.conversation_history = []
            st.session_state.vector_store = None
            st.session_state.current_chat_title = None
            if "chat_select" in st.session_state:
                del st.session_state["chat_select"]
            st.rerun()

        theme_toggle = st.toggle("🌙 Dark Mode", value=(st.session_state.theme == "dark"))
        st.session_state.theme = "dark" if theme_toggle else "light"

    show_chat_messages()

    user_question = st.chat_input("Ask something...")
    if user_question:
        handle_user_query(user_question)

if __name__ == "__main__":
    main()
