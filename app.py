import streamlit as st
import os
import openai
import tiktoken
from io import StringIO
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="RAG with OpenAI", layout="wide")
st.title("🧠 RAG Application using OpenAI (gpt-4o)")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "kill_session" not in st.session_state:
    st.session_state.kill_session = False
if "token_usage" not in st.session_state:
    st.session_state.token_usage = {"input": 0, "output": 0}

# Sidebar controls
with st.sidebar:
    st.header("🛠️ Controls")
    if st.button("🔴 Kill Session"):
        st.session_state.kill_session = True
        st.rerun()
    if st.button("🧹 Clear History"):
        st.session_state.history = []
        st.session_state.token_usage = {"input": 0, "output": 0}
        st.success("History and tokens cleared.")
    if st.session_state.history:
        # Export conversation history
        chat_export = StringIO()
        for i, msg in enumerate(st.session_state.history):
            chat_export.write(f"{msg['role'].capitalize()} {i+1}:\n{msg['message']}\n\n")
        st.download_button("📥 Download Chat History", chat_export.getvalue(), file_name="chat_history.txt")

    # Token and cost display
    total_input = st.session_state.token_usage["input"]
    total_output = st.session_state.token_usage["output"]
    total_tokens = total_input + total_output
    estimated_cost = (total_input / 1e6 * 5) + (total_output / 1e6 * 15)
    st.markdown("### 📊 Token Usage")
    st.text(f"Input Tokens: {total_input}")
    st.text(f"Output Tokens: {total_output}")
    st.text(f"Total Cost: ${estimated_cost:.4f}")

# Kill session check
if st.session_state.kill_session:
    st.warning("🔒 Session killed. Restart the app to continue.")
    st.stop()

# File upload
uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])

# Token counter
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    try:
        loader = PyPDFLoader("temp.pdf")
        data = loader.load()
    except Exception as e:
        st.error(f"❌ Error reading PDF: {str(e)}")
        st.stop()

    if not all(isinstance(doc, Document) for doc in data):
        st.error("❌ Failed to load PDF content into Document objects.")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)

    if not docs:
        st.error("❌ No valid document chunks.")
        st.stop()

    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"❌ Error creating vectorstore: {str(e)}")
        st.stop()

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following document content clearly:\n\n{context}"),
        ("human", "{input}")
    ])
    summary_chain = create_stuff_documents_chain(llm, summary_prompt)

    question_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the provided context to answer the question.\n\n{context}"),
        ("human", "{input}")
    ])
    question_chain = create_stuff_documents_chain(llm, question_prompt)
    rag_chain = create_retrieval_chain(retriever, question_chain)

    # User input
    query = st.chat_input("💬 Ask a question or type 'summarize'")
    if query:
        st.session_state.history.append({"role": "user", "message": query})
        st.session_state.token_usage["input"] += count_tokens(query)

        if "summary" in query.lower():
            st.subheader("📝 Summary")
            try:
                summaries = []
                for i, chunk in enumerate(docs):
                    partial = summary_chain.invoke({
                        "context": [chunk],
                        "input": "Summarize this."
                    })
                    result = partial.get("output", "") if isinstance(partial, dict) else str(partial)
                    st.session_state.token_usage["output"] += count_tokens(result)
                    summaries.append(f"Summary {i+1}:\n{result.strip()}")
                final_summary = "\n\n".join(summaries)
                st.session_state.history.append({"role": "assistant", "message": final_summary})
                st.write(final_summary)
            except Exception as e:
                st.error(f"❌ Error during summarization: {str(e)}")
        else:
            st.subheader("🤖 AI Assistant Response")
            try:
                response = rag_chain.invoke({"input": query})
                output = response.get("answer") or response.get("output") or str(response)
                st.session_state.token_usage["output"] += count_tokens(output)
                st.session_state.history.append({"role": "assistant", "message": output})
                st.write(output)
            except Exception as e:
                st.error(f"❌ Error during question answering: {str(e)}")

# Display chat history
if st.session_state.history:
    st.divider()
    st.subheader("🗂️ Chat History")
    for i, msg in enumerate(st.session_state.history):
        with st.expander(f"{msg['role'].capitalize()} {i+1}"):
            st.write(msg["message"])
