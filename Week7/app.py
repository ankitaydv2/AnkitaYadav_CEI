# import os
# import tempfile
# import streamlit as st
# from dotenv import load_dotenv

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.prompts import ChatPromptTemplate

# load_dotenv()

# st.set_page_config(
#     page_title="Document Q&A RAG System",
#     layout="wide"
# )

# st.title("Document Question Answering System using RAG")
# st.write("Upload a PDF document and ask questions based on its content.")

# api_key = os.getenv("GOOGLE_API_KEY")

# if not api_key:
#     st.error("GOOGLE_API_KEY not found. Please add your Gemini API key in the .env file.")
#     st.stop()

# uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

# # Initialize session state tracking so we don't rebuild the DB on every keypress
# if "retriever" not in st.session_state:
#     st.session_state.retriever = None
#     st.session_state.num_pages = 0
#     st.session_state.num_chunks = 0

# if uploaded_file is not None:
#     # Process the PDF ONLY if it hasn't been cached in session state yet
#     if st.session_state.retriever is None:
#         with st.spinner("Processing PDF document and building vector store..."):
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#                 temp_file.write(uploaded_file.read())
#                 temp_file_path = temp_file.name

#             try:
#                 loader = PyPDFLoader(temp_file_path)
#                 documents = loader.load()

#                 if len(documents) == 0:
#                     st.error("No text found in the uploaded PDF.")
#                     st.stop()

#                 text_splitter = RecursiveCharacterTextSplitter(
#                     chunk_size=1000,
#                     chunk_overlap=200
#                 )
#                 chunks = text_splitter.split_documents(documents)

#                 # Using local Hugging Face Sentence Transformers for offline/robust embedding generation
#                 embeddings = HuggingFaceEmbeddings(
#                     model_name="all-MiniLM-L6-v2"
#                 )

#                 vectorstore = Chroma.from_documents(
#                     documents=chunks,
#                     embedding=embeddings
#                 )

#                 # Store components securely in Streamlit's cache
#                 st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
#                 st.session_state.num_pages = len(documents)
#                 st.session_state.num_chunks = len(chunks)

#             finally:
#                 # Clean up local temporary file safely
#                 if os.path.exists(temp_file_path):
#                     os.remove(temp_file_path)

#     # Display Document Statistics
#     st.success("PDF processed successfully.")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric(label="Total Pages Loaded", value=st.session_state.num_pages)
#     with col2:
#         st.metric(label="Total Text Chunks Created", value=st.session_state.num_chunks)

#     # Initialize standard Gemini inference model
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         temperature=0.2
#     )

#     # Question Input Field
#     question = st.text_input("Ask a question from the uploaded document:")

#     if question:
#         with st.spinner("Retrieving relevant context and generating answer..."):
#             # Pull retriever instantly from session state memory
#             retrieved_docs = st.session_state.retriever.invoke(question)

#             context = "\n\n".join(
#                 [doc.page_content for doc in retrieved_docs]
#             )

#             prompt = ChatPromptTemplate.from_template("""
# You are a helpful AI assistant.

# Answer the question using only the given context.
# If the answer is not present in the context, say:
# "The answer is not available in the uploaded document."

# Context:
# {context}

# Question:
# {question}

# Answer:
# """)

#             final_prompt = prompt.format(
#                 context=context,
#                 question=question
#             )

#             response = llm.invoke(final_prompt)

#         st.subheader("Generated Answer")
#         st.write(response.content)

#         st.subheader("Retrieved Context")
#         for i, doc in enumerate(retrieved_docs):
#             with st.expander(f"Chunk {i + 1} - Preview"):
#                 st.write(doc.page_content)

# else:
#     # If the user removes the file, wipe out the current session memory state
#     st.session_state.retriever = None
#     st.info("Please upload a PDF file to start the RAG question-answering process.")

import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


st.set_page_config(
    page_title="Document Q&A RAG System",
    layout="wide"
)

st.title("Document Question Answering System using RAG")
st.write("Upload a PDF document and ask questions based on its content.")


if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "num_pages" not in st.session_state:
    st.session_state.num_pages = 0

if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)


def make_answer_from_context(question, context):
    question_lower = question.lower()

    if "overview" in question_lower or "summary" in question_lower or "summarize" in question_lower:
        return (
            "Based on the retrieved document content, this document contains the following relevant information:\n\n"
            + context[:2500]
        )

    return (
        "Based on the retrieved document context, the most relevant information is:\n\n"
        + context[:2500]
    )


if uploaded_file is not None:

    if st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.retriever = None
        st.session_state.num_pages = 0
        st.session_state.num_chunks = 0
        st.session_state.uploaded_file_name = uploaded_file.name

    if st.session_state.retriever is None:
        with st.spinner("Processing PDF document and building vector store..."):

            temp_file_path = None

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                loader = PyPDFLoader(temp_file_path)
                documents = loader.load()

                if len(documents) == 0:
                    st.error("No readable text found in the uploaded PDF.")
                    st.stop()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = text_splitter.split_documents(documents)

                if len(chunks) == 0:
                    st.error("No text chunks created from the PDF.")
                    st.stop()

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )

                st.session_state.retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 5}
                )

                st.session_state.num_pages = len(documents)
                st.session_state.num_chunks = len(chunks)

            except Exception as e:
                st.error(f"Error while processing PDF: {e}")
                st.stop()

            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

    st.success("PDF processed successfully.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Pages Loaded", st.session_state.num_pages)

    with col2:
        st.metric("Total Text Chunks Created", st.session_state.num_chunks)

    question = st.text_input("Ask a question from the uploaded document:")

    if question:
        with st.spinner("Retrieving relevant context..."):

            retrieved_docs = st.session_state.retriever.invoke(question)

            context = "\n\n".join(
                [doc.page_content for doc in retrieved_docs]
            )

            answer = make_answer_from_context(question, context)

        st.subheader("Generated Answer")
        st.write(answer)

        st.subheader("Retrieved Context")

        for i, doc in enumerate(retrieved_docs):
            with st.expander(f"Chunk {i + 1} - Preview"):
                st.write(doc.page_content)

else:
    st.session_state.retriever = None
    st.session_state.num_pages = 0
    st.session_state.num_chunks = 0
    st.session_state.uploaded_file_name = None

    st.info("Please upload a PDF file to start the RAG question-answering process.")