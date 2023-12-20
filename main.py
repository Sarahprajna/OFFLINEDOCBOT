import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
import torch
import chromadb
from transformers import AutoModel
from chromadb.config import Settings
from transformers import pipeline

device = torch.device('cpu')
access_token = "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"


def load_model():
    llm = AutoModel.from_pretrained("meta-llama/Llama-2-7b-chat", token=access_token)
    return llm


@st.cache
def qa_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model="meta-llama/Llama-2-7b-chat",
        local_files_only=True
    )

    return pipe


def chroma_settings():
    settings = Settings(
        chroma_api_impl="chromadb.api.fastapi.FastAPI",
        persist_directory='db',
        chroma_server_host='localhost',
        chroma_server_http_port='8000',
        anonymized_telemetry=False
    )
    return settings


def textsplitter():
    texts = [Document(page_content=txt) for txt in uploaded_text]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    tests = text_splitter.split_documents(texts)
    return tests


def embeddings():
    sent_embeddings = HuggingFaceInstructEmbeddings(model_name="all-MiniLM-L6-v2")
    return sent_embeddings


def database():
    settings = chroma_settings()
    tests = textsplitter()
    sent_embeddings = embeddings()
    client = chromadb.Client()
    cdb = Chroma.from_documents(tests, sent_embeddings, persist_directory="db", client_settings=settings, client=client)
    cdb.persist()
    return cdb


@st.cache
def qa_llm():
    llm = qa_pipeline()
    db = database(previously_asked_queries)  # Assuming previously_asked_queries contains user queries
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=True)
    return qa


def process_answer():
    global instruction
    global answer
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer


st.set_page_config(
    page_title="DOCBOT",
    page_icon="👾",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(f"""
    <style>
    .sidebar.fixed {{
        background-image: linear-gradient(to right, #8E236A, #4A148C);
    }}
    </style>
    """, unsafe_allow_html=True)
with st.expander("About Docbot"):
    st.markdown(
        """
        DOCBOT can read and understand any type of document including Pdfs,Word Documents and many more.
        DOCBOT is still under development this is just a demo of communications with multiple Documents.
        """
    )

st.header("CHAT WITH DOCBOT 👾")
user_input = st.text_area("ASK YOUR QUERY....")
st.write("Query:" + user_input)
st.write("DocBot:")


def handle_send_button_click():
    if not user_input:
        st.error("Please enter a query to proceed.")
    return


instruction = user_input
if user_input:
    answer, metadata = process_answer()
    st.write("DocBot:", answer)
    st.write("DocBot:", metadata)
if st.button("SEND"):
    handle_send_button_click()

with st.sidebar:
    st.sidebar.title("DOCBOT 👾",)
    pdfs = st.file_uploader("Upload Your Documents", accept_multiple_files=True)

    # Check if the PDFs are not None
    if pdfs is not None:
        for pdf in pdfs:
            file_extension = pdf.name.split(".")[-1].lower()
            if file_extension == "pdf":
                pdf_reader = PdfReader(pdf)
                document = []
                for page in pdf_reader.pages:
                    document += page.extract_text()
                uploaded_text = document
                is_pdf_uploaded = True
    submit_button = st.sidebar.button("SUBMIT")

    if submit_button:
        database()
        # Pass the uploaded_text variable as an argument
        st.sidebar.text("Files submitted and processed.")

previously_asked_queries = []

st.sidebar.markdown("## Previously Asked Queries")
