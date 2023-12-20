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
