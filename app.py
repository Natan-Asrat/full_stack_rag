import os
import tempfile
import zipfile
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_community.document_loaders import CSVLoader
from unstructured.partition.pdf import partition_pdf
from pydantic import BaseModel, validator
from langchain import hub
from typing import Union, List
from langchain_groq import ChatGroq
from langchain.chains import create_extraction_chain_pydantic
from langchain.schema.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever, MultiVectorRetriever
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
# Initial setup
load_dotenv()
import sys
package = __import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



prompt_template = """
Use the following context to answer the question at the end. If you don't know the answer, just say that you don't know, dont try to make up an answer.
<context>
{context}
</context>
Question: {input}
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "input"])
id_key = "doc_key"

documents = []
docstore_elements = []
vectorstore_elements = []
db_multi_vector = None
retriever_multi_vector = None

# Initialize language model
llm = ChatGroq()
summarize_chain = load_summarize_chain(llm)
# Cache resource to load model
@st.cache_resource
def load_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Function to get embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

# Custom Embeddings class
class CustomEmbeddings:
    def embed_documents(self, texts):
        return [get_embedding(chunk.page_content) for chunk in texts]
    
    def embed_query(self, text):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

embedding_model = CustomEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)

# Function to process PDF files
def process_pdf(file_path):
    elements = partition_pdf(
        filename=file_path,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        strategy="hi_res",
        infer_table_structure=True
    )
    return elements

# Function to process uploaded files
def process_uploaded_files(uploaded_files):
    results = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith('.zip'):
                # Handle ZIP files
                zip_path = os.path.join(temp_dir, uploaded_file.name)
                with open(zip_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            else:
                # Save non-ZIP files directly to temp
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

        # Process files
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            if uploaded_file.name.endswith('.pdf'):
                pdf_data = process_pdf(file_path)
                results.append(('PDF', pdf_data))
            elif uploaded_file.name.endswith('.xml'):
                loader = DirectoryLoader(path=temp_dir, glob='**/*.xml', loader_cls=UnstructuredXMLLoader)
                xml_documents = loader.load()
                results.append(('XML', xml_documents))
            elif uploaded_file.name.endswith('.csv'):
                loader = DirectoryLoader(path=temp_dir, glob='**/*.csv', loader_cls=CSVLoader)
                csv_documents = loader.load()
                results.append(('CSV', csv_documents))
    
    return results

# Function to extract files
def extract_files(doc, extraction_type, pdf=False):
    unique_id = str(uuid.uuid4())
    if not pdf:
        splits = text_splitter.split_documents(doc)
        for split in splits:
            split.metadata[id_key] = unique_id
            docstore_elements.append(split)
    else:
        text = doc.text
        doc_gen = Document(page_content=text, metadata={id_key: unique_id})
        splits = [doc_gen]
        docstore_elements.append(doc_gen)

    if extraction_type == "propositions":
        # Custom proposition extraction logic
        pass
    elif extraction_type == "summary":
        chunk_summary = summarize_chain.run(splits)
        chunk_summary_document = Document(page_content=chunk_summary, metadata={id_key: unique_id})
        vectorstore_elements.append(chunk_summary_document)
    elif extraction_type == "basic":
        for split in splits:
            chunk_document = Document(page_content=split.page_content, metadata={id_key: unique_id})
            vectorstore_elements.append(chunk_document)

# Vector store initialization after file extraction
def initialize_vectorstore():
    global db_multi_vector, retriever_multi_vector

    docstore_multi_vector = InMemoryStore()
    db_multi_vector = Chroma.from_documents(vectorstore_elements, embedding_model)

    retriever_multi_vector = MultiVectorRetriever(
        vectorstore=db_multi_vector,
        docstore=docstore_multi_vector,
        id_key=id_key
    )

    retriever_multi_vector.docstore.mset([(doc.metadata[id_key], doc) for doc in docstore_elements])

# Streamlit sidebar and query section
st.sidebar.title("Upload Files")
uploaded_files = st.sidebar.file_uploader("Upload files (PDF, CSV, XML, ZIP)", type=['pdf', 'csv', 'xml', 'zip'], accept_multiple_files=True)
extraction_type = st.sidebar.selectbox("Select extraction type:", options=["basic", "propositions", "summary"])

if st.sidebar.button("Process Files"):
    if uploaded_files:
        results = process_uploaded_files(uploaded_files)
        for ext, docs in results:
            st.write(f"Loaded {len(docs)} {ext} documents.")
            for doc in docs:
                if ext == "PDF":
                    extract_files(doc, extraction_type, pdf=True)
                else:
                    extract_files(doc, extraction_type)
        initialize_vectorstore()  # Initialize vector store after files are processed
        st.sidebar.success("Files processed. You can now query the documents.")
    else:
        st.sidebar.error("Please upload at least one file.")

# User input for querying
if docstore_elements:
    query = st.text_input("Enter your query:")
    use_compression = st.checkbox("Enable Compression")

    if st.button("Submit Query"):
        if query:
            if use_compression:
                # Use compression retriever
                compressor = LLMChainExtractor.from_llm(llm)
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=retriever_multi_vector
                )
                compressed_docs = compression_retriever.get_relevant_documents(query)
                response = create_stuff_documents_chain(ChatGroq(), PROMPT).run(compressed_docs)
            else:
                # Without compression
                docs_retrieved_multi_vector = retriever_multi_vector.get_relevant_documents(query)
                response = create_stuff_documents_chain(ChatGroq(), PROMPT).run(docs_retrieved_multi_vector)
            
            st.write(response)
        else:
            st.error("Please enter a query.")
