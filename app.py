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
from langchain.chains.retrieval import create_retrieval_chain

from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever, MultiVectorRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
# Initial setup
load_dotenv()
import sys
package = __import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


st.header("Full Stack RAG by Natan")
prompt_template = """
Use the following context to answer the question at the end. If you don't know the answer, just say that you don't know, dont try to make up an answer.
<context>
{context}
</context>
Question: {input}
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "input"])
id_key = "doc_key"

# documents=[]
# if 'documents' not in st.session_state:
#     st.session_state.documents = []
if 'counts' not in st.session_state:
    st.session_state.counts = {}
# docstore_elements = []
# vectorstore_elements = []
if 'docstore_elements' not in st.session_state:
    st.session_state.docstore_elements = []
if 'vectorstore_elements' not in st.session_state:
    st.session_state.vectorstore_elements = []

# db_multi_vector = None
# retriever_multi_vector = None
if 'docstore_multi_vector' not in st.session_state:
    st.session_state.docstore_multi_vector = InMemoryStore()

if 'retriever_db' not in st.session_state:
    st.session_state.retriever_db = None
if 'retriever_multi_query_only' not in st.session_state:
    st.session_state.retriever_multi_query_only = None
if 'retriever_multi_vector_only' not in st.session_state:
    st.session_state.retriever_multi_vector_only = None
if 'retriever_multi_query_and_multi_vector' not in st.session_state:
    st.session_state.retriever_multi_query_and_multi_vector = None

if 'db_multi_vector' not in st.session_state:
    st.session_state.db_multi_vector = None

if 'temp_dir' not in st.session_state:  
    st.session_state.temp_dir = tempfile.TemporaryDirectory()
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
col1, col2 = st.columns([3, 1])  # Adjust column sizes as needed
with col1:
    select_files_holder = st.write("Upload files on the left!")
with col2:
    st.write("### Logs")
def log_message(message):
    st.session_state.log_messages.append(message)
    with col2:
        # st.write(f"{message}")
        st.markdown(
            f"""
            <span style="color: green; font-size: 14px;">{message}</span>
            """,
            unsafe_allow_html=True
        )
        # display_logs() 

# Function to display logs
def display_logs():
    if st.session_state.log_messages:
        st.write("### Logs")
        for msg in st.session_state.log_messages:
            st.write(f"- {msg}")

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
        embeddings = []
        for chunk in texts:
            if isinstance(chunk, Document):
                text_to_embed = chunk.page_content
            elif isinstance(chunk, str):
                text_to_embed = chunk
            else:
                raise TypeError(f"Unsupported type {type(chunk)} in embed_documents.")
            
            embedding = get_embedding(text_to_embed)
            embeddings.append(embedding)
        
        return embeddings
        # return [get_embedding(chunk.page_content) for chunk in texts]
    
    def embed_query(self, text):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

embedding_model = CustomEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)

class Sentences(BaseModel):
    sentences:  List[str]
obj = hub.pull("wfh/proposal-indexing")

extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=llm) 
extraction_runnable = obj | llm

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
    # global counts
    results = []
    with col1:
        select_files_holder = st.empty()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith('.zip'):
                # Handle ZIP files
                zip_path = os.path.join(temp_dir, uploaded_file.name)
                with open(zip_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                log_message(f"Extracted {uploaded_file.name} to temporary directory.")

            else:
                # Save non-ZIP files directly to temp
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                log_message(f"Uploaded file: {uploaded_file.name}")

        # Process files
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            if uploaded_file.name.endswith('.pdf'):
                ext = 'PDF'
                pdf_data = process_pdf(file_path)
                results.append((ext, pdf_data))
                
            elif uploaded_file.name.endswith('.xml'):
                ext = 'XML'
                loader = DirectoryLoader(path=temp_dir, glob='**/*.xml', loader_cls=UnstructuredXMLLoader)
                xml_documents = loader.load()
                results.append((ext, xml_documents))
            elif uploaded_file.name.endswith('.csv'):
                ext = 'CSV'
                loader = DirectoryLoader(path=temp_dir, glob='**/*.csv', loader_cls=CSVLoader)
                csv_documents = loader.load()
                results.append((ext, csv_documents))
            if ext in st.session_state.counts:
                st.session_state.counts[ext] = st.session_state.counts[ext] + 1 
            else:
                st.session_state.counts[ext] = 1
    
    return results
def get_propositions(text):
    log_message("Proposition input: " + text)  # Log the runnable output

    runnable_output = extraction_runnable.invoke({'input': text}).content
    log_message("Proposition Output: " + str(runnable_output))  # Log the runnable output
    propositions = extraction_chain.run(runnable_output)
    return propositions
# Function to extract files
def extract_files(doc, extraction_type, pdf=False):
    # global docstore_elements, vectorstore_elements
    if not pdf:
        splits = text_splitter.split_documents(doc)
        for split in splits:
            unique_id = split.metadata[id_key]
            split.metadata[id_key] = unique_id
            st.session_state.docstore_elements.append(split)
    else:
        splits = [
            Document(page_content=d.text, metadata={id_key: d.metadata[id_key]})
            for d in doc
        ]
        
        st.session_state.docstore_elements += splits
    for split in splits:
        unique_id = split.metadata[id_key]
        if extraction_type == "propositions":
            propositions = get_propositions(split.page_content)
            for p in propositions:
                for sentence in p.sentences:
                    chunk_summary_document = Document(page_content=sentence, metadata={id_key: unique_id})
                    st.session_state.vectorstore_elements.append(chunk_summary_document)
            # log_message(f"Extracted propositions for document ID: {unique_id}")

        elif extraction_type == "summary":
            chunk_summary = summarize_chain.run([split])
            chunk_summary_document = Document(page_content=chunk_summary, metadata={id_key: unique_id})
            st.session_state.vectorstore_elements.append(chunk_summary_document)
            # log_message(f"Generated summary for document ID: {unique_id}")

        elif extraction_type == "basic":
            chunk_document = Document(page_content=split.page_content, metadata={id_key: unique_id})
            st.session_state.vectorstore_elements.append(chunk_document)
            # log_message(f"Basic extraction for document ID: {unique_id}")

# Vector store initialization after file extraction
# def initialize_vectorstore():
#     # global db_multi_vector, retriever_multi_vector

    
#     # st.session_state.db_multi_vector = Chroma.from_documents(st.session_state.vectorstore_elements, embedding_model)

#     # st.session_state.retriever_multi_vector = MultiVectorRetriever(
#     #     vectorstore=st.session_state.db_multi_vector,
#     #     docstore=st.session_state.docstore_multi_vector,
#     #     id_key=id_key
#     # )

#     # st.session_state.retriever_multi_vector.docstore.mset([(doc.metadata[id_key], doc) for doc in st.session_state.docstore_elements])
#     temp_dir = tempfile.gettempdir()  # Get temporary directory
#     db_path = os.path.join(temp_dir, "chroma_db")
#     if 'db_multi_vector' not in st.session_state or st.session_state.db_multi_vector is None:
#         # Only create the vectorstore if it doesn't already exist
#         st.session_state.db_multi_vector = Chroma.from_documents(st.session_state.vectorstore_elements, embedding_model)
#         st.session_state.retriever_multi_vector = MultiVectorRetriever(
#             vectorstore=st.session_state.db_multi_vector,
#             docstore=st.session_state.docstore_multi_vector,
#             id_key=id_key
#         )
#         st.session_state.retriever_multi_vector.docstore.mset([(doc.metadata[id_key], doc) for doc in st.session_state.docstore_elements])
def initialize_vectorstore():
    db_path = os.path.join(st.session_state.temp_dir.name, "chroma_db")  # Use user's temporary directory

    # Initialize the vectorstore from existing database or create a new one
    if os.path.exists(db_path):
        # st.session_state.db_multi_vector = Chroma.load_from_path(db_path, embedding_model)
        st.session_state.db_multi_vector = Chroma(persist_directory=db_path, embedding_function=embedding_model)
        log_message(f"Loaded existing vector store from {db_path}.")

    else:
        st.session_state.db_multi_vector = Chroma.from_documents(st.session_state.vectorstore_elements, embedding_model, persist_directory=db_path)
        st.session_state.db_multi_vector.persist()  # Save to temporary file
        log_message("Created a new vector store.")

    st.session_state.retriever_db = st.session_state.db_multi_vector.as_retriever()
    st.session_state.retriever_multi_vector_only = MultiVectorRetriever(
        vectorstore=st.session_state.db_multi_vector,
        docstore=st.session_state.docstore_multi_vector,
        id_key=id_key
    )
    st.session_state.retriever_multi_query_only = MultiQueryRetriever.from_llm(
        retriever=st.session_state.retriever_db,
        llm=llm
    )
    st.session_state.retriever_multi_query_and_multi_vector = MultiQueryRetriever.from_llm(
        retriever=st.session_state.retriever_multi_vector_only,
        llm=llm
    )
    st.session_state.retriever_multi_vector_only.docstore.mset([(doc.metadata[id_key], doc) for doc in st.session_state.docstore_elements])
    log_message("Initialized retrievers for multi-query and multi-vector retrieval.")


# Streamlit sidebar and query section
st.sidebar.title("Upload Files")
uploaded_files = st.sidebar.file_uploader("Upload files (PDF, CSV, XML, ZIP)", type=['pdf', 'csv', 'xml', 'zip'], accept_multiple_files=True)
extraction_type = st.sidebar.selectbox("Select extraction type:", options=["basic", "propositions", "summary"])

if st.sidebar.button("Process Files"):
    if uploaded_files:
        with st.spinner('Processing files...'):
            results = process_uploaded_files(uploaded_files)
            log_message(f"Starting extraction...")
            for ext, docs in results:
                uuids = [str(uuid.uuid4()) for _ in docs]  # Generate UUIDs
                for doc, doc_uuid in zip(docs, uuids):
                    if isinstance(doc.metadata, dict):
                        doc.metadata[id_key] = doc_uuid
                    else:
                        metadata_dict = doc.metadata.to_dict()
                        metadata_dict[id_key] = doc_uuid
                        doc.metadata = metadata_dict

                # with col1:
                #     st.write(f"Loaded {st.session_state.counts[ext]} {ext} documents with {len(docs)} elements.")
                log_message(f"Loaded {st.session_state.counts[ext]} {ext} documents with {len(docs)} elements.")

                if ext == "PDF":
                    extract_files(docs, extraction_type, pdf=True)
                else:
                    extract_files(docs, extraction_type)
            initialize_vectorstore()  # Initialize vector store after files are processed
            st.sidebar.success("Files processed. You can now query the documents.")
            log_message("File processing completed and vector store initialized.")

    else:
        st.sidebar.error("Please upload at least one file.")
with col1:
# User input for querying
    if st.session_state.docstore_elements:
        query = st.text_input("Enter your query:")
        use_compression = st.checkbox("Enable Compression")
        enable_multi_query = st.checkbox("Enable Multi Query")
        enable_multi_vector = st.checkbox("Enable Multi Vector")
        if st.button("Submit Query"):
            if query:
                log_message(f"Query submitted: {query}")

                if use_compression:
                    # Use compression retriever
                    with st.spinner('Compressing relevant documents, please wait...'):
                        compressor = LLMChainExtractor.from_llm(llm)
                        retriever = None
                        if enable_multi_query and enable_multi_vector:
                            retriever = st.session_state.retriever_multi_query_and_multi_vector
                        elif enable_multi_query: 
                            retriever = st.session_state.retriever_multi_query_only
                        elif enable_multi_vector:
                            retriever = st.session_state.retriever_multi_vector_only
                        else:
                            retriever = st.session_state.retriever_db
                        compression_retriever = ContextualCompressionRetriever(
                            base_compressor=compressor,
                            base_retriever=retriever
                        )
                        with st.spinner('Retrieving relevant documents, please hold on...'):
                            unique_docs = compression_retriever.get_relevant_documents(query)
                        log_message(f"Retrieved {len(unique_docs)} unique documents using compression.")

                        # docs_retrieved_multi_vector = compression_retriever.get_relevant_documents(query)
                else:
                    with st.spinner('Retrieving relevant documents, please hold on...'):
                        if enable_multi_query and enable_multi_vector: 
                            with st.spinner('Analyzing your query for optimal results, please hold on...'):
                                unique_docs = st.session_state.retriever_multi_query_and_multi_vector.get_relevant_documents(query)
                        elif enable_multi_query:
                            with st.spinner('Analyzing your query for optimal results, please hold on...'):
                                unique_docs = st.session_state.retriever_multi_query_only.get_relevant_documents(query)      
                        elif enable_multi_vector:
                            unique_docs = st.session_state.retriever_multi_vector.get_relevant_documents(query)
                        log_message(f"Retrieved {len(unique_docs)} unique documents without compression.")

                        # Without compression
                        # docs_retrieved_multi_vector = st.session_state.retriever_multi_vector.get_relevant_documents(query)
                with st.spinner('Generating response, this may take a moment...'):
                    docs_chain = create_stuff_documents_chain(ChatGroq(), PROMPT)
                    retrieval_chain_multi_vector = create_retrieval_chain(st.session_state.retriever_multi_vector_only, docs_chain)
                    # response = retrieval_chain_multi_vector.invoke({"context": docs_retrieved_multi_vector, "input": query})
                    response = retrieval_chain_multi_vector.invoke({"context": unique_docs, "input": query})
                    with col1:
                        st.write(response['answer'])
                    log_message("Response generated and displayed.")

            else:
                st.error("Please enter a query.")
                log_message("Query submission failed: No query entered.")

