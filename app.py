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
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

id_key = "doc_key"

documents=[]
docstore_elements=[]
vectorstore_elements = []

db_multi_vector = None
retriever_multi_vector = None

llm = ChatGroq()
obj = hub.pull("wfh/proposal-indexing")
summarize_chain = load_summarize_chain(llm)

@st.cache_resource
def load_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model
tokenizer, model = load_model()
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

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


class Sentences(BaseModel):
    sentences:  List[str]

extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=llm) 
extraction_runnable = obj | llm

# Define your Pydantic model
class PDFData(BaseModel):
    title: str
    content: str

    @validator('title', pre=True)
    def check_title(cls, value):
        if not value:
            raise ValueError('Title cannot be empty')
        return value

# Function to process PDF files using partition_pdf
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
    # return [PDFData(title=element['title'], content=element['content']) for element in elements]

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
def get_propositions(text):
    
    runnable_output = extraction_runnable.invoke({'input': text}).content
    print("Runnable Output:", runnable_output)  # Debugging output
    propositions = extraction_chain.run(runnable_output)
    return propositions

def extract_pdf(elements, type="propositions"):
    for e in elements:
        text = e.text
        extract_files(text, pdf=True)

# def extract_files(doc, pdf=False):
#     unique_id = str(uuid.uuid4())
#     if pdf:
#         text = e.text
#         docstore_elements.append(Document(page_content=doc, metadata={id_key: unique_id}))
#     else:
#         splits = text_splitter.split_documents(doc)
#         doc.metadata[id_key] = unique_id
#         docstore_elements.append(doc)
#     if type == "propositions":
#         prop = []
#         if pdf:
#             prop = get_propositions(text)
#         else:
#             p=[]
#             for split in splits:
#                 p = get_propositions(split.page_content)
#                 prop.extend(p)
#         props = [p.sentences for p in prop]
#         for sentence in props:
#             for s in sentence:  
#                 chunk_summary_document = Document(page_content=s, metadata={id_key: unique_id})
#                 vectorstore_elements.append(chunk_summary_document)
#     elif type == "summary":
#         if pdf:
#             chunk_summary = summarize_chain.run([text])
#         else:
#             chunk_summary = summarize_chain.run(splits)
#         chunk_summary_document = Document(page_content=chunk_summary, metadata={id_key: unique_id})
#         vectorstore_elements.append(chunk_summary_document)
#     elif type == "basic":
#         if pdf:
#             splits = text_splitter.split_documents([Document(page_content=text)])
#         for split in splits:
#             chunk_document = Document(page_content=split.page_content, metadata={id_key: unique_id})
#             vectorstore_elements.append(chunk_document)

def extract_files(doc, extraction_type):
    unique_id = str(uuid.uuid4())
    splits = text_splitter.split_documents(doc)

    for split in splits:
        split.metadata[id_key] = unique_id
        docstore_elements.append(split)

    if extraction_type == "propositions":
        propositions = get_propositions(doc.page_content)
        for p in propositions:
            for sentence in p.sentences:
                chunk_summary_document = Document(page_content=sentence, metadata={id_key: unique_id})
                vectorstore_elements.append(chunk_summary_document)
    elif extraction_type == "summary":
        chunk_summary = summarize_chain.run(splits)
        chunk_summary_document = Document(page_content=chunk_summary, metadata={id_key: unique_id})
        vectorstore_elements.append(chunk_summary_document)
    elif extraction_type == "basic":
        for split in splits:
            chunk_document = Document(page_content=split.page_content, metadata={id_key: unique_id})
            vectorstore_elements.append(chunk_document)



def store_files():
    pass

# Streamlit UI
st.title("File Upload and Processing")
extraction_type = st.selectbox("Select extraction type:", options=["basic", "propositions", "summary"])

uploaded_files = st.file_uploader("Upload files (PDF, CSV, XML, ZIP)", type=['pdf', 'csv', 'xml', 'zip'], accept_multiple_files=True)

if st.button("Process Files"):
    if uploaded_files:
        results = process_uploaded_files(uploaded_files)
        
        # for ext, docs in results:
        #     st.write(f"Loaded {len(docs)} {ext} documents.")
        #     for doc in docs:
        #         if ext is "PDF":
        #             extract_pdf(doc)
        #             # st.write(f"**Title:** {doc.title}")
        #             # st.write(f"**Content:** {doc.content[:200]}...")  # Display the first 200 characters
        #         else:
        #             extract_files(doc)
        #     store_files()
        for ext, docs in results:
            st.write(f"Loaded {len(docs)} {ext} documents.")
            for doc in docs:
                if ext == "PDF":
                    # for pdf_element in doc:
                    #     extract_files(pdf_element, extraction_type)
                    extract_files([doc], extraction_type)
                else:
                    extract_files(doc, extraction_type)
                    st.write(doc) 

    else:
        st.error("Please upload at least one file.")
