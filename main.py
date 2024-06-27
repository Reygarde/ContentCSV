import streamlit as st
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# LLM and key loading function
def load_LLM(openai_api_key):
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    return llm

# Page title and header
st.set_page_config(page_title="Ask from CSV File with FAQs about Napoleon")
st.header("Ask from CSV File with FAQs about Napoleon")

# Input OpenAI API Key
def get_openai_api_key():
    input_text = st.text_input(
        label="OpenAI API Key ",  
        placeholder="Ex: sk-2twmA8tfCb8un4...", 
        key="openai_api_key_input", 
        type="password")
    return input_text

openai_api_key = get_openai_api_key()

# Check if the OpenAI API key is provided
if openai_api_key:
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

    vectordb_file_path = "my_vectordb"

    def create_db():
        loader = CSVLoader(file_path='napoleon-faqs.csv', source_column="prompt")
        documents = loader.load()
        vectordb = FAISS.from_documents(documents, embedding)

        # Save vector database locally
        vectordb.save_local(vectordb_file_path)

    def execute_chain():
        # Check if vector database exists
        if not os.path.exists(vectordb_file_path):
            st.warning("Vector database not found. Creating a new one...")
            create_db()
        
        # Load the vector database from the local folder with dangerous deserialization allowed
        vectordb = FAISS.load_local(vectordb_file_path, embedding, allow_dangerous_deserialization=True)

        # Create a retriever for querying the vector database
        retriever = vectordb.as_retriever(score_threshold=0.7)

        template = """Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
        If the answer is not found in the context, respond "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}"""

        prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )
        
        llm = load_LLM(openai_api_key=openai_api_key)

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        return chain

    if __name__ == "__main__":
        if not os.path.exists(vectordb_file_path):
            create_db()
        chain = execute_chain()

    # Button to re-create the database
    btn = st.button("Private button: re-create database")
    if btn:
        create_db()
        st.success("Database re-created successfully.")

    question = st.text_input("Question: ")

    if question:
        chain = execute_chain()
        response = chain(question)

        st.header("Answer")
        st.write(response["result"])
