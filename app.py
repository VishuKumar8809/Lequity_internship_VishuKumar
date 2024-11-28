import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    if not pdf_docs:
        st.error("No PDF files uploaded.")
        return ""
    
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    
    if not text:
        st.error("Could not extract text from PDFs.")
    
    return text

def get_text_chunks(text):
    if not text:
        st.error("No text to chunk.")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        st.error("Text splitting failed.")
    
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("No text chunks to create vector store.")
        return
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("Vector store created successfully.")
    except Exception as e:
        st.error(f"Vector store creation failed: {e}")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    Make sure to provide all the details. If the answer is not in the provided context, 
    just say, "answer is not available in the context", don't provide the wrong answer.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question}, 
            return_only_outputs=True
        )
        
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"Error processing question: {e}")

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)

if __name__ == "__main__":
    main()