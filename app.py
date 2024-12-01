#import all required libraries and modules
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

class PDFProcessor:
    """Class to handle PDF processing."""
    @staticmethod
    def get_pdf_text(pdf_docs):
        """Extract text from PDF files."""
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    @staticmethod
    def get_text_chunks(text):
        """Split text into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_text(text)


class VectorStoreManager:
    """Class to manage FAISS vector store."""
    def __init__(self, api_key):
        self.api_key = api_key

    def create_vector_store(self, text_chunks):
        """Generate and save a FAISS vector store."""
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

    def load_vector_store(self):
        """Load an existing FAISS vector store."""
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


class ConversationalAI:
    """Class to handle the conversational AI logic."""
    def __init__(self):
        self.chain = self.setup_conversational_chain()

    @staticmethod
    def setup_conversational_chain():
        """Set up a conversational chain."""
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details.\n\n
        Context:\n{context}\n
        Question:\n{question}\n
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)

    def get_response(self, user_question, vector_store):
        """Get the AI response for the user question."""
        docs = vector_store.similarity_search(user_question)
        response = self.chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]


class ChatWithPDFApp:
    """Main application class."""
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")
        genai.configure(api_key=self.api_key)
        self.vector_store_manager = VectorStoreManager(self.api_key)
        self.conversational_ai = ConversationalAI()

    def run(self):
        """Run the Streamlit application."""
        st.set_page_config("Chat PDF", layout="wide")
        st.header("Chat with PDF using Gemini!")
        user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            vector_store = self.vector_store_manager.load_vector_store()
            response = self.conversational_ai.get_response(user_question, vector_store)
            st.write("Reply: ", response)

        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                        accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = PDFProcessor.get_pdf_text(pdf_docs)
                    text_chunks = PDFProcessor.get_text_chunks(raw_text)
                    self.vector_store_manager.create_vector_store(text_chunks)
                    st.success("Processing complete! You can now ask questions.")


if __name__ == "__main__":
    app = ChatWithPDFApp()
    app.run()
