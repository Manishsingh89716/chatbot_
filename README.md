*** Chat with PDF using Gemini
* This project is a Streamlit-based web application that allows users to upload multiple PDF files, process them into a searchable knowledge base, and interact with the content using a conversational AI model (Gemini). The system leverages Google Generative AI for embeddings and FAISS for vector storage.
* 
* Features
* PDF Upload: Upload multiple PDF files to extract text.
* Text Chunking: Splits the extracted text into manageable chunks for better embedding performance.
* FAISS Indexing: Stores and retrieves text chunks using vector embeddings.
* Conversational Interface: Chat with the processed PDFs using Gemini's AI.
* Secure API Integration: Uses Google Generative AI for embeddings and conversational models.
* 
* Installation
* 1. Clone the Repository
* git clone https://github.com/your-username/chat-with-pdf.git
* cd chat-with-pdf
* 2. Install Dependencies
* pip install streamlit PyPDF2 langchain langchain-google-genai python-dotenv faiss-cpu google-generativeai
* 3. Set Up Environment Variables

* Create a .env file in the project directory and add your Google API Key:
* env
* GOOGLE_API_KEY=your_google_api_key_here
* Replace your_google_api_key_here with your actual Google API key.
* 
* Usage

* 1. Run the Application
* streamlit run app.py

* 2. Upload PDFs
* Use the sidebar to upload multiple PDF files.
* Click on Submit & Process to generate the FAISS vector index.

* 3. Ask Questions
* Enter your query in the input box on the main page.
* The AI will provide detailed answers based on the uploaded PDFs.
* 
* File Structure

* .
* ├── app.py                  # Main application code
* ├── .env                    # Environment variables for sensitive data
* ├── requirements.txt        # Python dependencies (optional)
* └── README.md               # Project documentation
* 
* Workflow
* PDF Processing:
* PDFs are uploaded via Streamlit.
* Text is extracted using PyPDF2.
* Text chunks are created using RecursiveCharacterTextSplitter.
* 
* Vector Indexing:
* Text chunks are embedded using Google Generative AI embeddings.
* FAISS is used to create and store a searchable vector index.
* 
* Conversational AI:
* Queries are matched with text chunks using FAISS.
* AI models (Gemini) generate detailed responses based on the context.
* 
* Requirements
* Python 3.9 or higher
* Libraries:
* Streamlit
* PyPDF2
* LangChain
* FAISS
* Google Generative AI
* Python dotenv
* 
* Notes
* Ensure your Google API Key is valid and has access to Generative AI services.
* The application supports multiple PDFs but ensure individual PDF sizes are reasonable for optimal performance.
* For secure deserialization of FAISS indices, ensure the faiss_index directory is trustworthy.
* 
* Troubleshooting
* Error: GOOGLE_API_KEY not set
* Ensure the .env file is properly configured with your Google API Key.
* Error: ValueError: allow_dangerous_deserialization
* This occurs if FAISS indexing requires loading pickle files. The code already includes allow_dangerous_deserialization=True, but ensure the faiss_index directory is secure.
* 
* License
* This project is licensed under the MIT License.
* 
* Contributing
* Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
***