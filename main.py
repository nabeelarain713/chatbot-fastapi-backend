# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
os.environ['GOOGLE_API_KEY'] = "AIzaSyDAM58VDbpL8jXwH63swl3n-zNFPu8wOuY"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Initialize the model and memory (per session)
sessions = {}

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str

def initialize_session(session_id: str):
    if session_id not in sessions:
        # Initialize model
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        
        # Load and split the PDF
        pdf_file = "/app/data/DataSetPdf (1).pdf"  # Adjust path for your deployment
        pdf_loader = PyPDFLoader(pdf_file)
        pages = pdf_loader.load_and_split()
        
        # Define prompt template
        prompt_template = """Answer the question as precise as possible using the provided context. If the answer is
                            not contained in the context, say "answer not available in context" \n\n
                            Context: \n {context}?\n
                            Question: \n {question} \n
                            Answer:
                          """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # Create chains
        stuff_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        memory = ConversationBufferMemory()
        conversation_chain = ConversationChain(llm=model, memory=memory)
        
        sessions[session_id] = {
            "model": model,
            "pages": pages,
            "stuff_chain": stuff_chain,
            "conversation_chain": conversation_chain,
            "memory": memory
        }

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    session_id = chat_request.session_id
    initialize_session(session_id)
    
    session = sessions[session_id]
    question = chat_request.question
    
    if question.lower() in ["goodbye", "bye"]:
        if session_id in sessions:
            del sessions[session_id]
        return ChatResponse(response="Goodbye! Session ended.")
    
    # Attempt to answer from the PDF
    stuff_answer = session["stuff_chain"](
        {"input_documents": session["pages"], "question": question}, 
        return_only_outputs=True
    )
    answer_text = stuff_answer['output_text']

    if "answer not available in context" in answer_text.lower():
        # If answer not found in PDF, use the conversation chain
        response = session["conversation_chain"].run(input=question)
    else:
        # Clean the response text
        response = answer_text.strip()
        if response.startswith("AI: "):
            response = response.replace("AI: ", "")
        
        # Add to memory
        session["memory"].save_context({"input": question}, {"output": response})
        
        # Get conversation-aware response
        combined_prompt = f"User question: {question}\n\nAnswer from PDF: {response}\n\nProvide me the same response that I passed to you."
        response = session["conversation_chain"].run(input=combined_prompt)
    
    return ChatResponse(response=response)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}