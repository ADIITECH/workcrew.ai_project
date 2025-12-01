from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import os
import uuid
import json
from extractor import extract_details
from recommender import recommend_service, get_service_details, SERVICES
from memory_manager import Memory

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Recruitment Agency Sales Agent")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize memory
memory = Memory()

# Models
class MessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ServiceRecommendation(BaseModel):
    service_id: str
    name: str
    description: str
    price_range: str
    match_score: float
    reasoning: str

class ChatResponse(BaseModel):
    message_id: str
    content: str
    timestamp: str
    metadata: Dict[str, Any] = {}
    session_id: str
    service_recommendation: Optional[ServiceRecommendation] = None

# Routes
@app.get("/")
async def root():
    return {"message": "Recruitment Agency Sales Agent API"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: MessageRequest):
    # Generate session ID if not provided
    if not request.session_id:
        request.session_id = str(uuid.uuid4())
    
    # Save user message
    memory.add(request.session_id, "user", request.message)
    
    try:
        # Extract details
        extracted = extract_details(request.message)
        
        # Get service recommendation
        service_id = recommend_service(extracted)
        service = get_service_details(service_id)
        
        # Generate response
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        You are a friendly and professional sales assistant for a recruitment agency.
        
        Previous conversation:
        {memory.get_conversation(request.session_id, limit=5)}
        
        Client's hiring needs:
        {json.dumps(extracted, indent=2)}
        
        Recommended service: {service['name']}
        Service details: {service['description']}
        
        Write a helpful response that:
        1. Acknowledges their requirements
        2. Explains why the recommended service is a good fit
        3. Asks if they'd like more details or to proceed
        """
        
        response = model.generate_content(prompt).text
        
        # Save assistant response
        memory.add(request.session_id, "assistant", response)
        
        return {
            "message_id": str(uuid.uuid4()),
            "content": response,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": request.session_id,
            "metadata": {
                "extracted_details": extracted,
                "service_id": service_id
            },
            "service_recommendation": {
                "service_id": service_id,
                "name": service["name"],
                "description": service["description"],
                "price_range": service["price_range"],
                "match_score": 0.9,  # This could be calculated based on matching logic
                "reasoning": f"Recommended based on {', '.join(extracted.get('roles', []))} roles and urgency: {extracted.get('urgency', False)}"
            }
        }
        
    except Exception as e:
        error_msg = f"Error processing your request: {str(e)}"
        memory.add(request.session_id, "system", f"Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

@app.get("/services")
async def list_services():
    """List all available recruitment services"""
    return {"services": list(SERVICES.values())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
