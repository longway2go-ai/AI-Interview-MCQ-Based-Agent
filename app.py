# app.py

import os
import io
import json
import random
import re
import tempfile
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import google.generativeai as genai
from groq import Groq
import openai
from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uvicorn

# ==== Initialization ====
app = FastAPI()
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

chroma_client = chromadb.Client(Settings(anonymized_telemetry=False, allow_reset=True))
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model


# ==== State Management ====
class QuizState:
    def __init__(self):
        self.reset()
        self.gemini_api_key = ""
        self.groq_api_key = ""
        self.openai_api_key = ""
        self.backend = "gemini"

    def reset(self):
        self.topics = []
        self.current_question_index = 0
        self.qa_history = []
        self.total_score = 0
        self.total_questions = 40
        self.questions_answered = 0
        self.quiz_ended_early = False
        self.current_topic = ""
        self.difficulty = "easy"
        self.current_question_data = None
        self.document_type = "resume"
        self.document_content = ""
        self.vector_collection = None
        self.document_chunks = []

state = QuizState()


# ==== LLM Functions ====
def llm_query(prompt, backend="gemini"):
    api_keys = {
        "gemini": state.gemini_api_key,
        "groq": state.groq_api_key,
        "openai": state.openai_api_key
    }
    
    api_key = api_keys.get(backend)
    if not api_key:
        raise Exception(f"{backend.capitalize()} API key not configured")
    
    try:
        if backend == "gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            return model.generate_content(prompt).text.strip()
        elif backend == "groq":
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        elif backend == "openai":
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"{backend.capitalize()} API Error: {str(e)}")


# ==== Vector Database (RAG) ====
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk.strip())
    return chunks


def create_vector_database(document_text, document_type):
    try:
        chroma_client.delete_collection(name="document_collection")
    except:
        pass
    
    collection = chroma_client.create_collection(
        name="document_collection",
        metadata={"document_type": document_type}
    )
    
    chunks = chunk_text(document_text, chunk_size=500, overlap=50)
    model = get_embedding_model()
    embeddings = model.encode(chunks).tolist()
    
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    state.vector_collection = collection
    state.document_chunks = chunks
    return collection


def retrieve_relevant_context(query, top_k=3):
    if state.vector_collection is None:
        return ""
    
    try:
        model = get_embedding_model()
        query_embedding = model.encode([query]).tolist()
        results = state.vector_collection.query(query_embeddings=query_embedding, n_results=top_k)
        return "\n\n".join(results['documents'][0])
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ""


# ==== Document Processing ====
def extract_text_from_file(file: UploadFile) -> str:
    if file.content_type == "application/pdf":
        pdf_bytes = file.file.read()
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join([page.extract_text() or "" for page in reader.pages]).strip()
    elif file.content_type.startswith("text/"):
        return file.file.read().decode("utf-8").strip()
    return ""


def extract_document_topics(document_text, document_type, backend):
    fallback_topics = {
        "resume": ["Technical Skills", "Work Experience", "Education", "Programming Languages", "Frameworks", "Database Management", "Problem Solving", "Project Management"],
        "research_paper": ["Research Methods", "Data Analysis", "Literature Review", "Methodology", "Statistical Analysis", "Hypothesis Testing", "Results Interpretation", "Academic Writing"],
        "technical_document": ["System Architecture", "Implementation", "API Design", "Best Practices", "Configuration", "Troubleshooting", "Documentation", "Technical Specifications"]
    }
    
    if document_type == "research_paper":
        context = retrieve_relevant_context("main topics, key concepts, research areas, methodologies", top_k=5)
        if context:
            document_text = context
    
    prompts = {
        "resume": "Extract 8-10 technical topics/skills for interview questions from this resume.",
        "research_paper": "Extract 8-10 main research topics/concepts/methodologies from this paper.",
        "technical_document": "Extract 8-10 main technical topics/technologies/processes from this document."
    }
    
    prompt = f"""{prompts.get(document_type, prompts["resume"])}
    
Document: {document_text[:3000]}...

Return ONLY a Python list: ['Topic 1', 'Topic 2', ...]"""
    
    try:
        topics_text = llm_query(prompt, backend)
        list_match = re.search(r'\[.*?\]', topics_text, re.DOTALL)
        if list_match:
            import ast
            parsed = ast.literal_eval(list_match.group(0))
            if isinstance(parsed, list) and len(parsed) >= 3:
                return [str(t).strip() for t in parsed if t and len(str(t).strip()) > 2][:10]
    except Exception as e:
        print(f"Topic extraction failed: {e}")
    
    return fallback_topics.get(document_type, fallback_topics["resume"])


def generate_single_mcq_question(topic, backend, difficulty="easy", document_type="resume"):
    difficulty_map = {"easy": "basic/fundamental", "moderate": "intermediate/practical", "hard": "advanced/expert"}
    
    additional_context = ""
    if document_type == "research_paper" and state.vector_collection:
        context = retrieve_relevant_context(f"{topic} concepts methodology findings", top_k=3)
        if context:
            additional_context = f"\n\nDocument context:\n{context[:1500]}"
    
    contexts = {
        "resume": f"Create a {difficulty_map[difficulty]} interview question about '{topic}'.",
        "research_paper": f"Create a {difficulty_map[difficulty]} question about '{topic}' from the research paper.{additional_context}",
        "technical_document": f"Create a {difficulty_map[difficulty]} technical question about '{topic}'."
    }
    
    prompt = f"""{contexts.get(document_type, contexts["resume"])}

Return ONLY JSON:
{{
    "question": "Your question",
    "options": ["A", "B", "C", "D"],
    "correct_answer": 0,
    "explanation": "Why correct"
}}"""
    
    try:
        response = llm_query(prompt, backend)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception as e:
        print(f"Question generation error: {e}")
    
    # Fallback question
    return {
        "question": f"What is the primary use of {topic} in this context?",
        "options": [f"Correct answer about {topic}", "Incorrect option 1", "Incorrect option 2", "Incorrect option 3"],
        "correct_answer": 0,
        "explanation": f"Tests {difficulty} understanding of {topic}."
    }


# ==== Routes ====
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/start_quiz")
async def start_quiz(
    request: Request,
    backend: str = Form("gemini"),
    api_key: str = Form(...),
    document_file: UploadFile = File(...),
    difficulty: str = Form("easy"),
    document_type: str = Form("resume"),
    quiz_type: str = Form("mcq")
):
    valid_types = ["resume", "research_paper", "technical_document"]
    if document_type not in valid_types:
        return JSONResponse({"error": f"Invalid document type. Must be one of: {valid_types}"}, status_code=400)
    
    if not api_key or len(api_key.strip()) < 10:
        return JSONResponse({"error": "Please provide a valid API key."}, status_code=400)
    
    # Store API key
    if backend == "gemini":
        state.gemini_api_key = api_key.strip()
    elif backend == "groq":
        state.groq_api_key = api_key.strip()
    elif backend == "openai":
        state.openai_api_key = api_key.strip()
    else:
        return JSONResponse({"error": "Invalid backend selected."}, status_code=400)
    
    # Extract document
    document_text = extract_text_from_file(document_file)
    if not document_text or len(document_text.strip()) < 50:
        return JSONResponse({"error": "Document is empty or too short."}, status_code=400)
    
    # Reset and configure
    state.reset()
    state.document_type = document_type
    state.document_content = document_text[:5000]
    state.backend = backend
    state.difficulty = difficulty
    
    # Re-store API key after reset
    if backend == "gemini":
        state.gemini_api_key = api_key.strip()
    elif backend == "groq":
        state.groq_api_key = api_key.strip()
    elif backend == "openai":
        state.openai_api_key = api_key.strip()
    
    # Create vector DB for research papers
    if document_type == "research_paper":
        try:
            create_vector_database(document_text, document_type)
        except Exception as e:
            print(f"Vector DB creation failed: {e}")
    
    # Extract topics
    try:
        state.topics = extract_document_topics(document_text, document_type, backend)
    except Exception as e:
        print(f"Using fallback topics: {e}")
        fallback = {
            "resume": ["Technical Skills", "Work Experience", "Problem Solving"],
            "research_paper": ["Research Methods", "Data Analysis", "Methodology"],
            "technical_document": ["System Architecture", "Implementation", "Best Practices"]
        }
        state.topics = fallback.get(document_type, fallback["resume"])
    
    rag_status = "enabled" if document_type == "research_paper" and state.vector_collection else "disabled"
    
    return JSONResponse({
        "topics": state.topics,
        "quiz_type": "mcq",
        "document_type": document_type,
        "rag_enabled": document_type == "research_paper",
        "vector_db_status": rag_status,
        "message": f"Quiz ready! RAG mode: {rag_status}"
    })


@app.post("/next_question")
async def next_question(prev_answer: str = Form("")):
    if state.current_question_index >= state.total_questions or not state.topics:
        return JSONResponse({"question": None, "topic": None})
    
    topic = random.choice(state.topics)
    state.current_topic = topic
    
    question_data = generate_single_mcq_question(topic, state.backend, state.difficulty, state.document_type)
    question_data['topic'] = topic
    question_data['question_id'] = state.current_question_index
    state.current_question_data = question_data
    
    return JSONResponse({
        "question": question_data["question"],
        "options": question_data["options"],
        "topic": topic,
        "question_type": "mcq",
        "question_number": state.current_question_index + 1,
        "total_questions": state.total_questions
    })


@app.post("/submit_mcq")
async def submit_mcq(selected_answer: int = Form(...)):
    if not state.current_question_data:
        return JSONResponse({"error": "No current question data"}, status_code=400)
    
    correct_answer = state.current_question_data["correct_answer"]
    is_correct = selected_answer == correct_answer
    score = 1 if is_correct else 0
    
    state.qa_history.append({
        "topic": state.current_question_data["topic"],
        "question": state.current_question_data["question"],
        "options": state.current_question_data["options"],
        "selected_answer": selected_answer,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
        "score": score,
        "explanation": state.current_question_data["explanation"]
    })
    
    state.total_score += score
    state.questions_answered += 1
    state.current_question_index += 1
    
    return JSONResponse({
        "is_correct": is_correct,
        "correct_answer": correct_answer,
        "explanation": state.current_question_data["explanation"],
        "score": score,
        "total_score": state.total_score,
        "max_possible_score": state.total_questions
    })


@app.post("/end_quiz")
async def end_quiz():
    state.quiz_ended_early = True
    return JSONResponse({"success": True})


@app.post("/tts")
async def tts(text: str = Form("")):
    # Placeholder for frontend compatibility - returns empty response
    return JSONResponse({"error": "TTS feature disabled"}, status_code=501)


@app.get("/final_score")
async def final():
    percentage_score = round((state.total_score / state.questions_answered) * 100, 2) if state.questions_answered > 0 else 0
    
    return JSONResponse({
        "final_score": state.total_score,
        "total_possible": state.total_questions,
        "questions_answered": state.questions_answered,
        "percentage_score": percentage_score,
        "completed_early": state.quiz_ended_early,
        "document_type": state.document_type,
        "history": state.qa_history
    })


@app.get("/download_results")
async def download_results():
    if not state.qa_history:
        return JSONResponse({"error": "No quiz data available"}, status_code=400)
    
    percentage_score = round((state.total_score / state.questions_answered) * 100, 2) if state.questions_answered > 0 else 0
    
    content = f"""QUIZ RESULTS REPORT
==================

QUIZ SUMMARY
-----------
Document Type: {state.document_type.replace('_', ' ').title()}
Difficulty: {state.difficulty.title()}
Backend: {state.backend}
RAG Mode: {'Yes' if state.document_type == 'research_paper' and state.vector_collection else 'No'}
Questions Answered: {state.questions_answered}/{state.total_questions}
Correct: {state.total_score} | Incorrect: {state.questions_answered - state.total_score}
Score: {percentage_score}%

QUESTIONS & ANSWERS
==================
"""
    
    for i, qa in enumerate(state.qa_history, 1):
        content += f"""
Q{i}: {qa['topic']}
{qa['question']}

A) {qa['options'][0]}
B) {qa['options'][1]}
C) {qa['options'][2]}
D) {qa['options'][3]}

Your Answer: {chr(65 + qa['selected_answer'])}. {qa['options'][qa['selected_answer']]}
Correct Answer: {chr(65 + qa['correct_answer'])}. {qa['options'][qa['correct_answer']]}
Result: {'✓ Correct' if qa['is_correct'] else '✗ Incorrect'}
Explanation: {qa['explanation']}

"""
    
    content += f"\nFinal Score: {state.total_score}/{state.questions_answered} ({percentage_score}%)\n"
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8')
    temp_file.write(content)
    temp_file.close()
    
    filename = f"quiz_results_{state.document_type}_{state.difficulty}.txt"
    return FileResponse(temp_file.name, media_type="text/plain", filename=filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
