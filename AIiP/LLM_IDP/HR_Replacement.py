import os
import sys
import time
from typing import Any, List, Dict
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

class MyLogger(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self) :
        for f in self.files:
            f.flush()

class CandidateSummary(BaseModel):
    score: int = Field(description="The overall matching score of the candidate in the range 0 to 10.", default=5)
    strengths: List[str] = Field(description="A list of the candidate's strengths. Empty if none.", default=[])
    weaknesses: List[str] = Field(description="A list of the candidate's weaknesses. Empty if none.", default=[])
    summary: str = Field(description="A short summary of the candidate in 2-3 sentences.", default="")

# SUBTASK 1: Zaktualizowano domyślne parametry dla lepszego kontekstu
def load_and_process_cv(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    candidate_name = os.path.basename(pdf_path).replace(".pdf", "")
    for chunk in chunks:
        chunk.metadata["candidate"] = candidate_name

    print(f"Loaded {len(documents)} pages from {candidate_name}, split into {len(chunks)} chunks")
    return chunks

def load_multiple_cvs(cv_folder: str, size: int = 500, overlap: int = 50) -> Dict[str, List[Document]]:
    cv_documents: Dict[str, List[Document]] = {}

    if not os.path.exists(cv_folder):
        print(f"Folder {cv_folder} does not exist!")
        return cv_documents

    for filename in os.listdir(cv_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(cv_folder, filename)
            
            cv_documents[filename] = load_and_process_cv(path, chunk_size=size, chunk_overlap=overlap)

    return cv_documents

# SUBTASK 2
def create_vector_store(all_documents: List[Document]) -> VectorStore:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings
    )
    print(f"Created store with {len(all_documents)} chunks")
    return vector_store

# SUBTASK 3
def search_relevant_candidates(vector_store: VectorStore, job_description: str, k: int = 10) -> Dict[str, Dict[str, Any]]:
    results = vector_store.similarity_search_with_score(job_description, k=k)

    candidate_scores: Dict[str, Dict[str, Any]] = {}
    
    for doc, score in results:
        name = doc.metadata.get("candidate", "Unknown")
        if name not in candidate_scores:
            candidate_scores[name] = {"scores": [], "fragments": [], "avg_score": 0.0}
        
        candidate_scores[name]["scores"].append(score)
        candidate_scores[name]["fragments"].append(doc)

    for name in candidate_scores:
        scores = candidate_scores[name]["scores"]
        candidate_scores[name]["avg_score"] = sum(scores) / len(scores)

    return candidate_scores

# SUBTASK 4
def analyze_with_agent(candidate_name: str, cv_chunks: List[Document], job_description: str) -> CandidateSummary:
    # Używamy modelu Llama 3.3 przez Groq
    hr = init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq")

    system_prompt = (
        "You are an HR specialist looking to recruit a new candidate for the given position. "
        "You must analyze the provided Curriculum Vitae (CV) text and extract data into the specified JSON schema. "
        "Do not invent or hallucinate data. "
        f"Job description: {job_description}"
    )

    cv_text_whole = "\n".join([doc.page_content for doc in cv_chunks])
    prompt_input = f"{candidate_name}'s CV:\n{cv_text_whole}"

    chat_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])

    runnable_sequence = chat_prompt_template | hr.with_structured_output(schema=CandidateSummary)
    return runnable_sequence.invoke({"text": prompt_input})

def create_final_ranking(vectorstore: VectorStore, job_description: str, top_k: int = 5) -> List[Dict[str, Any]]:
    
    similar_candidates = search_relevant_candidates(vectorstore, job_description, k=25)
    ranking: List[Dict[str, Any]] = []
    
    for candidate_name, data in similar_candidates.items():
        candidate_summary = analyze_with_agent(candidate_name, data['fragments'], job_description)
        
        ranking.append({
            'name': candidate_name,
            'similarity_score': data['avg_score'],
            'llm_score': candidate_summary.score,
            'final_score': (candidate_summary.score / 10),
            'strengths': candidate_summary.strengths,
            'weaknesses': candidate_summary.weaknesses,
            'summary': candidate_summary.summary
        })

    ranking.sort(key=lambda x: x['llm_score'], reverse=True)
    return ranking[:top_k]

def log_ranking(ranking: List[Dict[str, Any]], params_info: str):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = f"ranking-{timestr}.log"
    with open(filename, "w", encoding="utf-8") as f:
        og_stdout = sys.stdout
        sys.stdout = MyLogger(sys.stdout, f)
        
        print(f"EXPERIMENT PARAMETERS: {params_info}")
        print("="*30)
        print("CANDIDATE RANKING")
        for i, candidate in enumerate(ranking, 1):
            print(f"#{i} - {candidate['name'].upper()}")
            print(f"   • LLM Score:            {candidate['llm_score']}/10")
            print(f"   • Similarity (RAG):     {candidate['similarity_score']:.2f}")
            print(f"\nSTRENGTHS: {', '.join(candidate['strengths'])}")
            print(f"AREAS FOR IMPROVEMENT: {', '.join(candidate['weaknesses'])}")
            print(f"SUMMARY: {candidate['summary']}\n")
        
        sys.stdout = og_stdout

def main():
    load_dotenv()
    job_description = """
    Position: Senior Python Developer
    Mandatory Requirements:
    - Minimum 5 years of experience in Python
    - Fluent knowledge of Django or Flask
    - Experience with Docker
    - Knowledge of SQL databases
    """

    cv_folder = "./cvs"
    
    
    chunk_size_exp = 1000
    k_exp = 10
    
    cv_documents = load_multiple_cvs(cv_folder, size=chunk_size_exp)
    
    all_docs = []
    for docs in cv_documents.values():
        all_docs.extend(docs)

    vectorstore = create_vector_store(all_docs)
    
    
    ranking = create_final_ranking(vectorstore, job_description)
    
    log_ranking(ranking, f"chunk_size={chunk_size_exp}, model=llama-3.3, k={k_exp} (search context increased)")

if __name__ == "__main__":
    main()