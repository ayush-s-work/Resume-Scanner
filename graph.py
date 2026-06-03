from typing import TypedDict, Dict, List

from langgraph.graph import StateGraph, START, END 

from scanner import extract_text_from_pdf, clean_resume_text, extract_resume_sections
from embedding_service import generate_embedding


class ResumeState(TypedDict):
    file_path: str
    job_description: str
    raw_text: str
    cleaned_text: str
    sections: Dict[str, str]
    section_embeddings: Dict[str, List[float]]
    job_embedding: List[float]


def extract_text_node(state: ResumeState) -> ResumeState:
    raw_text = extract_text_from_pdf(state["file_path"])
    state["raw_text"] = raw_text
    return state


def clean_text_node(state: ResumeState) -> ResumeState:
    cleaned_text = clean_resume_text(state["raw_text"])
    state["cleaned_text"] = cleaned_text
    return state

def extract_section_node(state: ResumeState) -> ResumeState:
    sections = extract_resume_sections(state["cleaned_text"])
    state["sections"] = sections
    return state

def generate_section_embeddings_node(state: ResumeState) -> ResumeState:
    section_embeddings = {}

    for section_name, content in state["sections"].items():
        if content.strip():
            section_embeddings[section_name] = generate_embedding(content)

    state["section_embeddings"] = section_embeddings
    return state


def generate_job_embedding_node(state: ResumeState) -> ResumeState:
    state["job_embedding"] = generate_embedding(state["job_description"])
    return state

workflow = StateGraph(ResumeState)

workflow.add_node("extract_text", extract_text_node)
workflow.add_node("clean_text", clean_text_node)
workflow.add_node("extract_sections", extract_section_node)
workflow.add_node("generate_section_embeddings", generate_section_embeddings_node)
workflow.add_node("generate_job_embedding", generate_job_embedding_node)


workflow.add_edge(START, "extract_text")
workflow.add_edge("extract_text", "clean_text")
workflow.add_edge("clean_text", "extract_sections")
workflow.add_edge("extract_sections", "generate_section_embeddings")
workflow.add_edge("generate_section_embeddings", "generate_job_embedding")
workflow.add_edge("generate_job_embedding", END)

app = workflow.compile()


if __name__ == "__main__":
    result = app.invoke({
    "file_path": "resume_ayush.pdf",
    "job_description": "Looking for a Python backend developer with FastAPI, PostgreSQL, Docker and API development experience.",
    })

    print("Sections found:")
    print(result["sections"].keys())

    print("\nEmbeddings generated for:")
    print(result["section_embeddings"].values())

    print("\nJob embedding length:")
    print(result["job_embedding"])




