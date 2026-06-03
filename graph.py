from typing import TypedDict, Dict, List, Any
import math

from langgraph.graph import StateGraph, START, END

from scanner import (
    extract_text_from_pdf,
    clean_resume_text,
    extract_resume_sections
)

from embedding_service import generate_embedding


class ResumeState(TypedDict):
    file_paths: List[str]
    job_description: Dict[str, str]

    resumes: List[Dict[str, Any]]
    job_embeddings: Dict[str, List[float]]

    ranked_resumes: List[Dict[str, Any]]
    best_resume: Dict[str, Any]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    magnitude_1 = math.sqrt(sum(a * a for a in vec1))
    magnitude_2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude_1 == 0 or magnitude_2 == 0:
        return 0.0

    return dot_product / (magnitude_1 * magnitude_2)


def extract_text_node(state: ResumeState) -> ResumeState:
    resumes = []

    for file_path in state["file_paths"]:
        raw_text = extract_text_from_pdf(file_path)

        resumes.append({
            "file_path": file_path,
            "raw_text": raw_text
        })

    state["resumes"] = resumes
    return state


def clean_text_node(state: ResumeState) -> ResumeState:
    for resume in state["resumes"]:
        resume["cleaned_text"] = clean_resume_text(
            resume["raw_text"]
        )

    return state


def extract_section_node(state: ResumeState) -> ResumeState:
    for resume in state["resumes"]:
        resume["sections"] = extract_resume_sections(
            resume["cleaned_text"]
        )

    return state


def generate_section_embeddings_node(state: ResumeState) -> ResumeState:
    for resume in state["resumes"]:
        section_embeddings = {}

        for section_name, content in resume["sections"].items():
            if content and content.strip():
                section_embeddings[section_name] = generate_embedding(content)

        resume["section_embeddings"] = section_embeddings

    return state


def generate_job_embedding_node(state: ResumeState) -> ResumeState:
    job_embeddings = {}

    for section_name, content in state["job_description"].items():
        if content and content.strip():
            job_embeddings[section_name] = generate_embedding(content)

    state["job_embeddings"] = job_embeddings
    return state


def calculate_score_node(state: ResumeState) -> ResumeState:
    ranked_resumes = []

    section_weights = {
        "skills": 0.40,
        "experience": 0.30,
        "projects": 0.20,
        "education": 0.10
    }

    for resume in state["resumes"]:
        section_embeddings = resume["section_embeddings"]

        total_score = 0.0
        total_weight = 0.0
        section_scores = {}

        for job_section, job_embedding in state["job_embeddings"].items():

            best_score = 0.0

            for resume_section, resume_embedding in section_embeddings.items():
                score = cosine_similarity(
                    job_embedding,
                    resume_embedding
                )

                best_score = max(best_score, score)

            weight = section_weights.get(job_section, 0.10)

            section_scores[job_section] = best_score

            total_score += best_score * weight
            total_weight += weight

        final_score = total_score / total_weight if total_weight else 0.0

        ranked_resumes.append({
            "file_path": resume["file_path"],
            "final_score": final_score,
            "section_scores": section_scores,
            "sections": resume["sections"]
        })

    ranked_resumes = sorted(
        ranked_resumes,
        key=lambda x: x["final_score"],
        reverse=True
    )

    state["ranked_resumes"] = ranked_resumes
    state["best_resume"] = ranked_resumes[0] if ranked_resumes else {}

    return state


workflow = StateGraph(ResumeState)

workflow.add_node("extract_text", extract_text_node)
workflow.add_node("clean_text", clean_text_node)
workflow.add_node("extract_sections", extract_section_node)
workflow.add_node("generate_section_embeddings", generate_section_embeddings_node)
workflow.add_node("generate_job_embedding", generate_job_embedding_node)
workflow.add_node("calculate_score", calculate_score_node)

workflow.add_edge(START, "extract_text")
workflow.add_edge("extract_text", "clean_text")
workflow.add_edge("clean_text", "extract_sections")
workflow.add_edge("extract_sections", "generate_section_embeddings")
workflow.add_edge("generate_section_embeddings", "generate_job_embedding")
workflow.add_edge("generate_job_embedding", "calculate_score")
workflow.add_edge("calculate_score", END)

app = workflow.compile()


if __name__ == "__main__":
    result = app.invoke({
        "file_paths": [
            "backend-developer.pdf",
            "resume.pdf"
        ],
        "job_description": {
            "education": "Bachelors of Technology",
            "skills": "Python, FastAPI, PostgreSQL, Docker, AI/ML",
            "experience": "2+ years of backend development experience",
            "projects": "Backend APIs, database integration, Docker-based deployment"
        }
    })

    print("\n===== Ranked Resumes =====\n")

    for index, resume in enumerate(result["ranked_resumes"], start=1):
        print(f"Rank #{index}")
        print(f"Resume: {resume['file_path']}")
        print(f"Final Score: {resume['final_score']:.4f}")

        print("Section Scores:")
        for section, score in resume["section_scores"].items():
            print(f"  {section}: {score:.4f}")

        print("-" * 50)

    print("\n===== Best Matching Resume =====")
    print(f"Resume: {result['best_resume']['file_path']}")
    print(f"Score: {result['best_resume']['final_score']:.4f}")