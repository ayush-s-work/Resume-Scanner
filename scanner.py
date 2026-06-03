import fitz
import re
from pathlib import Path


FILE_PATH = "resume.pdf"


SECTION_ALIASES = {
    "summary": ["summary", "about", "about me", "profile", "objective", "career objective"],
    "experience": ["experience", "work experience", "professional experience", "employment"],
    "internship": ["internship", "internships", "training", "industrial training"],
    "skills": ["skills", "technical skills", "technologies", "tools", "tech stack"],
    "education": ["education", "academic background", "academics", "qualification"],
}


def extract_text_from_pdf(file_path: str) -> str:
    """Extract raw text from a PDF file."""

    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = ""

    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"

    return text.strip()


def clean_resume_text(raw_text: str) -> str:
    """Clean extra spaces and blank lines but keep line breaks."""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def normalize_heading(line: str) -> str:
    """Normalize a line to compare it with section headings."""

    line = line.lower().strip()
    line = re.sub(r"[:\-–—]", "", line)
    line = re.sub(r"\s+", " ", line)

    return line


def detect_section(line: str):
    """Detect whether a line is a resume section heading."""

    normalized_line = normalize_heading(line)

    for section_name, aliases in SECTION_ALIASES.items():
        if normalized_line in aliases:
            return section_name

    return None


def extract_resume_sections(cleaned_text: str) -> dict:
    """Extract categorized resume sections."""

    sections = {
        "summary": "",
        "experience": "",
        "internship": "",
        "skills": "",
        "education": "",
    }

    current_section = None

    for line in cleaned_text.split("\n"):
        detected_section = detect_section(line)

        if detected_section:
            current_section = detected_section
            continue

        if current_section:
            sections[current_section] += line + "\n"

    return {
        section: content.strip()
        for section, content in sections.items()
    }


def main():
    raw_text = extract_text_from_pdf(FILE_PATH)
    print("Raw text type:", type(raw_text))

    cleaned_text = clean_resume_text(raw_text)
    sections = extract_resume_sections(cleaned_text)

    print("\nExtracted Sections:")
    for section_name, content in sections.items():
        print(f"\n--- {section_name.upper()} ---")
        print(content if content else "Not found")


if __name__ == "__main__":
    main()