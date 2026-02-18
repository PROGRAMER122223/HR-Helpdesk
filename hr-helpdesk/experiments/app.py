from fastapi import FastAPI,File, Form
from pydantic import BaseModel
from typing import List, Optional
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import Docx2txtLoader, DirectoryLoader
from langchain_google_vertexai import ChatVertexAI


# ---------------------------------------------------------
# 1. FASTAPI APP
# ---------------------------------------------------------
app = FastAPI(title="HR Policy Generator API")


# ---------------------------------------------------------
# 2. REQUEST & RESPONSE MODELS
# ---------------------------------------------------------
class GeneratePoliciesRequest(BaseModel):
    company_name: str
    company_size: str
    tone: str
    company_category: str
    country: str
    input_dir: str = "../docs/Company Policies"
    output_dir: str = "../generated_data3"
    glob_pattern: str = "*.docx"


class GeneratedPolicyInfo(BaseModel):
    source_docx: str
    output_markdown: str


class GeneratePoliciesResponse(BaseModel):
    company_name: str
    total_files_processed: int
    generated_policies: List[GeneratedPolicyInfo]


# ---------------------------------------------------------
# 3. PROMPT TEMPLATE
# ---------------------------------------------------------
policy_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an HR policy writer creating policies. "
     "You will receive an HR POLICY Template extracted from a DOCX file.\n"
     "Generate a NEW policy that follows the same structure/headings style, "
     "but write ORIGINAL content.\n"
     "Rules:\n"
     "- Do not copy long phrases verbatim from the template\n"
     "- Generate a concise and well-structured policy"),
    
    ("user",
     "TEMPLATE (reference):\n---\n{template_text}\n\n"
     "Generate policy:\n"
     "- company: {company_name}\n"
     "- company size: {company_size}\n"
     "- tone: {tone}\n"
     "- company category: {company_category}\n"
     "- country context: {country}\n"
     "Return only Markdown")
])


# ---------------------------------------------------------
# 4. INITIALIZE GEMINI MODEL & CHAIN
# ---------------------------------------------------------
llm = ChatVertexAI(
    model_name="gemini-2.5-flash-lite",
    temperature=0.2,
    max_output_tokens=2048,
)

chain = policy_prompt | llm


# ---------------------------------------------------------
# 5. HELPER: DOCX PATH → MARKDOWN PATH
# ---------------------------------------------------------
def combine_dir_with_markdown(dir_path: str, docx_path: str) -> str:
    filename = os.path.basename(docx_path)
    name_without_ext = os.path.splitext(filename)[0]
    markdown_name = name_without_ext.replace(" ", "_") + ".md"
    return os.path.join(dir_path, markdown_name)


# ---------------------------------------------------------
# 6. MAIN ENDPOINT
# ---------------------------------------------------------
@app.post("/generate-policies", response_model=GeneratePoliciesResponse)
def generate_policies(request: GeneratePoliciesRequest):
    # Ensure output directory exists
    os.makedirs(request.output_dir, exist_ok=True)

    # Load all DOCX files from the input directory
    directory_loader = DirectoryLoader(
        request.input_dir,
        glob=request.glob_pattern,
        loader_cls=Docx2txtLoader
    )
    docs = directory_loader.load()

    generated_info: List[GeneratedPolicyInfo] = []

    for doc in docs:
        # Use ONLY the current document's content
        template_text = doc.page_content

        # Build output markdown path
        source_path = doc.metadata["source"]
        output_path = combine_dir_with_markdown(request.output_dir, source_path)

        # Invoke LLM chain
        response = chain.invoke(
            {
                "template_text": template_text,
                "company_name": request.company_name,
                "company_size": request.company_size,
                "tone": request.tone,
                "company_category": request.company_category,
                "country": request.country,
            }
        )

        # Save generated markdown
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.content)

        generated_info.append(
            GeneratedPolicyInfo(
                source_docx=source_path,
                output_markdown=output_path
            )
        )

    return GeneratePoliciesResponse(
        company_name=request.company_name,
        total_files_processed=len(generated_info),
        generated_policies=generated_info
    )