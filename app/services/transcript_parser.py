from pydantic import BaseModel
from typing import List
from groq import Groq
# from together import Together

import os
import json

from dotenv import load_dotenv
load_dotenv()


# ---------- Pydantic Models ----------
class TranscriptRequest(BaseModel):
    transcript: str

class ParsedSession(BaseModel):
    student_name: str
    objective_description: str
    memo: str

class MatchStudent(BaseModel):
    id: str
    name: str
    similarity: float
    summary: str
    disability_type: str
    grade_level: int
class SubjectArea(BaseModel):
    id: str
    name: str
class Goal(BaseModel):
    id: str
    title: str
class MatchObjective(BaseModel):
    id: str
    description: str
    similarity: float
    queried_objective_description: str
    objective_type: str
    target_accuracy: float
    subject_area: SubjectArea
    goal: Goal

class StudentWithObjectives(BaseModel):
    student: MatchStudent
    objectives: List[MatchObjective]

class ObjectiveProgress(BaseModel):
    trials_completed: int
    trials_total: int
class SuggestedSession(BaseModel):
    parsed_session_id: str
    raw_input: str
    memo: str
    objective_progress: ObjectiveProgress
    # student_suggestions: List[MatchStudent]
    # objective_suggestions: List[MatchObjective]
    matches: List[StudentWithObjectives]


# def get_together_client():
#     """Initialize Together client lazily to avoid startup errors when API key is missing."""
#     api_key = os.getenv("TOGETHER_API_KEY")
#     if not api_key:
#         raise ValueError("TOGETHER_API_KEY environment variable is not set")
#     return Together(api_key=api_key)

# model = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    
    return Groq(api_key=api_key)

model = os.getenv("GROQ_MODEL")

# ---------- LLM Calls ----------
def call_llm_extract_sessions(transcript: str, student_names: List[str] = None) -> List[dict]:
    student_names_text = ""
    if student_names and len(student_names) > 0:
        student_names_text = "The following are the actual student names in your system. Please use matches from this list when possible:\n"
        student_names_text += ", ".join(student_names)
        student_names_text += "\nIf no direct match is found from transcript and this list, use the inferred student name as is.\n"
    
    prompt = f"""
        You are an intelligent assistant that extracts structured session logs from raw notes or transcripts written by teachers. These logs are used to track IEP (Individualized Education Program) progress.

        Your job is to split the transcript into **individual session logs**, each representing a distinct activity, observation, or evaluation for a student. A single transcript may include multiple sessions, even for the same student or same objective — treat each meaningful unit as its own session.

        ---

        🧠 **Student Name Matching**:
        You may use the following list of known student names to guide your extraction:
        {", ".join(student_names) if student_names else "None provided"}

        However, do NOT skip sessions if names are not found in this list. Use your best judgment to extract a likely student name (e.g. "Johnny", "the student", "they") or leave it as `"student_name": null` if unknown.

        ---

        📌 For each session, extract:
        - `student_name`: Name of the student, or best guess. Use `null` if not clear.
        - `objective_description`: What the student was doing, phrased as a **third-person skill-based goal** (e.g. "Johnny is working on identifying main ideas in a passage.")
        - `memo`: What happened during this session, phrased as a **third-person observation or performance summary**.

        ---

        ❗ Guidelines:
        - Do NOT combine multiple sessions into one. Each line or paragraph that describes a different moment should be its own JSON object.
        - If sessions repeat the same student or objective, that’s okay — extract them separately.
        - NEVER return an empty list unless the transcript is truly just filler (e.g. "No sessions today").

        ---

        🎯 Respond ONLY with a **JSON list** like this:

        [
        {{
            "student_name": "Johnny",
            "objective_description": "Johnny is working on solving word problems with three-digit numbers.",
            "memo": "He independently solved 10 out of 12 correctly, needing some support with regrouping."
        }},
        {{
            "student_name": "Sara",
            "objective_description": "Sara is practicing using complete sentences in written responses.",
            "memo": "Sara used capital letters and periods in all 5 sentences today."
        }}
        ]

        Transcript:
        \"\"\"{transcript}\"\"\"
        """

    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You extract structured IEP session logs from transcripts."},
                {"role": "user", "content": prompt}
            ],
        )

        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("OpenAI returned an empty response")

        raw_output = response.choices[0].message.content.strip()

        print("groq session extraction prompt", prompt)
        print("raw_output", raw_output)

        if not raw_output:
            raise RuntimeError("OpenAI returned an empty string")

        try:
            parsed_output = json.loads(raw_output)
            if not isinstance(parsed_output, list):
                raise RuntimeError("OpenAI response is not a list")
            return parsed_output
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse OpenAI response as JSON: {str(e)}\nResponse content: {raw_output}")

    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {str(e)}")


def infer_trials_completed(
    transcript: str, 
    parsed_memo: str,
    student_name: str,
    student_disability_type: str,
    student_grade_level: int,
    student_summary: str,
    objective_description: str,
    objective_type: str, 
    target_accuracy: float
) -> dict:
    system_prompt = (
        "You are an assistant that extracts objective progress data from session logs for IEP tracking.\n"
        "Each session is a single activity or observation of a student.\n"
        "You will be given:\n"
        "- The raw transcript from the teacher\n"
        "- A summary of that session\n"
        "- Student metadata (name, grade, disability, profile summary)\n"
        "- Objective metadata (description, type, and target accuracy if applicable)\n"
        "\n"
        "Respond ONLY with JSON like this:\n"
        "{\n"
        "  \"trials_completed\": <int>,\n"
        "  \"trials_total\": <int>\n"
        "}\n"
        "\n"
        "If objective_type is 'binary', return 1/1 if the student clearly met the goal, or 0/1 if not.\n"
        "If objective_type is 'trial', infer numerator/denominator from test scores, percentages, or activity/observation performance metric.\n"
        "For example, 'scored 50%' = 50/100 or '12 out of 15 correct' = 12/15."
    )

    user_prompt = f"""
        Student Name: {student_name}
        Student Summary: {student_summary}

        Objective Description: {objective_description}
        Objective Type: {objective_type}
        {f"Target Accuracy: {target_accuracy * 100:.0f}%" if objective_type == "trial" else ""}

        Parsed Session Memo:
        {parsed_memo}

        Full Raw Transcript:
        \"\"\"{transcript}\"\"\"
            """

    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
        )
        content = response.choices[0].message.content.strip()

        return json.loads(content)

    except Exception as e:
        print("❌ Error inferring trials:", e)
        return {
            "trials_completed": 0,
            "trials_total": 0
        }
