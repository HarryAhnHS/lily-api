from together import Together
import os
import json
from groq import Groq

# def get_together_client():
#     """Initialize Together client lazily to avoid startup errors when API key is missing."""
#     api_key = os.getenv("TOGETHER_API_KEY")
#     model = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")

#     if not api_key:
#         raise ValueError("TOGETHER_API_KEY environment variable is not set")
#     return Together(api_key=api_key)

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    
    return Groq(api_key=api_key)

model = os.getenv("GROQ_MODEL")

def generate_and_store_student_summary(supabase, student_id: str, user_id: str):
    print(f"✅ Generating and storing student summary for student {student_id}")
    # 1. Fetch latest 5 sessions
    sessions_res = (
        supabase.table("sessions")
        .select(
            "id, created_at, raw_input, memo, "
            "objectives(id, description, goal_id, subject_area_id, "
            "goals(title), subject_areas(name))"
        )
        .eq("student_id", student_id)    
        .eq("teacher_id", user_id)
        .order("created_at", desc=True)
        .limit(10)
        .execute()
    )

    # 2. Fetch all objectives
    objectives_res = (
        supabase.table("objectives")
        .select("id, description, reporting_frequency, goals(title), subject_areas(name)")
        .eq("student_id", student_id)
        .eq("teacher_id", user_id)
        .execute()
    )

    # 3. Fetch student
    student_res = (
        supabase.table("students")
        .select("*")
        .eq("id", student_id)
        .eq("teacher_id", user_id)
        .execute()
    )

    formatted_input = {
        "student": {
            "disability_type": student_res.data[0]["disability_type"],
            "grade_level": student_res.data[0]["grade_level"],
            "summary": student_res.data[0]["summary"],
        },
        "latest_sessions": [
            {
                "date": s["created_at"],
                "objective": s["objectives"]["description"] if s.get("objectives") else "Unknown",
                "subject_area": s["objectives"]["subject_areas"]["name"] if s.get("objectives") and s["objectives"].get("subject_areas") else "Unknown",
                "goal": s["objectives"]["goals"]["title"] if s.get("objectives") and s["objectives"].get("goals") else "Unknown",
                "memo": s.get("memo") or s.get("raw_input")
            }
            for s in sessions_res.data
        ],
        "objectives": [
            {
                "description": o["description"],
                "subject_area": o["subject_areas"]["name"] if o.get("subject_areas") else "Unknown",
                "goal": o["goals"]["title"] if o.get("goals") else "Unknown",
            }
            for o in objectives_res.data
        ]
    }

    prompt = f"""
        You are an IEP assistant tasked with writing a short (max 100 words) progress update about a student.

        The student's general information:
        - Grade Level: {formatted_input["student"]["grade_level"]}
        - Disability Type: {formatted_input["student"]["disability_type"]}

        Previous Summary (if available):
        {formatted_input["student"]["summary"]}

        Objectives the student is working on:
        {json.dumps(formatted_input["objectives"], indent=2)}

        Latest Sessions (chronological recent activities):
        {json.dumps(formatted_input["latest_sessions"], indent=2)}

        Write a natural short paragraph summarizing the student's progress. Use gender and name-neutral language.
        Mention trends, strengths, improvements, and progress towards goals. Keep it factual but positive.
        Make sure it is under 100 words.
    """

    summary = call_llm_student_summary(prompt)

    # 3. Store summary back in students table
    update_res = (
        supabase.table("students")
        .update({"summary": summary})
        .eq("id", student_id)
        .eq("teacher_id", user_id)
        .execute()
    )

    return summary


def call_llm_student_summary(prompt: str) -> str:
    try:
        client = get_groq_client()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes IEP progress summaries based on session logs and objectives."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=600,
        )

        content = response.choices[0].message.content.strip()
        return content

    except Exception as e:
        print("❌ LLM summary generation failed:", e)
        return "Unable to generate summary at this time."