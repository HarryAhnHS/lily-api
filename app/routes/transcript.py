from fastapi import APIRouter, Depends, HTTPException, Body
from uuid import uuid4
from app.dependencies.auth import user_supabase_client
from typing import List
from app.services.transcript_parser import (
    TranscriptRequest,
    ObjectiveProgress,
    ParsedSession,
    SuggestedSession,
    MatchStudent,
    MatchObjective,
    StudentWithObjectives,
    call_llm_extract_sessions,
    infer_trials_completed
)
from app.utils.semantic_matcher import top_k_semantic_matches

router = APIRouter()

@router.post("/analyze", response_model=List[SuggestedSession])
async def analyze_transcript_for_sessions(
    payload: TranscriptRequest,
    context=Depends(user_supabase_client)
):
    supabase = context["supabase"]
    teacher_id = context["user_id"]
    transcript = payload.transcript

    try:
        students_res = (
            supabase.table("students")
            .select("id, name, grade_level, disability_type, summary")
            .eq("teacher_id", teacher_id)
            .execute()
        ).data or []
        
        # Extract student names for the LLM to use
        student_names = [student["name"] for student in students_res]
        
        parsed_sessions = call_llm_extract_sessions(transcript, student_names)
        if not parsed_sessions:
            raise HTTPException(
                status_code=422,
                detail="No valid session data found in transcript. Try rephrasing or using manual form."
            )

        session_suggestions = []

        for item in parsed_sessions:
            try:
                parsed = ParsedSession(**item)
            except Exception as e:
                print("❌ Failed to parse session:", item)
                continue

            student_matches = top_k_semantic_matches(parsed.student_name, students_res, key="name", top_k=5)
            grouped_matches = []

            for student in student_matches:
                student_id = student["id"]

                obj_res = (
                    supabase.table("objectives")
                    .select("""
                        id, description, objective_type, target_accuracy, student_id,
                        goal:goals(id, title),
                        subject_area:subject_areas(id, name)
                    """)
                    .eq("teacher_id", teacher_id)
                    .eq("student_id", student_id)
                    .execute()
                )
                objectives = obj_res.data or []

                objective_matches = top_k_semantic_matches(
                    parsed.objective_description,
                    objectives,
                    key="description",
                    top_k=5
                )

                # Now we have results from semantic matcher, we create final objects
                full_student = next((s for s in students_res if s["id"] == student["id"]), None)
                if not full_student:
                    continue  # skip if student metadata missing

                student=MatchStudent(
                    id=student["id"],
                    name=student["name"],
                    similarity=student["similarity"],
                    summary=full_student.get("summary") or "",
                    disability_type=full_student.get("disability_type") or "",
                    grade_level=full_student.get("grade_level") or 0
                )

                objectives=[
                    MatchObjective(
                        id=o["id"],
                        description=o["description"],
                        similarity=o["similarity"],
                        queried_objective_description=parsed.objective_description,
                        objective_type=full_obj.get("objective_type", "trial"),
                        target_accuracy=float(full_obj.get("target_accuracy") or 1.0),
                        subject_area=full_obj.get("subject_area"),
                        goal=full_obj.get("goal")
                    )
                    for o in objective_matches
                    if (full_obj := next((obj for obj in objectives if obj["id"] == o["id"]), None))
                ]

                # Create final matches object for this student
                grouped_matches.append(StudentWithObjectives(
                    student=student,
                    objectives=objectives
                ))

            # Inferring progress using this student + objective
            # Use first available student + objective for inferring progress
            best_objective = None
            best_student = None
            for match in grouped_matches:
                if match.objectives:
                    best_objective = match.objectives[0]
                    best_student = match.student
                    break

            if best_objective:
                inferred = infer_trials_completed(
                    transcript=transcript,
                    parsed_memo=parsed.memo,
                    student_name=best_student.name,
                    student_disability_type=best_student.disability_type,
                    student_grade_level=best_student.grade_level,
                    student_summary=best_student.summary,
                    objective_description=best_objective.description,
                    objective_type=best_objective.objective_type,
                    target_accuracy=best_objective.target_accuracy
                )
                objective_progress = ObjectiveProgress(
                    trials_completed=inferred["trials_completed"],
                    trials_total=inferred["trials_total"]
                )
            else:
                objective_progress = ObjectiveProgress(trials_completed=0, trials_total=0)

            session_suggestions.append(SuggestedSession(
                parsed_session_id=str(uuid4()),
                raw_input=transcript,
                memo=parsed.memo,
                objective_progress=objective_progress,
                matches=grouped_matches
            ))

        return session_suggestions

    except HTTPException:
        # Re-raise HTTPExceptions to preserve status codes
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transcript analysis failed: {str(e)}")
    
# Sample prompt: John and Bobby both took the exact same math test today, and John got 15 out of 20 right. But bobby only got 50% right. John struggled with long division.
# [
#     {
#         "parsed_session_id": "899dbabd-18b3-42ef-9397-3826a889e16e",
#         "raw_input": "John and Bobby both took the exact same math test today, and John got 15 out of 20 right. But bobby only got 50% right. John struggled with long division.",
#         "memo": "John scored 15 out of 20, struggling with long division.",
#         "objective_progress": {
#             "trials_completed": 15,
#             "trials_total": 20
#         },
#         "matches": [
#             {
#                 "student": {
#                     "id": "a838dc9c-c7db-4f33-ac04-0aa90659d551",
#                     "name": "John",
#                     "similarity": 1.0,
#                     "summary": "",
#                     "disability_type": "adas",
#                     "grade_level": 10
#                 },
#                 "objectives": [
#                     {
#                         "id": "59389ed9-0692-4c15-9425-34be4c3f4544",
#                         "description": "Get 80% or more on his math assessment.",
#                         "similarity": 0.5927335619926453,
#                         "queried_objective_description": "John is working on taking a math test.",
#                         "objective_type": "trial",
#                         "target_accuracy": 0.8,
#                         "subject_area": {
#                             "id": "66833d1e-af00-4343-8e23-0a26f3ae5020",
#                             "name": "Subject 5"
#                         },
#                         "goal": {
#                             "id": "77105fd8-1971-4a5b-875a-dcd83c52a78e",
#                             "title": "Goal 5 for subject 5"
#                         }
#                     },
#                     {
#                         "id": "3378a707-4044-4c8e-a5c3-4a8de56df87e",
#                         "description": "John will pronounce his R's correctly with more than 80% accuracy across 4/5 entries.",
#                         "similarity": 0.42731910943984985,
#                         "queried_objective_description": "John is working on taking a math test.",
#                         "objective_type": "trial",
#                         "target_accuracy": 0.8,
#                         "subject_area": {
#                             "id": "fe9f71cf-2dbf-4d2d-a3e2-6eef221995ec",
#                             "name": "Speech / Language"
#                         },
#                         "goal": {
#                             "id": "013b2610-34db-4b41-a31a-682276485ecd",
#                             "title": "Improve Speaking and Pronounciation"
#                         }
#                     }
#                 ]
#             },
#             {
#                 "student": {
#                     "id": "2e678888-5c54-4e5a-a843-a19e166b6932",
#                     "name": "Bobby",
#                     "similarity": 0.3810436725616455,
#                     "summary": "",
#                     "disability_type": "",
#                     "grade_level": 12
#                 },
#                 "objectives": [
#                     {
#                         "id": "59b7ea2f-33d0-430c-b120-8fa66b1fb07e",
#                         "description": "Bobby should get 80% or higher on a simple arithmetic math exam, over 4 out of 5 trials. ",
#                         "similarity": 0.4619380235671997,
#                         "queried_objective_description": "John is working on taking a math test.",
#                         "objective_type": "trial",
#                         "target_accuracy": 0.8,
#                         "subject_area": {
#                             "id": "40e274a1-7bca-456f-bd13-a7a38f0cfcc6",
#                             "name": "Academic"
#                         },
#                         "goal": {
#                             "id": "2698a8fc-a7be-4dd7-abff-aa466f8a105b",
#                             "title": "Mathematics and Computation"
#                         }
#                     },
#                     {
#                         "id": "43e0cfaf-fd81-4092-bbc4-f150197bb5b5",
#                         "description": "Put on coat successfully without help 4 out of 5 times ",
#                         "similarity": 0.1035725474357605,
#                         "queried_objective_description": "John is working on taking a math test.",
#                         "objective_type": "trial",
#                         "target_accuracy": 1.0,
#                         "subject_area": {
#                             "id": "66833d1e-af00-4343-8e23-0a26f3ae5020",
#                             "name": "Subject 5"
#                         },
#                         "goal": {
#                             "id": "8489cbc5-1ba9-4bc5-897c-9bc40eeffedc",
#                             "title": "Goal 5"
#                         }
#                     }
#                 ]
#             },
#             {
#                 "student": {
#                     "id": "1b7faeb3-b3b3-4ee8-a7bd-6899c27d9333",
#                     "name": "Julian Danna",
#                     "similarity": 0.3616858124732971,
#                     "summary": "",
#                     "disability_type": "Other Health Impaired",
#                     "grade_level": 10
#                 },
#                 "objectives": [
#                     {
#                         "id": "c52b8d42-a4d5-450f-99b4-8590495e60c5",
#                         "description": "Improve math computation",
#                         "similarity": 0.3709166646003723,
#                         "queried_objective_description": "John is working on taking a math test.",
#                         "objective_type": "trial",
#                         "target_accuracy": 1.0,
#                         "subject_area": {
#                             "id": "40e274a1-7bca-456f-bd13-a7a38f0cfcc6",
#                             "name": "Academic"
#                         },
#                         "goal": {
#                             "id": "3d5e70f4-7476-447e-bc54-e1006b66eeac",
#                             "title": "Successfully complete academic course requirements"
#                         }
#                     },
#                     {
#                         "id": "2dcec03a-19f3-4ed3-8c2a-f0ac21f91cae",
#                         "description": "Improve work habits and study skills",
#                         "similarity": 0.26136839389801025,
#                         "queried_objective_description": "John is working on taking a math test.",
#                         "objective_type": "trial",
#                         "target_accuracy": 1.0,
#                         "subject_area": {
#                             "id": "f55639e5-26a8-47bd-8739-24641a2e1c88",
#                             "name": "Study Skills"
#                         },
#                         "goal": {
#                             "id": "84a214dd-7cfc-4404-973a-2a93b67ddd92",
#                             "title": "Julian will Maintain and Improve Study Skill Levels"
#                         }
#                     },
#                     {
#                         "id": "bce94e4f-f00c-42d4-9697-f29ca5c815ae",
#                         "description": "Organize material including classwork, contact major assignments, and homework",
#                         "similarity": 0.19801855087280273,
#                         "queried_objective_description": "John is working on taking a math test.",
#                         "objective_type": "trial",
#                         "target_accuracy": 1.0,
#                         "subject_area": {
#                             "id": "f55639e5-26a8-47bd-8739-24641a2e1c88",
#                             "name": "Study Skills"
#                         },
#                         "goal": {
#                             "id": "84a214dd-7cfc-4404-973a-2a93b67ddd92",
#                             "title": "Julian will Maintain and Improve Study Skill Levels"
#                         }
#                     }
#                 ]
#             }
#         ]
#     },
#     {
#         "parsed_session_id": "8a16fe4a-2386-4cb9-8d76-2089a26177ec",
#         "raw_input": "John and Bobby both took the exact same math test today, and John got 15 out of 20 right. But bobby only got 50% right. John struggled with long division.",
#         "memo": "Bobby scored 50% right.",
#         "objective_progress": {
#             "trials_completed": 10,
#             "trials_total": 20
#         },
#         "matches": [
#             {
#                 "student": {
#                     "id": "2e678888-5c54-4e5a-a843-a19e166b6932",
#                     "name": "Bobby",
#                     "similarity": 1.0,
#                     "summary": "",
#                     "disability_type": "",
#                     "grade_level": 12
#                 },
#                 "objectives": [
#                     {
#                         "id": "59b7ea2f-33d0-430c-b120-8fa66b1fb07e",
#                         "description": "Bobby should get 80% or higher on a simple arithmetic math exam, over 4 out of 5 trials. ",
#                         "similarity": 0.7604694962501526,
#                         "queried_objective_description": "Bobby is working on taking a math test.",
#                         "objective_type": "trial",
#                         "target_accuracy": 0.8,
#                         "subject_area": {
#                             "id": "40e274a1-7bca-456f-bd13-a7a38f0cfcc6",
#                             "name": "Academic"
#                         },
#                         "goal": {
#                             "id": "2698a8fc-a7be-4dd7-abff-aa466f8a105b",
#                             "title": "Mathematics and Computation"
#                         }
#                     },
#                     {
#                         "id": "43e0cfaf-fd81-4092-bbc4-f150197bb5b5",
#                         "description": "Put on coat successfully without help 4 out of 5 times ",
#                         "similarity": 0.09715627878904343,
#                         "queried_objective_description": "Bobby is working on taking a math test.",
#                         "objective_type": "trial",
#                         "target_accuracy": 1.0,
#                         "subject_area": {
#                             "id": "66833d1e-af00-4343-8e23-0a26f3ae5020",
#                             "name": "Subject 5"
#                         },
#                         "goal": {
#                             "id": "8489cbc5-1ba9-4bc5-897c-9bc40eeffedc",
#                             "title": "Goal 5"
#                         }
#                     }
#                 ]
#             },
#             {
#                 "student": {
#                     "id": "a838dc9c-c7db-4f33-ac04-0aa90659d551",
#                     "name": "John",
#                     "similarity": 0.3810437023639679,
#                     "summary": "",
#                     "disability_type": "adas",
#                     "grade_level": 10
#                 },
#                 "objectives": [
#                     {
#                         "id": "59389ed9-0692-4c15-9425-34be4c3f4544",
#                         "description": "Get 80% or more on his math assessment.",
#                         "similarity": 0.5445058941841125,
#                         "queried_objective_description": "Bobby is working on taking a math test.",
#                         "objective_type": "trial",
#                         "target_accuracy": 0.8,
#                         "subject_area": {
#                             "id": "66833d1e-af00-4343-8e23-0a26f3ae5020",
#                             "name": "Subject 5"
#                         },
#                         "goal": {
#                             "id": "77105fd8-1971-4a5b-875a-dcd83c52a78e",
#                             "title": "Goal 5 for subject 5"
#                         }
#                     },
#                     {
#                         "id": "3378a707-4044-4c8e-a5c3-4a8de56df87e",
#                         "description": "John will pronounce his R's correctly with more than 80% accuracy across 4/5 entries.",
#                         "similarity": 0.22091807425022125,
#                         "queried_objective_description": "Bobby is working on taking a math test.",
#                         "objective_type": "trial",
#                         "target_accuracy": 0.8,
#                         "subject_area": {
#                             "id": "fe9f71cf-2dbf-4d2d-a3e2-6eef221995ec",
#                             "name": "Speech / Language"
#                         },
#                         "goal": {
#                             "id": "013b2610-34db-4b41-a31a-682276485ecd",
#                             "title": "Improve Speaking and Pronounciation"
#                         }
#                     }
#                 ]
#             },
#             {
#                 "student": {
#                     "id": "d61cb8f0-d9b3-4e4b-9e29-629a9a9dea1d",
#                     "name": "Jayden Latimer",
#                     "similarity": 0.3809165358543396,
#                     "summary": "",
#                     "disability_type": "Communication Impairment (CI)",
#                     "grade_level": 1
#                 },
#                 "objectives": [
#                     {
#                         "id": "e9726fca-16d9-4bfa-a921-6ea5c87d6486",
#                         "description": "Given when asked by the teacher, Jayden will identify numbers up to 10 by naming, on 4 out of 5 trials, as evaluated/determined by Formal/Informal Assessments At Opportunity, by 04/26/2024.",
#                         "similarity": 0.35215622186660767,
#                         "queried_objective_description": "Bobby is working on taking a math test.",
#                         "objective_type": "trial",
#                         "target_accuracy": 1.0,
#                         "subject_area": {
#                             "id": "043b0c56-b7ae-477d-a72e-3506554b1272",
#                             "name": "Mathematics"
#                         },
#                         "goal": {
#                             "id": "37679c4a-958b-4d6f-8b87-89eff55b9589",
#                             "title": "Given specialized instruction with growth and improvement in math skills, Jayden will understand counting and cardinality, at 80% mastery, as evaluated/determined by Data Collection Quarterly, by 04/26/2024."
#                         }
#                     },
#                     {
#                         "id": "7c67ea66-23b1-49e1-a4ea-bc47e600ab6e",
#                         "description": "Given adaptive materials as needed (pencil grip or small pencil, slant board, high visual contrast) and demonstration, Jayden will trace the letters in his name, with 90% accuracy and proper formation, as evaluated/determined by Observation Twice a Month, and Class Work Quarterly, by 04/26/2024.",
#                         "similarity": 0.3391943573951721,
#                         "queried_objective_description": "Bobby is working on taking a math test.",
#                         "objective_type": "trial",
#                         "target_accuracy": 1.0,
#                         "subject_area": {
#                             "id": "891a8a6f-6fdc-470d-b15d-f7aefadffc6d",
#                             "name": "Occupational Therapy"
#                         },
#                         "goal": {
#                             "id": "41fbcd82-f00a-4dc0-8581-ec7f82835658",
#                             "title": "Given adaptive materials and minimal cues, Jayden will improve bilateral integration and visual motor coordination, as needed for pre-writing and use of classroom materials, as evaluated/determined by Observation Twice a Month, and Class Work Quarterly, by 04/26/2024."
#                         }
#                     },
#                     {
#                         "id": "544738ff-0a22-494e-9e1f-22cc47e7a750",
#                         "description": "Given when asked by the teacher, Jayden will count orally to 20, on 4 out of 5 trials, as evaluated/determined by Formal/Informal Assessments At Opportunity, by 04/26/2024.",
#                         "similarity": 0.31678903102874756,
#                         "queried_objective_description": "Bobby is working on taking a math test.",
#                         "objective_type": "trial",
#                         "target_accuracy": 1.0,
#                         "subject_area": {
#                             "id": "043b0c56-b7ae-477d-a72e-3506554b1272",
#                             "name": "Mathematics"
#                         },
#                         "goal": {
#                             "id": "37679c4a-958b-4d6f-8b87-89eff55b9589",
#                             "title": "Given specialized instruction with growth and improvement in math skills, Jayden will understand counting and cardinality, at 80% mastery, as evaluated/determined by Data Collection Quarterly, by 04/26/2024."
#                         }
#                     }
#                 ]
#             }
#         ]
#     }
# ]