# mirae-api

## Setup

### Environment Variables
Before running the application, make sure to set the following environment variables:

```bash
export TOGETHER_API_KEY=your_together_api_key_here
export TOGETHER_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo-Free  # Optional, defaults to this model
```

You can also create a `.env` file in the project root with these variables.

### Running the Application
```bash
uvicorn app.main:app --reload
```

# TODO 04/09/25

0. CRUD Edit and delete students, objectives, sessions.

1. Implement goals under each subject_area. goals should have title, description. 
    - Each goal has objectives.

2. objective_measure_type
- Each session is logged once a week - 1 data entry per week
- Each session needs a measure_type:
    1. trial based objectives (x/n)
    2. rating scale (performance 1 - 5)
    3. binary y/n - trial out of 1
    4. frequency

3. Student page/modal - show name, description, a list of objectives, and progress tracker later.

4. log frequency per month (notification and use to )
    - n per week/month

5. objective progress tracking 
- objective_completion (bool), objective_completed_at
- objective_completion_percentage, objective_completion_timeline
    - Each session needs a progress goal (8 out of 10 times) at what timeline to be marked as complete.
        - Progress goal over what timeline (ex. 90% of the time in one quarter)

6. changing validation form based on objective_type

7. progress tracking based on objective_type

6. Onboarding with oauth (name, school, role)

7. editable profile page 

8. cloud?

9. encryption

10. completed objective / goal -> suggest new objective / goal

11. annual / quarterly reports - LLM summary for student objectives based on logs