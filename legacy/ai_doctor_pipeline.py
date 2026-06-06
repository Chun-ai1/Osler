"""
ai_doctor_pipeline.py — compatibility shim
------------------------------------------
The real pipeline has been replaced by nexus_runner.py.
This file re-exports everything from nexus_runner so that any module
doing `from ai_doctor_pipeline import X` keeps working unchanged.
"""
from nexus_runner import *  # noqa: F401, F403
from nexus_runner import (
    ai_doctor_pipeline,
    run,
    extract_symptoms,
    assign_triage,
    detect_red_flags,
    allow_pills_gate,
    parse_duration_to_hours,
    load_symptom_db,
    normalize_history,
    HARD_BLOCK_OTC_SYMPTOMS,
    TRIAGE_FLOORS,
)