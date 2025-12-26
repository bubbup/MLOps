from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama  
from langchain_core.tools import tool
import json
from datetime import datetime, timedelta
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ════════════════════════════════════════════════════════════════════════════
# LLM CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:latest")

print(f"\n LLM Configuration: {OLLAMA_MODEL}")
print(f"   Base URL: {OLLAMA_BASE_URL}\n")

try:
    llm_researcher = ChatOllama(
        model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
        temperature=0.7, top_p=0.9, num_predict=2048, timeout=120
    )
    llm_scheduler = ChatOllama(
        model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
        temperature=0.5, top_p=0.9, num_predict=2048, timeout=120
    )
    llm_validator = ChatOllama(
        model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
        temperature=0.3, top_p=0.9, num_predict=1024, timeout=120
    )
    LLM_AVAILABLE = True
except Exception as e:
    print(f"  LLM unavailable: {str(e)[:50]}")
    LLM_AVAILABLE = False
    llm_researcher = llm_scheduler = llm_validator = None

# STATE DEFINITION

class PlannerState(TypedDict):
    """Central state - persists across all nodes"""
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    course_info: dict
    study_plan: dict
    validation_feedback: str
    plan_status: Literal["pending", "analyzed", "pending_validation", "validated", "needs_revision"]
    revision_count: int

# EXECUTION LOGGER
class ExecutionLogger:
    def __init__(self):
        self.lines = []
        self.start_time = datetime.now()

    def log(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.lines.append(f"[{ts}] [{elapsed:>6.2f}s] {msg}")

    def save(self, path="study_planner_execution_log.txt"):
        with open(path, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("       LANGGRAPH MULTI-AGENT STUDY PLANNER - EXECUTION LOG\n")
            f.write("\n".join(self.lines) + "\n")
            f.write(f"Total execution time: {(datetime.now() - self.start_time).total_seconds():.2f}s\n")
            f.write("="*80 + "\n")
        print(f"\n Execution log saved to: {path}")

logger = ExecutionLogger()

# TOOLS

@tool
def analyze_course_content(course_name: str, topics: list) -> dict:
    """Analyzes course structure"""
    analysis = {
        "course": course_name,
        "total_topics": len(topics),
        "estimated_hours": 0,
        "difficulty_levels": {"Beginner": 0, "Intermediate": 0, "Advanced": 0},
        "topics_breakdown": {}
    }
    
    for i, topic in enumerate(topics):
        difficulty = ["Beginner", "Intermediate", "Advanced"][i % 3]
        hours = 2 + (i % 4)
        analysis["estimated_hours"] += hours
        analysis["difficulty_levels"][difficulty] += 1
        analysis["topics_breakdown"][topic] = {
            "difficulty": difficulty,
            "hours": hours
        }
    
    logger.log(f"TOOL | analyze_course_content | course={course_name} | topics={len(topics)} | hours={analysis['estimated_hours']}")
    return analysis

@tool
def create_study_schedule(course_name: str, total_hours: int, study_days: int) -> dict:
    """Creates optimized study schedule"""
    daily_hours = total_hours / study_days
    schedule = {
        "course": course_name,
        "estimated_hours": total_hours,
        "total_duration_days": study_days,
        "daily_commitment_hours": round(daily_hours, 2),
        "daily_breakdown": {}
    }
    
    start_date = datetime.now()
    for day in range(study_days):
        date = start_date + timedelta(days=day)
        schedule["daily_breakdown"][date.strftime("%A, %B %d")] = {
            "study_hours": round(daily_hours, 2),
            "break_intervals": "25min focus / 5min break",
            "recommended_time": "Morning (9 AM - 12 PM)" if day % 2 == 0 else "Evening (6 PM - 9 PM)"
        }
    
    logger.log(f"TOOL | create_study_schedule | days={study_days} | daily_hours={round(daily_hours, 2)} | total_hours={total_hours}")
    return schedule

@tool
def validate_study_plan(plan: dict) -> dict:
    """Validates plan feasibility"""
    feedback = {
        "is_feasible": True,
        "issues": [],
        "recommendations": [],
        "score": 85
    }
    
    daily_hours = plan.get("daily_commitment_hours", 0)
    
    if daily_hours > 10:
        feedback["is_feasible"] = False
        feedback["issues"].append("Daily hours exceeds 10 - unrealistic")
        feedback["score"] = 30
    elif daily_hours > 6:
        feedback["recommendations"].append("Consider reducing daily hours for sustainability")
        feedback["score"] = 75
    else:
        feedback["score"] = 95
        feedback["recommendations"].append("Schedule is realistic and well-balanced")
    
    if not plan.get("daily_breakdown"):
        feedback["is_feasible"] = False
        feedback["issues"].append("Missing daily breakdown")
        feedback["score"] = 20
    
    logger.log(f"TOOL | validate_study_plan | feasible={feedback['is_feasible']} | score={feedback['score']} | daily_hours={daily_hours}")
    return feedback

# NODES 

def researcher_node(state: PlannerState) -> dict:
    """Analyze course content"""
    logger.log(f"NODE ENTRY | researcher | status={state.get('plan_status')} | topics={len(state.get('course_info', {}).get('topics', []))}")
    
    course_data = state.get("course_info", {})
    analysis = analyze_course_content.invoke({
        "course_name": course_data.get("name", "Unknown"),
        "topics": course_data.get("topics", [])
    })
    
    response = f" Analyzed: {analysis['course']} ({analysis['total_topics']} topics, ~{analysis['estimated_hours']} hours)"
    
    if LLM_AVAILABLE and llm_researcher:
        try:
            logger.log(f"LLM CALL | researcher | requesting learning order suggestions")
            llm_result = llm_researcher.invoke(
                f"Briefly suggest best learning order for: {', '.join(course_data.get('topics', [])[:3])}"
            )
            response += f"\n AI: {llm_result.content[:100]}"
            logger.log(f"LLM RESPONSE | researcher | chars={len(llm_result.content)}")
        except Exception as e:
            logger.log(f"LLM ERROR | researcher | {str(e)[:50]}")
    
    # ✅ Full state merge
    updated_state = {
        **state,  # Preserve all existing fields
        "messages": state["messages"] + [AIMessage(content=response)],
        "course_info": {**state["course_info"], **analysis},
        "plan_status": "analyzed"
    }
    
    logger.log(f"NODE EXIT  | researcher | status={updated_state['plan_status']} | est_hours={analysis['estimated_hours']}")
    return updated_state

def scheduler_node(state: PlannerState) -> dict:
    """Create study schedule"""
    logger.log(f"NODE ENTRY | scheduler | status={state.get('plan_status')} | revision={state.get('revision_count', 0)}")
    
    course_info = state.get("course_info", {})
    hours = course_info.get("estimated_hours", 20)
    
    schedule = create_study_schedule.invoke({
        "course_name": course_info.get("course", "Study Plan"),
        "total_hours": hours,
        "study_days": 14
    })
    
    response = f"📅 Schedule: {schedule['total_duration_days']} days @ {schedule['daily_commitment_hours']} hrs/day"
    
    if LLM_AVAILABLE and llm_scheduler:
        try:
            logger.log(f"LLM CALL | scheduler | requesting study tips")
            llm_result = llm_scheduler.invoke(
                f"Give 2 practical tips for studying {schedule['daily_commitment_hours']} hours per day"
            )
            response += f"\n Tips: {llm_result.content[:100]}"
            logger.log(f"LLM RESPONSE | scheduler | chars={len(llm_result.content)}")
        except Exception as e:
            logger.log(f"LLM ERROR | scheduler | {str(e)[:50]}")
    
    # ✅ Full state merge
    updated_state = {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "study_plan": schedule,
        "plan_status": "pending_validation"
    }
    
    logger.log(f"NODE EXIT  | scheduler | status={updated_state['plan_status']} | days={schedule['total_duration_days']}")
    return updated_state

def validator_node(state: PlannerState) -> dict:
    """Validate plan feasibility"""
    logger.log(f"NODE ENTRY | validator | status={state.get('plan_status')} | has_plan={bool(state.get('study_plan'))}")
    
    plan = state.get("study_plan", {})
    
    if not plan:
        logger.log(f"NODE ERROR | validator | No study plan to validate")
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content="❌ ERROR: No plan to validate")],
            "validation_feedback": "{}",
            "plan_status": "needs_revision"
        }
    
    feedback = validate_study_plan.invoke({"plan": plan})
    
    # Determine next status
    next_status = "validated" if feedback['is_feasible'] else "needs_revision"
    response = f" Validation: {next_status.upper()} (Score: {feedback['score']}/100)"
    
    if feedback['issues']:
        response += f"\n  Issues: {feedback['issues'][0]}"
    if feedback['recommendations']:
        response += f"\n Notes: {feedback['recommendations'][0]}"
    
    # ✅ Handle revision count increment HERE (not in routing)
    new_revision_count = state.get("revision_count", 0)
    if next_status == "needs_revision":
        new_revision_count += 1
        logger.log(f"VALIDATOR | Plan needs revision | revision_count={new_revision_count}")
    
    # ✅ Full state merge
    updated_state = {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "validation_feedback": json.dumps(feedback),
        "plan_status": next_status,
        "revision_count": new_revision_count
    }
    
    logger.log(f"NODE EXIT  | validator | status={updated_state['plan_status']} | score={feedback['score']} | revision={new_revision_count}")
    return updated_state

# ROUTING 

def route_from_start(state: PlannerState) -> Literal["researcher", "end"]:
    logger.log(f"ROUTE | START → researcher")
    return "researcher"

def route_after_researcher(state: PlannerState) -> Literal["scheduler", "end"]:
    status = state.get("plan_status")
    if status == "analyzed":
        logger.log(f"ROUTE | researcher → scheduler | status={status}")
        return "scheduler"
    logger.log(f"ROUTE | researcher → END | status={status} (error)")
    return "end"

def route_after_scheduler(state: PlannerState) -> Literal["validator", "end"]:
    status = state.get("plan_status")
    if status == "pending_validation":
        logger.log(f"ROUTE | scheduler → validator | status={status}")
        return "validator"
    logger.log(f"ROUTE | scheduler → END | status={status} (error)")
    return "end"

def route_after_validator(state: PlannerState) -> Literal["scheduler", "end"]:
    status = state.get("plan_status")
    revisions = state.get("revision_count", 0)
    
    if status == "validated":
        logger.log(f"ROUTE | validator → END | status={status} (SUCCESS)")
        return "end"
    elif status == "needs_revision" and revisions < 2:
        logger.log(f"ROUTE | validator → scheduler | status={status} | revision={revisions}")
        return "scheduler"
    else:
        logger.log(f"ROUTE | validator → END | status={status} | revision={revisions} (max reached)")
        return "end"

# GRAPH CONSTRUCTION
def create_study_planner_graph():
    """Build the LangGraph workflow"""
    builder = StateGraph(PlannerState)
    
    builder.add_node("researcher", researcher_node)
    builder.add_node("scheduler", scheduler_node)
    builder.add_node("validator", validator_node)
    
    builder.add_conditional_edges(START, route_from_start, {"researcher": "researcher", "end": END})
    builder.add_conditional_edges("researcher", route_after_researcher, {"scheduler": "scheduler", "end": END})
    builder.add_conditional_edges("scheduler", route_after_scheduler, {"validator": "validator", "end": END})
    builder.add_conditional_edges("validator", route_after_validator, {"scheduler": "scheduler", "end": END})
    
    return builder.compile()

# EXECUTION
def run_study_planner():
    graph = create_study_planner_graph()
    
    initial_state = {
        "messages": [HumanMessage(content="I need a study plan for Machine Learning Fundamentals")],
        "course_info": {
            "name": "Machine Learning Fundamentals",
            "topics": ["Linear Regression", "Logistic Regression", "Decision Trees", 
                      "Random Forests", "Neural Networks", "SVM", "K-Means", "PCA"]
        },
        "study_plan": {},
        "validation_feedback": "",
        "plan_status": "pending",
        "revision_count": 0
    }
    
    print("      LANGGRAPH MULTI-AGENT STUDY PLANNER")
    print(f"\n Request: {initial_state['messages'][0].content}")
    print(f" Course: {initial_state['course_info']['name']}")
    print(f" Topics: {len(initial_state['course_info']['topics'])}")
    print(f" LLM: {'ACTIVE' if LLM_AVAILABLE else 'SIMULATED'}")
    print("      EXECUTING WORKFLOW")
    
    logger.log(f"WORKFLOW START | course={initial_state['course_info']['name']} | topics={len(initial_state['course_info']['topics'])}")
    
    # ✅ Use invoke() for clean final state capture
    final_state = graph.invoke(initial_state)
    
    logger.log(f"WORKFLOW COMPLETE | status={final_state.get('plan_status')} | has_plan={bool(final_state.get('study_plan'))}")
    
    print("\n" + "="*70)
    print("      WORKFLOW COMPLETE")
    print("="*70)
    
    # Display results
    if final_state and final_state.get("study_plan"):
        plan = final_state.get("study_plan")
        status = final_state.get("plan_status", "unknown")
        
        print(f"\n Status: {status.upper()}")
        print(f"\n YOUR COMPLETE STUDY PLAN")
        print(f"   Course: {plan.get('course')}")
        print(f"   Duration: {plan.get('total_duration_days')} days")
        print(f"   Daily Commitment: {plan.get('daily_commitment_hours')} hours/day")
        print(f"   Total Hours: {plan.get('estimated_hours')} hours")
        print(f"\n FULL SCHEDULE ({plan.get('total_duration_days')} days):")
        
        # ✅ Print ALL days (not just [:3])
        for i, (day, details) in enumerate(plan.get("daily_breakdown", {}).items()):
            print(f"   Day {i+1:2d}: {day:20s} | {details['study_hours']} hrs | {details['recommended_time']}")
        
        print()
        
        # Log plan details
        logger.log(f"PLAN DETAILS | status={status} | days={plan.get('total_duration_days')} | daily_hrs={plan.get('daily_commitment_hours')} | total_hrs={plan.get('estimated_hours')}")
    else:
        print("\n No study plan generated")
        print(f"   Final Status: {final_state.get('plan_status') if final_state else 'None'}")
        logger.log(f"ERROR | No study plan generated | status={final_state.get('plan_status') if final_state else 'None'}")
        print()
    
    # ✅ Save execution log
    logger.save("study_planner_execution_log.txt")

if __name__ == "__main__":
    try:
        run_study_planner()
    except Exception as e:
        print(f"\n Error: {e}")
        logger.log(f"FATAL ERROR | {str(e)}")
        logger.save("study_planner_execution_log.txt")
        import traceback
        traceback.print_exc()
