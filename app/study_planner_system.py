"""
LangGraph Multi-Agent Study Planner System - WITH OLLAMA LLM


Architecture:
- Supervisor Node: Routes tasks and manages workflow
- Researcher Agent: Analyzes courses with LLM insights
- Scheduler Agent: Creates optimized schedules with LLM recommendations  
- Validator Agent: Validates with LLM-enhanced feedback
- State: Persistent memory with checkpointer
- Conditional Edge: Validation loop for plan refinement
- Logging: Real-time execution trace with detailed output
- LLM: Deepseek R1 1.5B via Ollama
"""

from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama  
from langchain_core.tools import tool
import json
from datetime import datetime, timedelta
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("  python-dotenv not installed. Using defaults.")


# LLM CONFIGURATION - Ollama + Deepseek R1

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:latest")

print(" INITIALIZING LLM CONFIGURATION")
print(f"Base URL: {OLLAMA_BASE_URL}")
print(f"Model: {OLLAMA_MODEL}")
print("Initializing LLM instances...\n")

try:
    # Initialize specialized LLM instances for different agents
    # Each has different temperature for different roles
    
    llm_researcher = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,      # Higher: creative analysis
        top_p=0.9,
        num_predict=2048,
        timeout=120           # Deepseek can be slow
    )
    
    llm_scheduler = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.5,      # Medium: practical planning
        top_p=0.9,
        num_predict=2048,
        timeout=120
    )
    
    llm_validator = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,      # Lower: strict validation
        top_p=0.9,
        num_predict=1024,
        timeout=120
    )
    
    print(" LLM instances initialized successfully!")

    
    LLM_AVAILABLE = True
    
except Exception as e:
    print(f"  WARNING: Could not initialize LLM: {e}")
    print("   System will continue with simulated responses")
    print("   Make sure Ollama is running: ollama serve\n")
    LLM_AVAILABLE = False
    llm_researcher = None
    llm_scheduler = None
    llm_validator = None



#all workflow activity

class ExecutionLogger:
    """Captures and formats execution logs for display"""
    
    def __init__(self):
        self.logs = []
        self.step_count = 0
        self.start_time = datetime.now()
    
    def log_step(self, step_type: str, node_name: str, details: dict):
        """Log a workflow step"""
        self.step_count += 1
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        log_entry = {
            "step": self.step_count,
            "type": step_type,
            "node": node_name,
            "timestamp": datetime.now().isoformat(),
            "elapsed_ms": int(elapsed * 1000),
            "details": details
        }
        
        self.logs.append(log_entry)
        return log_entry
    
    def log_tool_call(self, tool_name: str, params: dict, result: dict):
        """Log a tool execution"""
        self.logs.append({
            "type": "tool_call",
            "tool": tool_name,
            "params": params,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_state_update(self, updates: dict):
        """Log state changes"""
        self.logs.append({
            "type": "state_update",
            "updates": {k: v for k, v in updates.items() if k != "messages"},
            "timestamp": datetime.now().isoformat()
        })
    
    def get_formatted_log(self) -> str:
        """Generate a formatted execution log"""
        log_text = self._get_header()
        log_text += self._get_execution_trace()
        log_text += self._get_summary()
        return log_text
    
    def _get_header(self) -> str:
        """Generate log header"""
        llm_status = " OLLAMA LLM ACTIVE" if LLM_AVAILABLE else " SIMULATED (LLM UNAVAILABLE)"
        return f"""
                           
                                                                           
         Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  {llm_status}         



"""
    
    def _get_execution_trace(self) -> str:
        """Generate detailed execution trace"""
        trace = ""
        
        for entry in self.logs:
            if entry["type"] == "step":
                trace += f"""
  STEP {entry['step']} | {entry['node'].upper()}

Timestamp: {entry['timestamp']}
Elapsed: {entry['elapsed_ms']}ms

"""
                if "output" in entry["details"]:
                    trace += entry["details"]["output"]
                trace += "\n"
            
            elif entry["type"] == "tool_call":
                trace += f"""
 TOOL CALL: {entry['tool']}()

Parameters:
{json.dumps(entry['params'], indent=2)}

Result:
{json.dumps(entry['result'], indent=2)}

"""
            
            elif entry["type"] == "state_update":
                trace += f"""
STATE UPDATE

{json.dumps(entry['updates'], indent=2)}

"""
        
        return trace
    
    def _get_summary(self) -> str:
        """Generate execution summary"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        return f"""
 
  METRICS:
├─ Total Steps: {self.step_count}
├─ Total Time: {total_time:.2f} seconds
├─ Total Log Entries: {len(self.logs)}
├─ Tool Calls: {sum(1 for l in self.logs if l.get('type') == 'tool_call')}
├─ State Updates: {sum(1 for l in self.logs if l.get('type') == 'state_update')}
├─ LLM Status: {' Active (Ollama)' if LLM_AVAILABLE else 'Simulated'}
└─ Status:  COMPLETED

"""

# Global logger instance
execution_logger = ExecutionLogger()

# STATE DEFINITION

class PlannerState(TypedDict):
    """
    Central state object persisting throughout graph execution.
    This is our 'backpack' that every node can read from and write to.
    """
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    course_info: dict  # Information about courses to study
    study_plan: dict  # Generated study schedule
    validation_feedback: str  # Feedback from validation
    plan_status: Literal["pending", "validated", "needs_revision", "completed"]
    revision_count: int  # Track how many revisions we've done
    next_step: Literal["researcher", "scheduler", "validator", "end"]



# TOOLS 


@tool
def analyze_course_content(course_name: str, topics: list) -> dict:
    """
    Analyzes course content and creates a breakdown of topics.
    The researcher agent uses this to understand what needs to be studied.
    """
    analysis = {
        "course": course_name,
        "total_topics": len(topics),
        "topics_breakdown": {},
        "estimated_hours": 0,
        "difficulty_levels": {}
    }
    
    for i, topic in enumerate(topics):
        difficulty = ["Beginner", "Intermediate", "Advanced"][i % 3]
        hours = 2 + (i % 4)
        analysis["topics_breakdown"][topic] = {
            "difficulty": difficulty,
            "estimated_hours": hours,
            "prerequisites": []
        }
        analysis["estimated_hours"] += hours
        analysis["difficulty_levels"][difficulty] = analysis["difficulty_levels"].get(difficulty, 0) + 1
    
    execution_logger.log_tool_call(
        "analyze_course_content",
        {"course_name": course_name, "topics_count": len(topics)},
        analysis
    )
    
    return analysis


@tool
def create_study_schedule(course_name: str, total_hours: int, study_days: int) -> dict:
    """
    Creates an optimized study schedule based on course requirements.
    """
    daily_hours = total_hours / study_days
    schedule = {
        "course": course_name,
        "schedule_created": datetime.now().isoformat(),
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
    
    execution_logger.log_tool_call(
        "create_study_schedule",
        {"course_name": course_name, "total_hours": total_hours, "study_days": study_days},
        {
            "daily_commitment_hours": schedule["daily_commitment_hours"],
            "total_duration_days": schedule["total_duration_days"]
        }
    )
    
    return schedule


@tool
def validate_study_plan(plan: dict) -> dict:
    """
    Validates the study plan for feasibility and completeness.
    """
    feedback = {
        "is_feasible": True,
        "issues": [],
        "recommendations": [],
        "score": 0
    }
    
    if plan.get("daily_commitment_hours", 0) > 4:
        feedback["issues"].append("Daily commitment exceeds 4 hours - may be unsustainable")
        feedback["is_feasible"] = False
    
    if plan.get("total_duration_days", 0) < 7:
        feedback["recommendations"].append("Consider extending study period for better retention")
    
    if not plan.get("daily_breakdown"):
        feedback["issues"].append("Missing daily breakdown in schedule")
        feedback["is_feasible"] = False
    else:
        feedback["score"] = 85
        feedback["recommendations"].append("Schedule is well-structured and balanced")
    
    if feedback["is_feasible"]:
        feedback["score"] = min(100, feedback["score"] + 15)
    
    execution_logger.log_tool_call(
        "validate_study_plan",
        {"plan_keys": list(plan.keys())},
        feedback
    )
    
    return feedback


# NODE DEFINITIONS 


def supervisor_node(state: PlannerState) -> Command:
    """
    Supervisor node: Analyzes state and routes to appropriate agent.
    Acts as the project manager, deciding next steps.
    """
    
    if state["plan_status"] == "pending":
        next_step = "researcher"
        decision = "Starting workflow: routing to Researcher to analyze course content"
    elif state["plan_status"] == "validated":
        next_step = "end"
        decision = "Plan validated successfully. Workflow complete."
    elif state["plan_status"] == "needs_revision":
        state["revision_count"] += 1
        if state["revision_count"] <= 2:
            next_step = "scheduler"
            decision = f"Plan needs revision (attempt {state['revision_count']}). Routing to Scheduler."
        else:
            next_step = "end"
            decision = "Max revision attempts reached. Completing workflow."
    else:
        next_step = "scheduler"
        decision = "Course analyzed. Routing to Scheduler to create study plan."
    
    output = f"""
 SUPERVISOR NODE

Decision: {decision}
Current Status: {state['plan_status']}
Revision Count: {state['revision_count']}
Next Route: {next_step.upper()}
"""
    
    execution_logger.log_step(
        "node",
        "supervisor",
        {"decision": decision, "next_step": next_step, "output": output}
    )
    
    return Command(
        update={
            "next_step": next_step,
            "messages": state["messages"] + [AIMessage(content=f"Supervisor decision: {decision}")]
        },
        goto=next_step
    )


def researcher_node(state: PlannerState) -> dict:
    """
    Researcher Agent: Analyzes courses with LLM-enhanced insights.
    """
    
    course_data = state.get("course_info", {})
    
    # Step 1: Get structured analysis from tool
    analysis_result = analyze_course_content(
        course_name=course_data.get("name", "Unknown Course"),
        topics=course_data.get("topics", [])
    )
    
    response_text = f"""
 COURSE ANALYSIS COMPLETE

Course: {analysis_result['course']}
├─ Total Topics: {analysis_result['total_topics']}
├─ Estimated Hours: {analysis_result['estimated_hours']}
├─ Difficulty Breakdown:
│  ├─ Beginner: {analysis_result['difficulty_levels'].get('Beginner', 0)}
│  ├─ Intermediate: {analysis_result['difficulty_levels'].get('Intermediate', 0)}
│  └─ Advanced: {analysis_result['difficulty_levels'].get('Advanced', 0)}
"""
    
    # Step 2: Get LLM insights if available
    if LLM_AVAILABLE and llm_researcher:
        try:
            prompt = f"""You are an expert course analyst. Analyze this structure:
            
Course: {analysis_result['course']}
Topics: {', '.join(list(analysis_result['topics_breakdown'].keys())[:5])}... and {analysis_result['total_topics']-5} more
Total Hours: {analysis_result['estimated_hours']}
Levels: {analysis_result['difficulty_levels']}

Provide brief insights (2-3 lines):
1. Best learning sequence
2. Time allocation tips"""
            
            llm_response = llm_researcher.invoke(prompt)
            llm_insights = llm_response.content
            response_text += f"\n AI Analysis:\n{llm_insights}"
            
        except Exception as e:
            response_text += f"\n  LLM Analysis skipped: {str(e)[:50]}..."
    
    response_text += f"\n\nKey Topics: {', '.join(list(analysis_result['topics_breakdown'].keys())[:3])}...\nStatus: Ready for scheduling"
    
    output = f"""
 RESEARCHER AGENT (LLM-Enhanced)


{response_text}
"""
    
    execution_logger.log_step(
        "node",
        "researcher",
        {
            "course": analysis_result['course'],
            "topics_analyzed": analysis_result['total_topics'],
            "estimated_hours": analysis_result['estimated_hours'],
            "llm_used": LLM_AVAILABLE,
            "output": output
        }
    )
    
    execution_logger.log_state_update({
        "plan_status": "analyzed",
        "course_info": analysis_result
    })
    
    return {
        "messages": state["messages"] + [AIMessage(content=response_text)],
        "course_info": {**state["course_info"], **analysis_result},
        "plan_status": "analyzed"
    }


def scheduler_node(state: PlannerState) -> dict:
    """
    Scheduler Agent: Creates optimized schedules with LLM recommendations.
    """
    
    course_info = state.get("course_info", {})
    estimated_hours = course_info.get("estimated_hours", 20)
    study_days = 14
    
    # Step 1: Generate schedule from tool
    schedule = create_study_schedule(
        course_name=course_info.get("course", "Study Plan"),
        total_hours=estimated_hours,
        study_days=study_days
    )
    
    response_text = f"""
 STUDY SCHEDULE GENERATED

Course: {schedule['course']}
├─ Duration: {schedule['total_duration_days']} days
├─ Daily Commitment: {schedule['daily_commitment_hours']} hours/day
└─ Study Method: Pomodoro (25min focus / 5min break)
"""
    
    # Step 2: Get LLM optimization tips if available
    if LLM_AVAILABLE and llm_scheduler:
        try:
            prompt = f"""Optimize this study plan:
            
Course: {schedule['course']}
Days: {schedule['total_duration_days']}
Daily Hours: {schedule['daily_commitment_hours']}
Total: {estimated_hours} hours

Give 2 practical tips for studying this course."""
            
            llm_response = llm_scheduler.invoke(prompt)
            llm_tips = llm_response.content
            response_text += f"\n Optimization Tips:\n{llm_tips}"
            
        except Exception as e:
            response_text += f"\n  LLM Tips skipped: {str(e)[:50]}..."
    
    response_text += "\n\nSchedule: Day 1 (Morning), Day 2 (Evening), ... (14 days total)"
    
    output = f"""
 SCHEDULER AGENT (LLM-Optimized)


{response_text}

Status: Ready for validation
"""
    
    execution_logger.log_step(
        "node",
        "scheduler",
        {
            "course": schedule['course'],
            "daily_hours": schedule['daily_commitment_hours'],
            "duration": schedule['total_duration_days'],
            "llm_used": LLM_AVAILABLE,
            "output": output
        }
    )
    
    execution_logger.log_state_update({
        "plan_status": "pending_validation",
        "study_plan": schedule
    })
    
    return {
        "messages": state["messages"] + [AIMessage(content=response_text)],
        "study_plan": schedule,
        "plan_status": "pending_validation"
    }


def validator_node(state: PlannerState) -> dict:
    """
    Validator Node: Checks plan feasibility with optional LLM validation.
    """
    
    study_plan = state.get("study_plan", {})
    
    if not study_plan:
        output = " ERROR: No study plan to validate"
        execution_logger.log_step("node", "validator", {"output": output, "error": True})
        return {
            "messages": state["messages"] + [AIMessage(content=output)],
            "validation_feedback": "No plan available",
            "plan_status": "needs_revision"
        }
    
    # Validate the plan
    feedback = validate_study_plan(study_plan)
    
    validation_status = " APPROVED" if feedback['is_feasible'] else "  NEEDS REVISION"
    
    response_text = f"""
PLAN VALIDATION REPORT


Feasibility Status: {validation_status}
Quality Score: {feedback['score']}/100
"""
    
    if feedback['issues']:
        response_text += f"\nIssues Found ({len(feedback['issues'])}): {', '.join(feedback['issues'][:1])}..."
    
    if feedback['recommendations']:
        response_text += f"\nRecommendations: {', '.join(feedback['recommendations'][:1])}..."
    
    output = f"""
 VALIDATOR NODE


{response_text}
"""
    
    next_status = "validated" if feedback['is_feasible'] else "needs_revision"
    
    execution_logger.log_step(
        "node",
        "validator",
        {
            "is_feasible": feedback['is_feasible'],
            "score": feedback['score'],
            "issues": len(feedback['issues']),
            "output": output
        }
    )
    
    execution_logger.log_state_update({
        "plan_status": next_status,
        "validation_feedback": feedback
    })
    
    return {
        "messages": state["messages"] + [AIMessage(content=response_text)],
        "validation_feedback": json.dumps(feedback),
        "plan_status": next_status
    }


# Router Function


def route_after_validation(state: PlannerState) -> Literal["scheduler", "end"]:
    """
    Conditional edge: Routes based on validation results.
    This creates our decision loop for plan refinement.
    """
    
    if state["plan_status"] == "needs_revision" and state["revision_count"] < 2:
        decision = f"Revision needed (attempt {state['revision_count']}). Looping back to Scheduler."
        route = "scheduler"
    else:
        decision = "Plan validated. Moving to completion."
        route = "end"
    
    output = f"""
CONDITIONAL EDGE ROUTING
{decision}
Target: {route.upper()}
"""
    
    execution_logger.log_step(
        "routing",
        "validator→[conditional]",
        {"decision": decision, "route": route, "output": output}
    )
    
    return route




# EXECUTION & LOGGING

def run_study_planner():
    """
    Executes the study planner system with full logging and LLM integration.
    """
    print("\n")

    
    graph = create_study_planner_graph()
    
    initial_state = {
        "messages": [
            HumanMessage(content="I need a study plan for Machine Learning Fundamentals course")
        ],
        "course_info": {
            "name": "Machine Learning Fundamentals",
            "topics": [
                "Linear Regression", "Logistic Regression", "Decision Trees",
                "Random Forests", "Neural Networks", "Support Vector Machines",
                "K-Means Clustering", "Principal Component Analysis"
            ]
        },
        "study_plan": {},
        "validation_feedback": "",
        "plan_status": "pending",
        "revision_count": 0,
        "next_step": "researcher"
    }
    
    print(" USER REQUEST:")
    print(f"   {initial_state['messages'][0].content}")
    print()
    print(" COURSE INFO:")
    print(f"   Course: {initial_state['course_info']['name']}")
    print(f"   Topics: {len(initial_state['course_info']['topics'])} topics")
    print()
    print(f" LLM STATUS: {' OLLAMA ACTIVE' if LLM_AVAILABLE else '  SIMULATED MODE'}")
    print()
    print("  Starting multi-agent workflow...\n")
    
    config = {"configurable": {"thread_id": "study-plan-001"}}
    
    for event in graph.stream(initial_state, config):
        pass
    
    print(execution_logger.get_formatted_log())
    
    print(" WORKFLOW COMPLETE - EXECUTION LOG ABOVE")
    
    log_file = Path("study_planner_execution_log.txt")
    with open(log_file, "w") as f:
        f.write(execution_logger.get_formatted_log())
    
    print(f" Execution log saved to: {log_file}\n")



if __name__ == "__main__":
    try:
        run_study_planner()
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()