import asyncio
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage, ToolCallExecutionEvent
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

# ----------------------------
# Logging helper
# ----------------------------
class ExecutionLogger:
    def __init__(self, path: str = "study_planner_execution_log.txt"):
        self.path = path
        self.lines = []
        self.start_time = datetime.now()

    def log(self, source: str, content: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.lines.append(f"[{ts}] [{elapsed:>6.2f}s] {source}: {content}")

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("  AUTOGEN MULTI-AGENT STUDY PLANNER - SEQUENTIAL PIPELINE LOG\n")
            f.write("\n".join(self.lines) + "\n")
            f.write(f"Total execution time: {(datetime.now() - self.start_time).total_seconds():.2f}s\n")
            f.write("="*80 + "\n")


logger = ExecutionLogger()


# ----------------------------
# Tools (async functions)
# ----------------------------
async def analyze_course_content(course_name: str, topics: list[str]) -> str:
    """Analyzes course structure and returns JSON string"""
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
        analysis["topics_breakdown"][topic] = {"difficulty": difficulty, "hours": hours}

    logger.log("TOOL", f"analyze_course_content: {course_name} -> {analysis['total_topics']} topics, {analysis['estimated_hours']} hours")
    return json.dumps(analysis)


async def create_study_schedule(course_name: str, total_hours: int, study_days: int = 7) -> str:
    """Creates optimized study schedule and returns JSON string"""
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

    logger.log("TOOL", f"create_study_schedule: {study_days} days @ {round(daily_hours, 2)} hrs/day")
    return json.dumps(schedule)


async def validate_study_plan(plan_json: str) -> str:
    """Validates plan feasibility and returns JSON string"""
    plan = json.loads(plan_json)

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

    logger.log("TOOL", f"validate_study_plan: feasible={feedback['is_feasible']}, score={feedback['score']}")
    return json.dumps(feedback)


# ----------------------------
# Main (Sequential Pipeline Pattern from lab)
# ----------------------------
async def main():
    client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    print("\n" + "="*70)
    print("  AUTOGEN MULTI-AGENT STUDY PLANNER (SEQUENTIAL PIPELINE)")
    print(f"\n Model: gpt-4o-mini")
    print(f" Pattern: Sequential Pipeline (Team1 → Team2 → Team3)")
    print(f" Agents: Researcher → Scheduler → Validator")

    logger.log("SYSTEM", "Starting sequential pipeline")

    # ===================================================================
    # TEAM 1: RESEARCHER (Analyze course content)
    # ===================================================================
    researcher = AssistantAgent(
        name="Researcher",
        model_client=client,
        tools=[analyze_course_content],
        system_message="""You are a Researcher agent.
Call analyze_course_content(course_name, topics) and return ONLY the JSON result from the tool.
Do not add explanations. Just return the JSON."""
    )

    team1 = RoundRobinGroupChat(
        participants=[researcher],
        termination_condition=MaxMessageTermination(2)  # 1 prompt + 1 reply
    )

    print("\n--- PHASE 1: RESEARCHER (Analyzing Course) ---")
    logger.log("PHASE_1", "Starting Researcher team")
    
    task1 = """Analyze 'Machine Learning Fundamentals' with these 8 topics:
Linear Regression, Logistic Regression, Decision Trees, Random Forests, Neural Networks, SVM, K-Means, PCA."""
    
    result1 = await team1.run(task=task1)
    analysis_output = result1.messages[-1].content  # Get the agent's last message
    
    print(f"\n Researcher output:\n{analysis_output[:300]}...")
    logger.log("RESEARCHER_OUTPUT", analysis_output)

    # Parse the analysis to extract data for next phase
    try:
        analysis_data = json.loads(analysis_output)
        course_name = analysis_data["course"]
        estimated_hours = analysis_data["estimated_hours"]
    except:
        # Fallback if JSON parsing fails
        course_name = "Machine Learning Fundamentals"
        estimated_hours = 28

    # ===================================================================
    # TEAM 2: SCHEDULER (Create schedule)
    # ===================================================================
    scheduler = AssistantAgent(
        name="Scheduler",
        model_client=client,
        tools=[create_study_schedule],
        system_message="""You are a Scheduler agent.
Call create_study_schedule(course_name, total_hours, study_days=7) and return ONLY the JSON result from the tool.
Do not add explanations. Just return the JSON."""
    )

    team2 = RoundRobinGroupChat(
        participants=[scheduler],
        termination_condition=MaxMessageTermination(2)
    )

    print("\n--- PHASE 2: SCHEDULER (Creating 7-Day Schedule) ---")
    logger.log("PHASE_2", "Starting Scheduler team")
    
    task2 = f"""Create a 7-day study schedule for:
Course: {course_name}
Total hours: {estimated_hours}
Use study_days=7"""
    
    result2 = await team2.run(task=task2)
    schedule_output = result2.messages[-1].content
    
    print(f"\n Scheduler output:\n{schedule_output[:300]}...")
    logger.log("SCHEDULER_OUTPUT", schedule_output)

    # ===================================================================
    # TEAM 3: VALIDATOR (Validate plan)
    # ===================================================================
    validator = AssistantAgent(
        name="Validator",
        model_client=client,
        tools=[validate_study_plan],
        system_message="""You are a Validator agent.
Call validate_study_plan(plan_json) and return ONLY the JSON result from the tool.
Do not add explanations. Just return the JSON."""
    )

    team3 = RoundRobinGroupChat(
        participants=[validator],
        termination_condition=MaxMessageTermination(2)
    )

    print("\n--- PHASE 3: VALIDATOR (Validating Plan) ---")
    logger.log("PHASE_3", "Starting Validator team")
    
    task3 = f"""Validate this study plan:
{schedule_output}"""
    
    result3 = await team3.run(task=task3)
    validation_output = result3.messages[-1].content
    
    print(f"\n Validator output:\n{validation_output[:300]}...")
    logger.log("VALIDATOR_OUTPUT", validation_output)

    # ===================================================================
    # FINAL REPORT (Display the complete plan)
    # ===================================================================
    print("\n" + "="*70)
    print("  WORKFLOW COMPLETE - FINAL STUDY PLAN")
    
    try:
        schedule_data = json.loads(schedule_output)
        validation_data = json.loads(validation_output)
        
        print(f"\n Course: {schedule_data['course']}")
        print(f"  Duration: {schedule_data['total_duration_days']} days")
        print(f" Daily Commitment: {schedule_data['daily_commitment_hours']} hours/day")
        print(f" Total Hours: {schedule_data['estimated_hours']} hours")
        print(f"\n Validation: {'PASSED' if validation_data['is_feasible'] else 'FAILED'}")
        print(f" Score: {validation_data['score']}/100")
        
        if validation_data.get('recommendations'):
            print(f" Note: {validation_data['recommendations'][0]}")
        
        print(f"\n COMPLETE 7-DAY SCHEDULE:")
        print("-" * 70)
        for i, (day, details) in enumerate(schedule_data['daily_breakdown'].items(), 1):
            print(f"Day {i}: {day:20s} | {details['study_hours']} hrs | {details['recommended_time']}")
        print("-" * 70)
        
        logger.log("FINAL_STATUS", f"Validation: {validation_data['is_feasible']}, Score: {validation_data['score']}")
    except Exception as e:
        print(f"\n  Could not parse final output: {e}")
        logger.log("ERROR", str(e))

    logger.log("SYSTEM", "Sequential pipeline completed successfully")
    logger.save()
    print(f"\n Execution log saved to: study_planner_execution_log.txt\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        logger.save()
    except Exception as e:
        print(f"\n Error: {e}")
        logger.save()
        import traceback
        traceback.print_exc()
