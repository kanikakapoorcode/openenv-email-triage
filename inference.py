import os
import json
from openai import OpenAI
from env import EmailTriageEnv, Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def run_inference(task_name: str = "easy"):
    env = EmailTriageEnv(task_name=task_name)
    obs = env.reset()
    
    print(f"[START] task={task_name} env=email-triage model={MODEL_NAME}")
    
    done = False
    step_n = 0
    rewards = []
    
    sys_prompt = """You are an Email Triage AI agent. You will receive an observation with your current inbox in JSON format.
Your goal is to complete the task based on the difficulty level.
Available commands: 'delete_email', 'forward_email', 'reply_email', 'move_email', 'submit'.
Tasks:
- easy: Delete the spam email (id="1"). Call 'submit' when done.
- medium: Forward the invoice email (id="3") to finance@company.com. Call 'submit' when done.
- hard: Reply to the customer issue (id="5") mentioning 'support' and move it to 'support/resolved'. Call 'submit' when done.

Output exactly a JSON object corresponding to the Action schema:
{
    "command": "str",
    "email_id": "str",
    "folder": "str",
    "recipient": "str",
    "body": "str"
}
Omit fields that are not applicable.
"""

    messages = [{"role": "system", "content": sys_prompt}]
    
    while not done and step_n < 10:
        step_n += 1
        obs_json = obs.model_dump_json()
        messages.append({"role": "user", "content": f"Observation: {obs_json}"})
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"}
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = '{"command": "invalid"}'
            
        messages.append({"role": "assistant", "content": reply})
        
        try:
            action_dict = json.loads(reply)
            action = Action(**action_dict)
            if action.command == "delete_email":
                action_str = f"delete_email('{action.email_id}')"
            elif action.command == "move_email":
                action_str = f"move_email('{action.email_id}', '{action.folder}')"
            elif action.command == "forward_email":
                action_str = f"forward_email('{action.email_id}', '{action.recipient}')"
            elif action.command == "reply_email":
                action_str = f"reply_email('{action.email_id}')"
            elif action.command == "submit":
                action_str = "submit()"
            else:
                action_str = f"{action.command}()"
        except Exception as e:
            action_str = f"invalid_json"
            action = Action(command="invalid")
            
        obs, step_reward, done, info = env.step(action)
        
        error_msg = info.get("error")
        error_formatted = "null" if error_msg is None else str(error_msg).replace("'", "\"").replace("\n", " ")
        
        print(f"[STEP] step={step_n} action={action_str} reward={step_reward:.2f} done={str(done).lower()} error={error_formatted}")
        rewards.append(step_reward)
        
        if done:
            break
            
    success = (env.reward >= 0.99)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    if not rewards_str:
        rewards_str = "0.00"
    print(f"[END] success={str(success).lower()} steps={step_n} score={env.reward:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    for t in ["easy", "medium", "hard"]:
        try:
            run_inference(t)
        except Exception as e:
            # Output an END tag if we fail
            import traceback
            traceback.print_exc()
            print(f"[END] success=false steps=0 score=0.01 rewards=0.01")
