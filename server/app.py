from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

# We import env from the root directory assuming root is part of PYTHONPATH
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from env import EmailTriageEnv, Action, Observation

app = FastAPI()

# We maintain a global environment instance for the server
env = EmailTriageEnv()

class ResetRequest(BaseModel):
    task_name: str = "easy"

@app.get("/")
def ping():
    # Automated ping to the Space URL must return 200
    return {"status": "ok", "message": "Email Triage Environment Server Running"}

@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    # Respond to reset() endpoint
    try:
        task = req.task_name if req else "easy"
        env.task_name = task
        obs = env.reset()
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step(action: Action):
    # Respond to step() endpoint
    try:
        obs, reward, done, info = env.step(action)
        # Clamp all scores to strict (0, 1) interval
        clamped_reward = max(0.01, min(0.99, reward))
        if "total_reward" in info:
            info["total_reward"] = max(0.01, min(0.99, info["total_reward"]))
        if "step_reward" in info:
            info["step_reward"] = max(0.01, min(0.99, info["step_reward"]))
        return {
            "observation": obs.model_dump(),
            "reward": clamped_reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state():
    # Respond to state() endpoint
    return env.state()

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
