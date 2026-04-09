from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Any

class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    folder: str

class Observation(BaseModel):
    emails: List[Email]
    current_folder: str
    last_action_error: Optional[str] = None

class Action(BaseModel):
    command: str = Field(..., description="Action to perform: 'read_email', 'delete_email', 'forward_email', 'reply_email', 'move_email', 'submit'")
    email_id: Optional[str] = Field(None, description="ID of the email to act upon")
    folder: Optional[str] = Field(None, description="Target folder for 'move_email'")
    recipient: Optional[str] = Field(None, description="Recipient email address for 'forward_email' or 'reply_email'")
    body: Optional[str] = Field(None, description="Email body for 'reply_email' or 'forward_email'")

class Reward(BaseModel):
    score: float
    reason: str

class EmailTriageEnv:
    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name
        self.reset()
        
    def reset(self) -> Observation:
        self.steps = 0
        self.last_action_error = None
        self.done = False
        self.reward = 0.0
        
        # Setup initial state based on task
        if self.task_name == "easy":
            self.emails = [
                Email(id="1", sender="spammer@spam.com", subject="WINNER WINNER", body="Click here to claim your prize!", folder="inbox"),
                Email(id="2", sender="boss@company.com", subject="Meeting update", body="Meeting moved to 3pm.", folder="inbox")
            ]
        elif self.task_name == "medium":
            self.emails = [
                Email(id="3", sender="billing@vendor.com", subject="Invoice #1234", body="Please process this invoice for $500.", folder="inbox"),
                Email(id="4", sender="hr@company.com", subject="Policy update", body="Please read the new policy.", folder="inbox")
            ]
        elif self.task_name == "hard":
            self.emails = [
                Email(id="5", sender="customer@client.com", subject="Issue with product", body="The app crashes on startup. Please help.", folder="inbox"),
                Email(id="6", sender="newsletter@tech.com", subject="Daily news", body="Here is the news...", folder="inbox")
            ]
        else:
            raise ValueError(f"Unknown task: {self.task_name}")
            
        self.sent_emails = []
        return self._get_obs()
        
    def state(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "emails": [e.model_dump() for e in self.emails],
            "sent_emails": self.sent_emails,
            "steps": self.steps,
            "done": self.done,
            "reward": self.reward,
            "last_action_error": self.last_action_error
        }
        
    def _get_obs(self) -> Observation:
        return Observation(
            emails=[e for e in self.emails if e.folder != "trash"],
            current_folder="inbox",
            last_action_error=self.last_action_error
        )
        
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        self.steps += 1
        self.last_action_error = None
        step_reward = 0.0
        
        try:
            if action.command == "delete_email":
                target = next((e for e in self.emails if e.id == action.email_id), None)
                if not target:
                    raise ValueError(f"Email {action.email_id} not found.")
                target.folder = "trash"
                
                # Partial reward for deleting spam in easy task
                if self.task_name == "easy" and target.id == "1":
                    step_reward = 0.5
                elif self.task_name == "easy" and target.id == "2":
                    step_reward = -0.5 # Penalty for deleting important email
                    
            elif action.command == "move_email":
                target = next((e for e in self.emails if e.id == action.email_id), None)
                if not target:
                    raise ValueError(f"Email {action.email_id} not found.")
                if not action.folder:
                    raise ValueError("Target folder must be specified.")
                target.folder = action.folder
                
                if self.task_name == "hard" and target.id == "5" and action.folder.lower() == "support/resolved":
                    step_reward = 0.4
                    
            elif action.command == "forward_email":
                target = next((e for e in self.emails if e.id == action.email_id), None)
                if not target:
                    raise ValueError(f"Email {action.email_id} not found.")
                if not action.recipient:
                    raise ValueError("Recipient must be specified.")
                self.sent_emails.append({
                    "from": "agent@company.com",
                    "to": action.recipient,
                    "subject": f"Fwd: {target.subject}",
                    "body": action.body or ""
                })
                
                if self.task_name == "medium" and target.id == "3" and action.recipient == "finance@company.com":
                    step_reward = 0.5
                    
            elif action.command == "reply_email":
                target = next((e for e in self.emails if e.id == action.email_id), None)
                if not target:
                    raise ValueError(f"Email {action.email_id} not found.")
                if not action.body:
                    raise ValueError("Body must be specified.")
                self.sent_emails.append({
                    "from": "agent@company.com",
                    "to": target.sender,
                    "subject": f"Re: {target.subject}",
                    "body": action.body
                })
                
                if self.task_name == "hard" and target.id == "5" and "support" in action.body.lower():
                    step_reward = 0.5
                    
            elif action.command == "submit":
                self.done = True
                
            else:
                self.last_action_error = f"Unknown command: {action.command}"
                step_reward = -0.1
                
        except Exception as e:
            self.last_action_error = str(e)
            step_reward = -0.1
            
        self.reward = self._compute_total_reward()
        
        info = {
            "step_reward": step_reward,
            "total_reward": self.reward,
            "error": self.last_action_error
        }
        
        if self.steps >= 10:
            self.done = True
            
        return self._get_obs(), step_reward, self.done, info
        
    def _compute_total_reward(self) -> float:
        total = 0.0
        if self.task_name == "easy":
            # Target: spam (1) in trash, boss (2) not in trash
            spam_email = next((e for e in self.emails if e.id == "1"), None)
            boss_email = next((e for e in self.emails if e.id == "2"), None)
            if spam_email and spam_email.folder == "trash":
                total += 0.5
            else:
                total -= 0.5 # Penalty for not handling spam
                
            if boss_email and boss_email.folder != "trash":
                total += 0.5
            else:
                total -= 0.5
                
        elif self.task_name == "medium":
            # Target: forward #3 to finance@company.com
            forwarded = any(e["to"] == "finance@company.com" and "Fwd: Invoice" in e["subject"] for e in self.sent_emails)
            if forwarded:
                total += 1.0
            else:
                total += 0.0
                
        elif self.task_name == "hard":
            # Target: reply to #5 with helpful message, and move #5 to Support/Resolved
            replied = any(e["to"] == "customer@client.com" and "Re: Issue" in e["subject"] for e in self.sent_emails)
            customer_email = next((e for e in self.emails if e.id == "5"), None)
            
            score = 0.0
            if replied:
                score += 0.5
            if customer_email and customer_email.folder.lower() == "support/resolved":
                score += 0.5
            total += score
            
        # Clamp to open interval (0, 1) — scores must be strictly between 0 and 1
        return max(0.01, min(0.99, total))
