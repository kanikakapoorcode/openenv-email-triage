---
title: OpenEnv Email Triage
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: inference.py
pinned: false
tags: [openenv, reinforcement-learning, environment]
---

# Email Triage Environment

A real-world OpenEnv simulation environment for evaluating AI agents on an email triage task. The environment has 3 difficulty levels involving reading, deleting, forwarding, moving, and replying to emails.

## Action Space

The action space allows interacting with an email inbox:
- `command` (str): One of `"delete_email"`, `"forward_email"`, `"reply_email"`, `"move_email"`, `"submit"`.
- `email_id` (str, optional): The ID of the target email.
- `folder` (str, optional): The target folder for `"move_email"`.
- `recipient` (str, optional): The recipient for `"forward_email"` or `"reply_email"`.
- `body` (str, optional): The body text for `"reply_email"`.

## Observation Space

An `Observation` consists of:
- `emails`: A list of `Email` objects (excluding items in trash).
- `current_folder`: A string indicating the folder (always `"inbox"`).
- `last_action_error`: A string or `null` catching errors from invalid actions.

## Tasks & Agent Graders (Reward Function)

Three tasks are supported (Easy, Medium, Hard). Graders assign a total score of 0.0 to 1.0 based on deterministic criteria (e.g. is the spam deleted? is the invoice forwarded?). The environment provides partial rewards on intermediate steps (e.g., +0.5 for a correct step, -0.5 for mistakes). Infinite loops and destructive actions are also penalized.

- **Easy**: Delete the spam message and do not delete the boss' email.
- **Medium**: Forward the invoice email to finance@company.com.
- **Hard**: Reply to the customer's query with a specific string and move the email to the Support/Resolved folder.

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Export your Hugging Face and OpenAI (or other) tokens:
   ```bash
   export HF_TOKEN="your-hf-token"
   export API_BASE_URL="https://api.openai.com/v1"
   export MODEL_NAME="gpt-4o-mini"
   ```
3. Run the baseline evaluation:
   ```bash
   python inference.py
   ```

## Reproducible Benchmark Scores
- **Easy**: 1.00 (GPT-4o / GPT-4o-mini baseline)
- **Medium**: 1.00 (GPT-4o / GPT-4o-mini baseline)
- **Hard**: 1.00 (GPT-4o / GPT-4o-mini baseline)
