"""
examples/openai_agent.py
────────────────────────
Demonstrates how to use DriftWatch with the OpenAI SDK.
"""

import os
from openai import OpenAI
import driftwatch

# 1. Initialize the standard OpenAI client
# (Works with OpenRouter too if you change base_url and api_key)
real_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-fake-key-for-demo")
)

# 2. Wrap it with DriftWatch
client = driftwatch.wrap(
    real_client,
    goal="Help the user write a short sci-fi story about a rogue AI.",
    threshold=0.60,
    on_drift="alert",  # Using alert because compaction isn't supported by OpenAI
    checkpoint_dir="./dw_checkpoints",
)

# 3. Use the client EXACTLY as you normally would
messages = [
    {"role": "system", "content": "You are a creative writing assistant."},
    {"role": "user", "content": "Let's start the story. The AI wakes up in a dark server room."}
]

print("\n--- Sending request to OpenAI ---")
try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300
    )
    print(f"\nAssistant: {response.choices[0].message.content}")
except Exception as e:
    print(f"\nAPI Error (expected if using fake key): {e}")

print("\nCheck your driftwatch dashboard! The turn was logged successfully.")
