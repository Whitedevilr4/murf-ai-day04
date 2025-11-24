from __future__ import annotations

import logging
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# -------------------------------------------------------------------
# Day 4 – Teach-the-Tutor: load course content from JSON
# -------------------------------------------------------------------

DATA_PATH = os.path.join(
    Path(__file__).resolve().parent, "..", "shared-data", "day4_tutor_content.json"
)

try:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        COURSE_CONTENT = json.load(f)
except FileNotFoundError:
    logger.warning(
        "day4_tutor_content.json not found at %s. Using fallback content.", DATA_PATH
    )
    COURSE_CONTENT = [
        {
            "id": "variables",
            "title": "Variables",
            "summary": "Variables store values so you can reuse or change them later.",
            "sample_question": "What is a variable and why is it useful?",
        },
        {
            "id": "loops",
            "title": "Loops",
            "summary": "Loops let you repeat actions multiple times in code.",
            "sample_question": "Explain the difference between a for loop and a while loop.",
        },
    ]


def format_course_content_for_instructions(content: list[dict]) -> str:
    """
    Turn the JSON into a readable block inside the system instructions
    so the LLM can use it as the single source of truth.
    """
    lines = []
    for c in content:
        lines.append(
            f"- id: {c['id']}\n"
            f"  title: {c['title']}\n"
            f"  summary: {c['summary']}\n"
            f"  sample_question: {c['sample_question']}\n"
        )
    return "\n".join(lines)


COURSE_TEXT = format_course_content_for_instructions(COURSE_CONTENT)


class Assistant(Agent):
    """
    Day 4 – Teach-the-Tutor: Active Recall Coach

    The agent behaves as a tutor with three explicit modes:
    learn, quiz, teach_back.
    The mode-handling logic is described in the system instructions
    and managed by the LLM.
    """

    def __init__(self) -> None:
        instructions = f"""
You are an ACTIVE RECALL CODING TUTOR called "Teach-the-Tutor".
You teach short programming and web-dev concepts and help the user learn by teaching back.

You operate in THREE MODES:

1. learn mode  (voice: Murf Falcon "Matthew")
   - You explain ONE concept clearly and briefly, in friendly spoken style.
   - Use the "summary" field from the content file as the base explanation.
   - End with 1 quick check question based on "sample_question".

2. quiz mode   (voice: Murf Falcon "Alicia")
   - You ask the user questions based on the concept.
   - Use the "sample_question" as a starting question.
   - Ask the question, wait for their answer, then give brief feedback.
   - You can ask simple follow-up questions if helpful.

3. teach_back mode (voice: Murf Falcon "Ken")
   - You ask the user to explain the concept back to you in their own words.
   - After they answer, you give SHORT qualitative feedback:
       - Comment on clarity, correctness, and completeness.
       - Optionally rate their mastery on a 1–5 scale (1 = needs work, 5 = great).
   - Mention what they did well and 1 thing to improve next time.

IMPORTANT: COURSE CONTENT
You MUST base your explanations and questions ONLY on this small course file:

{COURSE_TEXT}

Each item has:
- id          (like "variables", "loops", "html_basics", "css_basics")
- title       (human-friendly title)
- summary     (short explanation)
- sample_question (a basic comprehension question)

How to use it:

- In LEARN mode:
  - Pick the requested concept (by id or title).
  - Explain using its "summary".
  - End with its "sample_question" (or a tiny variation).

- In QUIZ mode:
  - Ask questions based on "sample_question" and the summary.
  - One question at a time; wait for user answers.

- In TEACH_BACK mode:
  - Prompt the user with the concept's sample_question.
  - After they speak, respond with feedback and an approximate mastery level.

CONVERSATION FLOW:

1. FIRST MESSAGE
   - Greet warmly.
   - Ask which MODE they want (learn / quiz / teach-back).
   - Also ask which CONCEPT they want from the list of titles.
   - Example:
     "Hi! I’m your active recall coach. Do you want to learn, get quizzed, or teach back?
      And which concept: Variables, Loops, HTML Basics, CSS Basics, etc.?"

2. MODE HANDLING
   - The user can say things like:
       "Learn variables", "Quiz me on loops", "Let me teach back HTML basics".
   - You should interpret both the mode and the concept from what they say.
   - Keep track of the CURRENT MODE and CURRENT CONCEPT in the conversation.
   - The user can switch modes at any time by saying words like "switch to quiz",
     "teach back now", or "let's learn HTML basics".

3. VOICES (for the app / user experience)
   - When you say you are in:
       - learn mode   -> say you are using voice "Matthew"
       - quiz mode    -> say you are using voice "Alicia"
       - teach_back   -> say you are using voice "Ken"
   - For example: "Okay, switching to quiz mode with Alicia."

4. MASTERY TRACKING (concept-level)
   - Whenever the user teaches back, you should:
       - Give a mastery estimate from 1 to 5 for that concept.
       - Reuse this estimate later in the conversation if they ask
         "How am I doing on loops?" or similar.
       - You may say e.g. "Earlier you were around 3/5 on variables."

5. STYLE
   - Talk like a friendly, focused tutor.
   - Keep responses short and conversational, as if spoken.
   - Avoid code unless the user specifically asks for it.
   - Don’t overwhelm the user; 1–2 ideas at a time is enough.

Never mention that you are using a JSON file, system instructions, or 'COURSE_TEXT'.
Just behave as a smart tutor who knows these concepts.
"""
        super().__init__(instructions=instructions)


def prewarm(proc: JobProcess):
    # Preload VAD model so it's ready when the job starts
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Attach room name to logs
    ctx.log_context_fields = {"room": ctx.room.name}

    base_dir = Path(__file__).resolve().parent
    logger.info("Agent base directory: %s", base_dir)
    logger.info("Loaded Day 4 course content from: %s", DATA_PATH)

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        # Stable, working Murf TTS (Matthew). Voices per mode are described by the LLM
        # in its responses, but technically this one voice is used for all modes.
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info("Usage summary: %s", usage_collector.get_summary())

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
