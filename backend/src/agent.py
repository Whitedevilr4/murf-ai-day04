from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from typing import Any, Dict

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
from livekit.agents.llm import function_tool
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# -------------------------------------------------------------------
# Voice constants (actual Murf Falcon IDs)
# -------------------------------------------------------------------

VOICE_LEARN = "en-US-matthew"   # Learn mode – Matthew
VOICE_QUIZ = "en-US-alicia"     # Quiz mode – Alicia
VOICE_TEACH = "en-US-ken"       # Teach-back mode – Ken

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

# -------------------------------------------------------------------
# Voice switching helper (from your Murf doc, adapted to this project)
# -------------------------------------------------------------------


def switch_session_voice(session: AgentSession, new_voice: str) -> bool:
    """
    Switch the session's TTS voice by replacing the TTS instance.

    This follows the pattern in the Murf "Voice Switching Implementation"
    doc you shared, but keeps the same tokenizer + style setup.
    """
    try:
        logger.info(f"🎤 Switching session voice to: {new_voice}")

        new_tts = murf.TTS(
            voice=new_voice,
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        )

        updated = False

        # Primary attributes
        if hasattr(session, "_tts"):
            session._tts = new_tts
            updated = True

        if hasattr(session, "tts"):
            session.tts = new_tts
            updated = True

        # Internal agent output (used by voice pipeline)
        try:
            if hasattr(session, "_agent_output") and hasattr(
                session._agent_output, "_tts"
            ):
                session._agent_output._tts = new_tts
                updated = True
        except Exception as e:
            logger.warning(f"Could not update _agent_output._tts: {e}")

        if not updated:
            logger.warning("Voice switch attempted but no TTS fields were updated.")

        return updated
    except Exception as e:
        logger.error(f"Voice switch failed: {e}")
        return False


# -------------------------------------------------------------------
# LLM Agent definition
# -------------------------------------------------------------------


class Assistant(Agent):
    """
    Day 4 – Teach-the-Tutor: Active Recall Coach with REAL voice switching.

    Modes:
      - learn      -> Murf Falcon Matthew
      - quiz       -> Murf Falcon Alicia
      - teach_back -> Murf Falcon Ken

    The mode (and voice) are changed via the set_mode() tool.
    """

    def __init__(self) -> None:
        instructions = f"""
You are an ACTIVE RECALL CODING TUTOR called "Teach-the-Tutor".
You teach short programming and web-dev concepts and help the user learn by teaching back.

You operate in THREE MODES:

1. LEARN MODE  (voice persona: Murf Falcon "Matthew")
   - You explain ONE concept clearly and briefly, in friendly spoken style.
   - Use the "summary" field from the content file as the base explanation.
   - End with 1 quick check question based on "sample_question".

2. QUIZ MODE   (voice persona: Murf Falcon "Alicia")
   - You ask the user questions based on the concept.
   - Use the "sample_question" as a starting question.
   - Ask ONE question, wait for their answer, then give brief feedback.
   - You can ask simple follow-up questions if helpful.

3. TEACH_BACK MODE (voice persona: Murf Falcon "Ken")
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

HOW TO USE MODES & VOICES:

- There is a TOOL available: set_mode(mode: "learn" | "quiz" | "teach_back").
- Whenever the user chooses or switches mode, you MUST call set_mode with that mode.
    - Example: if the user says "Quiz me on loops", you:
        1) Call set_mode("quiz")
        2) Then behave in quiz mode and start quizzing on loops.
    - Example: "Let me teach back HTML basics":
        1) Call set_mode("teach_back")
        2) Ask the user to explain HTML basics in their own words.

- The tool set_mode automatically changes the actual Murf voice:
    - "learn"      -> Matthew
    - "quiz"       -> Alicia
    - "teach_back" -> Ken

CONVERSATION FLOW:

1. FIRST MESSAGE
   - Greet warmly.
   - Briefly explain the three modes (learn, quiz, teach-back).
   - Ask which MODE they want AND which CONCEPT they want from the list of titles.
   - Example:
     "Hi! I’m your active recall coach. Do you want to learn, get quizzed, or teach back?
      And which concept: Variables, Loops, HTML Basics, CSS Basics, etc.?"

2. MODE HANDLING
   - The user can say things like:
       "Learn variables", "Quiz me on loops", "Let me teach back HTML basics".
   - You should interpret both the mode and the concept from what they say.
   - ALWAYS call set_mode when changing the mode.
   - Keep track of the CURRENT MODE and CURRENT CONCEPT in the conversation.
   - The user can switch modes at any time by saying words like "switch to quiz",
     "teach back now", or "let's learn HTML basics".

3. VOICE PERSONAS (for the app / user experience)
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

TECHNICAL RULE:
- Never mention tools, JSON files, system instructions, or 'COURSE_TEXT'.
- Never talk about "set_mode" directly unless you’re calling it as a tool.
"""
        super().__init__(instructions=instructions)

    # ------------------------------------------------------------------
    # TOOL: set_mode – switches internal mode + Murf voice
    # ------------------------------------------------------------------

    @function_tool
    async def set_mode(self, mode: str) -> str:
        """
        Change the learning mode and switch voice: 'learn', 'quiz', or 'teach_back'.

        This is exposed as a tool to the LLM. It:
          - stores the mode in session.userdata["tutor"]["mode"]
          - switches Murf TTS voice using switch_session_voice(...)
        """
        m = (mode or "").strip().lower()
        if m not in ("learn", "quiz", "teach_back"):
            return "Unknown mode. Please choose 'learn', 'quiz', or 'teach_back'."

        # Ensure userdata is a dict we can write into
        userdata: Dict[str, Any] = self.session.userdata or {}
        tutor_state: Dict[str, Any] = userdata.get("tutor") or {}
        tutor_state["mode"] = m
        userdata["tutor"] = tutor_state
        self.session.userdata = userdata

        voice_map = {
            "learn": VOICE_LEARN,
            "quiz": VOICE_QUIZ,
            "teach_back": VOICE_TEACH,
        }
        new_voice = voice_map.get(m, VOICE_LEARN)

        switched = switch_session_voice(self.session, new_voice)
        if not switched:
            logger.warning("set_mode: voice switch to %s may not have taken effect.", new_voice)

        # Friendly textual confirmation back to the user / LLM
        if m == "learn":
            return "Mode set to LEARN with Matthew. I’ll focus on explaining concepts clearly."
        elif m == "quiz":
            return "Mode set to QUIZ with Alicia. I’ll start asking you short questions."
        else:
            return "Mode set to TEACH-BACK with Ken. You’ll explain the concept in your own words."


# -------------------------------------------------------------------
# Worker entrypoint
# -------------------------------------------------------------------


def prewarm(proc: JobProcess):
    # Preload VAD model so it's ready when the job starts
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Attach room name to logs
    ctx.log_context_fields = {"room": ctx.room.name}

    base_dir = Path(__file__).resolve().parent
    logger.info("Agent base directory: %s", base_dir)
    logger.info("Loaded Day 4 course content from: %s", DATA_PATH)

    # Initial userdata: store tutor state + anything else you want later
    initial_userdata: Dict[str, Any] = {
        "tutor": {
            "mode": "learn",  # default starting mode
        }
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        # Start with LEARN voice (Matthew); set_mode() will switch later
        tts=murf.TTS(
            voice=VOICE_LEARN,
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        userdata=initial_userdata,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info("Usage summary: %s", usage_collector.get_summary())

    ctx.add_shutdown_callback(log_usage)

    # Create agent and give it the session (session is also accessible as self.session)
    agent = Assistant()

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
