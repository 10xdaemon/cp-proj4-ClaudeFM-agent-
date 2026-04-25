import os
import json
import anthropic
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv
from tavily import TavilyClient
from src.recommender import Song, UserProfile, Recommender
from src.spotify_client import get_token, fetch_recommendations
from src.scorer import gaussian_score_normalized, llm_relevance_batch, blend
from src.guardrails import validate_query, validate_profile, confidence_score

load_dotenv()


@dataclass
class AgentResult:
    songs: list[Song] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)   # top 3 only
    reasoning_steps: list[dict] = field(default_factory=list)  # [{tool, input, output}]
    guardrail_warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0


_TOOLS = [
    {
        "name": "parse_user_intent",
        "description": (
            "Parse the user's playlist query into a structured profile and seed info. "
            "Always call this first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "preferred_mood": {
                    "type": "string",
                    "description": (
                        "One mood from the graph: euphoric, happy, uplifted, groovy, "
                        "relaxed, romantic, bittersweet, chill, nostalgic, peaceful, "
                        "focused, moody, melancholic, intense, dark, angry"
                    ),
                },
                "preferred_genre": {
                    "type": "string",
                    "description": "Primary genre e.g. pop, rock, lofi, hip-hop, electronic",
                },
                "target_energy": {"type": "number", "description": "0.0–1.0"},
                "target_tempo_bpm": {"type": "number", "description": "52–168 BPM"},
                "target_acousticness": {"type": "number", "description": "0.0–1.0"},
                "target_speechiness": {"type": "number", "description": "0.0–1.0"},
                "sigma": {
                    "type": "number",
                    "description": "Gaussian tolerance: 0.1 = strict, 0.4 = loose",
                },
                "seed_artists": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Artist names mentioned or implied",
                },
                "seed_genres": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Genre keywords for Spotify search",
                },
                "user_description": {
                    "type": "string",
                    "description": "One-sentence summary of what the user wants",
                },
            },
            "required": [
                "preferred_mood", "preferred_genre", "target_energy",
                "target_tempo_bpm", "target_acousticness", "target_speechiness",
                "sigma", "seed_artists", "seed_genres", "user_description",
            ],
        },
    },
    {
        "name": "tavily_search",
        "description": "Search the web for editorial context about the music request.",
        "input_schema": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "e.g. 'best chill lofi music for studying 2024'",
                }
            },
            "required": ["search_query"],
        },
    },
    {
        "name": "spotify_fetch",
        "description": "Fetch 10 song candidates from Spotify. Call after tavily_search.",
        "input_schema": {
            "type": "object",
            "properties": {
                "seed_artists": {"type": "array", "items": {"type": "string"}},
                "seed_genres": {"type": "array", "items": {"type": "string"}},
                "target_energy": {"type": "number"},
                "target_tempo": {"type": "number"},
                "target_acousticness": {"type": "number"},
            },
            "required": ["seed_artists", "seed_genres"],
        },
    },
    {
        "name": "score_songs",
        "description": (
            "Score and rank fetched songs using Gaussian + LLM hybrid scoring. "
            "Call after spotify_fetch."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_description": {"type": "string"},
                "editorial_context": {"type": "string"},
            },
            "required": ["user_description", "editorial_context"],
        },
    },
    {
        "name": "explain_results",
        "description": "Generate score-breakdown explanations for the top 3 songs. Call last.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

_SYSTEM_PROMPT = (
    "You are a music playlist assistant. Given the user's query, call the tools "
    "in this exact order: parse_user_intent → tavily_search → spotify_fetch → "
    "score_songs → explain_results. Do not skip any step. After explain_results, "
    "write a short friendly summary of the resulting playlist."
)


def run_agent(query: str, gaussian_weight: float = 0.5) -> AgentResult:
    """
    Run the Claude agentic loop for a user playlist query.

    Steps the agent executes via tool calls:
    1. parse_user_intent  → UserProfile + seed info
    2. tavily_search      → editorial context string
    3. spotify_fetch      → list of Song candidates
    4. score_songs        → hybrid blended scores
    5. explain_results    → one-line explanations for top 3 songs

    gaussian_weight: 0.0 = full LLM scoring, 1.0 = full Gaussian scoring.
    Returns an AgentResult with all intermediate steps captured.
    """
    result = AgentResult()
    result.guardrail_warnings.extend(validate_query(query))

    # Shared mutable state across tool calls
    state: dict = {
        "profile": None,
        "user_description": "",
        "editorial_context": "",
        "seed_artists": [],
        "seed_genres": [],
        "songs": [],
        "blended_scores": [],
        "explanations": [],
        "spotify_token": get_token(),
    }

    anthropic_client = anthropic.Anthropic()
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def _execute_tool(name: str, inputs: dict) -> str:
        if name == "parse_user_intent":
            profile = UserProfile(
                preferred_mood=inputs["preferred_mood"],
                preferred_genre=inputs["preferred_genre"],
                target_energy=float(inputs["target_energy"]),
                target_tempo_bpm=float(inputs["target_tempo_bpm"]),
                target_acousticness=float(inputs["target_acousticness"]),
                target_speechiness=float(inputs["target_speechiness"]),
                sigma=float(inputs["sigma"]),
            )
            warnings = validate_profile(profile)
            result.guardrail_warnings.extend(warnings)
            state["profile"] = profile
            state["user_description"] = inputs["user_description"]
            state["seed_artists"] = inputs.get("seed_artists", [])
            state["seed_genres"] = inputs.get("seed_genres", [])
            return json.dumps({"status": "ok", "warnings": warnings})

        if name == "tavily_search":
            search_result = tavily_client.search(inputs["search_query"], max_results=3)
            snippets = [r.get("content", "") for r in search_result.get("results", [])]
            editorial_context = " ".join(snippets)[:1500]
            state["editorial_context"] = editorial_context
            return editorial_context

        if name == "spotify_fetch":
            songs = fetch_recommendations(
                token=state["spotify_token"],
                seed_artists=inputs.get("seed_artists") or state["seed_artists"],
                seed_genres=inputs.get("seed_genres") or state["seed_genres"],
                target_energy=float(inputs.get("target_energy", 0.5)),
                target_tempo=float(inputs.get("target_tempo", 100.0)),
                target_acousticness=float(inputs.get("target_acousticness", 0.5)),
                limit=10,
            )
            state["songs"] = songs
            summaries = [
                {"title": s.title, "artist": s.artist, "genre": s.genre, "mood": s.mood}
                for s in songs
            ]
            return json.dumps({"count": len(songs), "songs": summaries})

        if name == "score_songs":
            songs = state["songs"]
            profile = state["profile"]
            if not songs or not profile:
                return json.dumps({"error": "Missing songs or profile."})

            user_description = inputs.get("user_description") or state["user_description"]
            editorial_context = inputs.get("editorial_context") or state["editorial_context"]

            g_scores = [gaussian_score_normalized(profile, s) for s in songs]
            llm_scores = llm_relevance_batch(
                songs, editorial_context, user_description, anthropic_client
            )
            blended = blend(g_scores, llm_scores, gaussian_weight)

            ranked = sorted(zip(songs, blended), key=lambda x: x[1], reverse=True)
            state["songs"] = [s for s, _ in ranked]
            state["blended_scores"] = [sc for _, sc in ranked]

            top_summaries = [
                {"title": s.title, "artist": s.artist, "score": round(sc, 3)}
                for s, sc in ranked[:10]
            ]
            return json.dumps({"ranked_songs": top_summaries})

        if name == "explain_results":
            songs = state["songs"][:3]
            profile = state["profile"]
            if not songs or not profile:
                return json.dumps({"error": "No ranked songs available."})

            explanations = []
            for song in songs:
                _, reasons = Recommender.score(profile, song)
                explanations.append(" | ".join(reasons))

            state["explanations"] = explanations
            return json.dumps({"explanations": explanations})

        return json.dumps({"error": f"Unknown tool: {name}"})

    # --- Agentic loop ---
    messages: list[Any] = [{"role": "user", "content": query}]

    while True:
        response = anthropic_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            tools=_TOOLS,  # type: ignore[arg-type]
            messages=messages,  # type: ignore[arg-type]
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break

        tool_results = []
        for tu in tool_uses:
            output = _execute_tool(tu.name, tu.input)
            result.reasoning_steps.append({
                "tool": tu.name,
                "input": tu.input,
                "output": output,
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": output,
            })

        messages.append({"role": "user", "content": tool_results})

    # --- Populate AgentResult from final state ---
    result.songs = state["songs"]
    result.scores = state["blended_scores"]
    result.explanations = state["explanations"]
    if result.scores:
        result.confidence = confidence_score(result.scores)

    return result
