import json
import anthropic
from src.recommender import Song, UserProfile, Recommender


def gaussian_score_normalized(user: UserProfile, song: Song) -> float:
    """
    Run the existing Gaussian + mood/genre scoring on a single song.
    Returns a value in 0.0–1.0 (raw score divided by max 8.0).
    """
    raw, _ = Recommender.score(user, song)
    normalized = raw / 8.0
    return normalized


def llm_relevance_batch(
    songs: list[Song],
    editorial_context: str,
    user_description: str,
    client: anthropic.Anthropic,
) -> list[float]:
    """
    Ask Claude to rate each song 0–1 for how well it fits the user and editorial context.
    Uses a single API call with all songs in the prompt. Returns a list parallel to songs.
    """
    song_lines = []
    for i, song in enumerate(songs):
        song_lines.append(
            f"{i}. \"{song.title}\" by {song.artist} "
            f"(genre: {song.genre}, mood: {song.mood}, "
            f"energy: {song.energy:.2f}, tempo: {song.tempo_bpm:.0f} BPM)"
        )
    songs_block = "\n".join(song_lines)

    prompt = (
        f"User description: {user_description}\n\n"
        f"Editorial context: {editorial_context}\n\n"
        f"Songs:\n{songs_block}\n\n"
        "Rate how well each song fits the user's taste and editorial context "
        "on a scale from 0.0 (no fit) to 1.0 (perfect fit). "
        "Return a JSON object with a single key 'scores' containing an array "
        "of numbers in the same order as the songs listed above."
    )

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "scores": {
                            "type": "array",
                            "items": {"type": "number"},
                        }
                    },
                    "required": ["scores"],
                    "additionalProperties": False,
                },
            }
        },
    )

    data = json.loads(response.content[0].text)
    scores = [float(s) for s in data["scores"]]
    return scores


def blend(
    g_scores: list[float],
    llm_scores: list[float],
    gaussian_weight: float,
) -> list[float]:
    """
    Combine Gaussian and LLM scores using a weighted average.
    gaussian_weight=1.0 means pure algorithm; 0.0 means pure LLM.
    Returns a list of blended scores parallel to the input lists.
    """
    blended = []
    for g, llm in zip(g_scores, llm_scores):
        score = (gaussian_weight * g) + ((1 - gaussian_weight) * llm)
        blended.append(score)
    return blended
