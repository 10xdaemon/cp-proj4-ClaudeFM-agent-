import threading
import time
import random
import streamlit as st
from src.agent import run_agent, AgentResult
from src.spotify_client import get_token, fetch_recommendations
from src.scorer import gaussian_score_normalized
from src.recommender import UserProfile

st.set_page_config(page_title="Claude FM", page_icon="📻", layout="wide")

# ── Session state initialization ───────────────────────────────────────────────
if "query" not in st.session_state:
    st.session_state.query = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_weight" not in st.session_state:
    st.session_state.last_weight = 0.5
if "extra_songs" not in st.session_state:
    st.session_state.extra_songs = []
if "extra_scores" not in st.session_state:
    st.session_state.extra_scores = []
if "seeds" not in st.session_state:
    st.session_state.seeds = {"artists": [], "genres": []}
if "profile" not in st.session_state:
    st.session_state.profile = None
if "seen_ids" not in st.session_state:
    st.session_state.seen_ids = set()
if "result_query" not in st.session_state:
    st.session_state.result_query = ""

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    gaussian_weight = st.slider(
        "dev's choice  ←→ agentic search ",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key="gaussian_weight",
        help="0 = full LLM scoring · 1 = full Gaussian scoring",
    )

    st.markdown("---")
    st.subheader("Preset Queries")

    if st.button("Study session"):
        st.session_state.query = "create a playlist for me to study to"
        st.rerun()
    if st.button("Party vibes"):
        st.session_state.query = "I'm at a party, play something energetic"
        st.rerun()
    if st.button("Late-night drive"):
        st.session_state.query = "make me a playlist for a late-night drive"
        st.rerun()

    st.markdown("---")
    st.caption("Authored by 10xdaemon & Claude Code")

# ── Main ──────────────────────────────────────────────────────────────────────
left, center, right = st.columns([1, 2, 1])
with center:
    st.markdown("<h1 style='text-align:center'>💽 Claude FM</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:gray; margin-top:-8px'>"
        "Type what you're about to do and get a personalized playlist.</p>",
        unsafe_allow_html=True,
    )

    query = st.text_area(
        "What are you up to?",
        key="query",
        placeholder="e.g. 'create a playlist for studying' or 'my friend loves Bad Bunny, we're at a party'",
        height=80,
    )

    _, btn_col = st.columns([3, 1])
    with btn_col:
        submitted = st.button("Build Playlist", type="primary", use_container_width=True)

if submitted and query.strip():
    cache_miss = (
        query != st.session_state.last_query
        or gaussian_weight != st.session_state.last_weight
    )
    if cache_miss:
        st.session_state.result = None

        _MUSICAL_FACTS = [
            "The world's oldest known song is 3,400 years old: a Hurrian hymn carved on a clay tablet.",
            "A piano has 12,000 parts, 10,000 of which are moving.",
            "'Happy Birthday to You' was the first song broadcast from space (1965).",
            "The loudest band ever recorded is KISS at 136 dB, louder than a jet engine.",
            "Beethoven was completely deaf when he composed his 9th Symphony.",
            "The 'Mozart Effect' was later debunked. Music won't make you smarter, but it will make you happier.",
            "Vinyl records outsold CDs for the first time since 1987 in 2022.",
            "A song stuck in your head is called an 'earworm' (Ohrwurm in German).",
            "The most covered song ever is 'Yesterday' by The Beatles...OVER 2,200 recorded versions.",
            "Listening to music releases dopamine, the same chemical triggered by eating chocolate.",
            "The electric guitar was invented in 1931 by George Beauchamp.",
            "'Bohemian Rhapsody' was rejected by radio stations for being too long. It became a #1 hit anyway.",
            "The human ear can distinguish over 400,000 different sounds.",
            "Hip-hop is now the most streamed genre globally, overtaking rock.",
            "Some whales sing songs that last up to 22 hours.",
        ]

        _, _c, _ = st.columns([1, 2, 1])
        with _c:
            st.markdown(
                "<style>[data-testid='stSpinner']>div{justify-content:center}</style>",
                unsafe_allow_html=True,
            )
            result_container: dict = {}

            def _run_agent() -> None:
                result_container["result"] = run_agent(query, gaussian_weight)

            thread = threading.Thread(target=_run_agent, daemon=True)
            thread.start()
            _shuffled = _MUSICAL_FACTS.copy()
            random.shuffle(_shuffled)
            with st.spinner("Building your playlist..."):
                fact_placeholder = st.empty()
                fact_idx = 0
                while thread.is_alive():
                    fact_placeholder.markdown(
                        f"<p style='text-align:center;color:gray'>🔉 {_shuffled[fact_idx % len(_shuffled)]}</p>",
                        unsafe_allow_html=True,
                    )
                    fact_idx += 1
                    if fact_idx % len(_shuffled) == 0:
                        random.shuffle(_shuffled)
                    time.sleep(6.6)
            fact_placeholder.empty()

        st.session_state.result = result_container["result"]
        st.session_state.result_query = query
        st.session_state.last_query = query
        st.session_state.last_weight = gaussian_weight
        st.session_state.extra_songs = []
        st.session_state.extra_scores = []
        st.session_state.seen_ids = {s.spotify_id for s in st.session_state.result.songs if s.spotify_id}
        for step in st.session_state.result.reasoning_steps:
            if step["tool"] == "parse_user_intent":
                inp = step["input"]
                st.session_state.seeds = {
                    "artists": inp.get("seed_artists", []),
                    "genres": inp.get("seed_genres", []),
                }
                st.session_state.profile = UserProfile(
                    preferred_mood=inp["preferred_mood"],
                    preferred_genre=inp["preferred_genre"],
                    target_energy=float(inp["target_energy"]),
                    target_tempo_bpm=float(inp["target_tempo_bpm"]),
                    target_acousticness=float(inp["target_acousticness"]),
                    target_speechiness=float(inp["target_speechiness"]),
                    sigma=float(inp["sigma"]),
                )

elif submitted and not query.strip():
    st.warning("Please enter a query before building a playlist.")

# ── Results ───────────────────────────────────────────────────────────────────
result: AgentResult | None = st.session_state.result
if result is not None:
    # ── Guardrail warnings ──────────────────────────────────────────────────
    if result.guardrail_warnings:
        for w in result.guardrail_warnings:
            st.warning(w)

    # ── Agent reasoning trace ────────────────────────────────────────────────
    _, _exp_c, _ = st.columns([1, 2, 1])
    with _exp_c:
        with st.expander(f"🤖 Under the hood : \"{st.session_state.result_query}\"", expanded=False):
            for step in result.reasoning_steps:
                st.markdown(f"**`{step['tool']}`**")
                st.json(step["input"], expanded=False)
                st.caption(str(step["output"])[:300])
                st.divider()

    # ── Top 3 songs: cover art + explanation ─────────────────────────────────
    left, center, right = st.columns([1, 3, 1])
    with center:
        st.subheader("Your Playlist")
        all_songs = result.songs + st.session_state.extra_songs
        all_scores = result.scores + st.session_state.extra_scores

        if result.songs:
            cols = st.columns(3)
            for i, (col, song, score, explanation) in enumerate(
                zip(cols, result.songs[:3], result.scores[:3], result.explanations)
            ):
                with col:
                    img_url = song.cover_art_url or f"https://placehold.co/200x200?text={i + 1}"
                    st.image(img_url, width=200)
                    st.markdown(f"**{song.title}**")
                    st.caption(f"{song.artist} · {song.genre}")
                    st.progress(min(max(score, 0.0), 1.0), text=f"{score:.2f}")
                    st.markdown(f"_{explanation}_")

    # ── Songs 4–10: compact list ─────────────────────────────────────────────
    if len(all_songs) > 3:
        st.markdown("#### More tracks")
        for i, (song, score) in enumerate(zip(all_songs[3:], all_scores[3:]), start=4):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{i}.** {song.title} — *{song.artist}*")
            c2.markdown(f"`{score:.2f}`")

    # ── Load More ────────────────────────────────────────────────────────────
    if st.button("Load More"):
        with st.spinner("Fetching more tracks..."):
            token = get_token()
            seeds = st.session_state.seeds
            more = fetch_recommendations(
                token=token,
                seed_artists=seeds["artists"],
                seed_genres=seeds["genres"],
                limit=10,
            )
            profile = st.session_state.profile
            if profile and more:
                g_scores = [gaussian_score_normalized(profile, s) for s in more]
                ranked = sorted(zip(more, g_scores), key=lambda x: x[1], reverse=True)
                new_ranked = [(s, sc) for s, sc in ranked if s.spotify_id not in st.session_state.seen_ids]
                st.session_state.extra_songs += [s for s, _ in new_ranked]
                st.session_state.extra_scores += [sc for _, sc in new_ranked]
                st.session_state.seen_ids.update(s.spotify_id for s, _ in new_ranked if s.spotify_id)
        st.rerun()
