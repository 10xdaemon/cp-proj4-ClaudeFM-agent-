# Project Completion Plan

## Environment
- [x] Virtual environment (`.venv`) created
- [x] Install dependencies — `pip install -r requirements.txt`
- [x] Create `.env` file with API keys (`ANTHROPIC_API_KEY`, `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `TAVILY_API_KEY`)

## Core Scorer — `src/scorer.py`
- [ ] `gaussian_score_normalized()` — call `Recommender.score()` and divide by 8.0
- [ ] `llm_relevance_batch()` — build batch prompt, call Claude API, parse JSON scores
- [ ] `blend()` — weighted average of Gaussian and LLM scores

## Agent Loop — `src/agent.py`
- [ ] `run_agent()` — implement the full Claude tool-calling agentic loop (parse intent → tavily search → Spotify fetch → score → explain)

## Streamlit UI — `app.py`
- [ ] Import and wire `run_agent` once agent is implemented
- [ ] Fix preset query buttons (populate `st.session_state["query"]` on click)
- [ ] Call `run_agent()` on submit and store result
- [ ] Display guardrail warnings (`result.guardrail_warnings`)
- [ ] Render agent reasoning trace (`result.reasoning_steps`)
- [ ] Show confidence badge (High / Medium / Low)
- [ ] Replace placeholder song cards with `result.songs[:3]` and `result.explanations`
- [ ] Render songs 4–10 as compact list (`result.songs[3:]`)
- [ ] Implement Load More pagination

## Tests
- [ ] Run existing tests and confirm they pass — `pytest`
- [ ] Add tests for `scorer.py` (`gaussian_score_normalized`, `blend`)
- [ ] Add tests for `guardrails.py` (`validate_profile`, `confidence_score`, `genre_dominance_flag`)
- [ ] Implement `tests/eval_harness.py` — run adversarial profiles and log results

## Docs (last)
- [ ] Rewrite `README.md` to reflect full project scope
- [ ] Update `model_card.md` to reflect full system
