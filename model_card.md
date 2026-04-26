# 🎧 Model Card: Local-Gaussian Scorer

## 1. Model Name

**RubberSoul v1** — Gaussian scoring component of OpenFM-agent

---

## 2. Intended Use

A scoring based recommender that ranks songs against a structured user taste profile. It assumes the user can describe what they want upfront and takes those inputs at face value without learning or adapting over time. This card covers the Gaussian scoring component (`recommender.py`) specifically — in the final system it runs as one half of a hybrid scorer alongside a Claude LLM scorer, with results blended via a user controlled weight.

---

## 3. How the Model Works

You give it a profile: a genre, mood, target values for energy, tempo, etc... For each song in the catalog, it adds up points across six dimensions. Genre is worth the most, an exact match gives 2 points, no match gives nothing. Mood uses a short adjacency graph, so an "intense" user still gets partial credit for a "focused" song since those moods are "one" step apart so to speak. Energy, acousticness, tempo, and speechiness are each scored with a bell curve: a perfect match earns the full weight, and the score falls off smoothly the farther the song is from the target. Everything sums to a maximum of `8.0`. The top results are then filtered so no more than two songs from the same genre appear back to back, adding a little variety.

---

## 4. Data

In the original simulation, the scorer ran against a static 20 song catalog (`songs.csv`) spanning 17 genres and 16 moods. In the final system, that catalog is replaced by live Spotify results — up to 10 tracks fetched per query. Because Spotify's `/audio-features` endpoint requires user authentication (unavailable under Client Credentials), audio features are not pulled per track; instead they are inferred from genre defaults in `_GENRE_DEFAULTS` and spread across results using search position as a relevance proxy. The mood and genre constraints from the original catalog no longer apply, but the inferred features introduce their own bias: every track in the same genre starts from the same baseline values before position adjustment.

---

## 5. Strengths

The system works best for users with a clear, coherent profile that lines up with a well represented genre eg. a pop or lofi listener with typical energy preferences will consistently get results that feel right. The mood graph is a genuine strength: rather than requiring an exact mood match, it rewards nearby moods proportionally, so a "relaxed" user still surfaces "chill" and "happy" songs without listing them explicitly. The Gaussian scoring also handles imprecision gracefully. A song doesn't have to perfectly hit a target energy to score well, it just needs to be close.

---

## 6. Limitations and Bias

The genre bonus is the single largest scoring component, worth +2.0 out of a maximum of 8.0 (25%), and is awarded as a binary all or nothing match rather than a gradual signal. This creates an uneven playing field based on catalog representation. As stated before, a user who prefers pop or lofi receives that bonus frequently because those genres appear multiple times in the dataset, while a user who prefers an underrepresented genre; like bossa nova or soul, never receives it at all, pushing every recommendation down by 2.0 points from the start. <br>

During adversarial testing this became concrete: a high energy profile still ranked *Spacewalk Thoughts* (ambient, energy proximity: +0.01) at #3 solely because its genre matched, meaning the genre bonus overrode nearly zero compatibility on every other dimension. In effect, the scoring does not just reflect how well a song fits a user, it also reflects how well the catalog was stocked with that user's genre to begin with.

---

## 7. Evaluation

Three baseline profiles were run first:
1. high-energy pop
1. chill lofi
1. intense rock,

to confirm if the results felt intuitive. Then seven adversarial profiles were designed to probe the edges. The results were surprising to say the least. <br>
### The edges

1. a mood the system doesn't recognize ("sad")
1. a sigma of zero
1. a target tempo above the catalog's ceiling (220 BPM)
1. contradictory high energy and high acousticness targets
1. a genre absent from the catalog, and both an extremely tight and extremely wide sigma

Each run was checked by reading the "per component" score breakdown for all five results, not just the rankings. The most surprising finding was that `sigma = 0` caused an outright crash, and that `sigma = 5.0` gave a song with energy `0.55` a near perfect energy score of `+1.99` when the target was `0.97`, the recommendation looked confident while quietly ignoring the mismatch.

---

## 8. Future Work

~~The most urgent fix would be guarding against a sigma of zero, which currently crashes the program.~~ **Resolved** — `sigma=0` is now auto-clamped to `0.01` in `guardrails.py`. The scoring would still benefit from including `valence` and `danceability`, which are already in the dataset but ignored entirely. The mood graph could also be expanded to cover moods users naturally reach for like "sad", that currently score zero without any warning.

---

## 9. Personal Reflection 
*original*:
The biggest takeaway was how much a single design decision (the +2.0 genre bonus) can quietly shape every result without being obvious from the outside. It also showed that a score can look very confident while failing on the dimension the user actually cares about most, which is exactly what makes real recommender systems hard to trust at times. It changed how I think about apps like Spotify and Youtube Music. Behind every "because you listened to X" label is probably a scoring system with similar quirks, just with millions of songs smoothing them out.

