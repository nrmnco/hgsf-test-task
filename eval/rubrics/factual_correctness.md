# Rubric: Factual Correctness

## Task
You are evaluating a research agent's answer against a reference answer for a given question.
The agent searched a local corpus, fetched pages, and synthesized a cited answer.

## Scoring

**2 — Correct**
All key facts from the reference answer are present in the agent's answer.
Minor differences in wording, phrasing, or additional correct detail are fine.
Nothing in the agent's answer contradicts the reference.

**1 — Partial**
Some key facts are present but the answer is meaningfully incomplete.
OR there is a minor factual error that does not change the overall conclusion.
OR the agent gave a vague answer when a specific one was available in the corpus.

**0 — Wrong**
The agent's answer contradicts the reference on a key fact.
OR the agent claimed the corpus did not contain the answer when it did.
OR the answer is entirely off-topic or fabricated.
OR the agent produced no answer at all.

## Important notes
- Do not penalise the agent for including extra correct information not in the reference.
- Do not penalise for different but equivalent phrasing (e.g. "August 2012" vs "2012").
- Do penalise if the agent expressed false uncertainty ("I could not find...") when the reference shows the answer exists.
- For unanswerable questions, score 2 if the agent correctly says it cannot answer, 0 if it fabricates an answer.

## Output format
Respond with valid JSON only. No extra text.
{"score": <0|1|2>, "rationale": "<one or two sentences explaining the score>"}
