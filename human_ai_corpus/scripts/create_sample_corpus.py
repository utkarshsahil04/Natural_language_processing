"""
Phase 1 helper - build a larger sample corpus for full pipeline testing.
Creates ~180 human + ~180 AI style responses across the prompt set.
"""

from pathlib import Path
import itertools
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
PROMPTS = RAW / "prompts.csv"

# Short informal human-like templates by category
HUMAN_TEMPLATES = {
    "advice": [
        "honestly i just {a}. works better than fancy plans for me.",
        "tbh {a}. also {b}. not perfect but it helps.",
        "i usually {a}. sometimes {b}. depends on the day.",
        "my tip is simple: {a}. dont overcomplicate it.",
        "idk what experts say but for me {a}. and yeah {b}.",
    ],
    "factual": [
        "ok so basically {a}. thats the simple version.",
        "from what i remember {a}. also {b}.",
        "short answer: {a}. longer answer gets messy lol.",
        "i think {a}. not 100% sure on details though.",
        "{a}. {b}. hope that makes sense.",
    ],
    "creative": [
        "alright here goes - {a}. then suddenly {b}.",
        "picture this: {a}. later {b}.",
        "tiny story: {a}. in the end {b}.",
        "hmm maybe {a}. and somehow {b}.",
        "ok rough draft: {a}. then {b}. kinda cute ngl.",
    ],
    "opinion": [
        "i feel like {a}. but also {b}. so mixed.",
        "personally {a}. others might disagree.",
        "depends. for me {a}. still {b} matters though.",
        "nah for me {a}. online takes can be dramatic.",
        "honest take: {a}. and yeah {b}.",
    ],
}

HUMAN_FILLERS = {
    "advice": [
        ("make tiny goals", "take breaks without guilt"),
        ("sleep properly", "talk to a friend when stuck"),
        ("keep a messy notebook", "reward myself after finishing"),
        ("study in short bursts", "put phone in another room"),
        ("ask seniors for tips", "stop comparing marks"),
    ],
    "factual": [
        ("it is basically a process that repeats", "small changes add up over time"),
        ("there are a few main parts", "people overcomplicate the definition"),
        ("the simple idea is cause and effect", "examples help more than theory"),
        ("it works through steps", "context matters a lot"),
        ("core idea is pretty straightforward", "details depend on the case"),
    ],
    "creative": [
        ("a quiet street after rain", "someone finds what they lost"),
        ("two people share an awkward silence", "they laugh and talk anyway"),
        ("an old object waits somewhere", "its owner comes back surprised"),
        ("the city feels tired and loud", "one small kind moment changes the mood"),
        ("night lights flicker on wet roads", "a stranger helps without asking much"),
    ],
    "opinion": [
        ("it helps if used carefully", "too much of it becomes noise"),
        ("real life practice matters more", "tools are fine with clear rules"),
        ("classroom focus is better for me", "flexibility is still useful"),
        ("grades are not the full story", "effort and curiosity count too"),
        ("freedom is good", "some limits protect people"),
    ],
}

AI_TEMPLATES = {
    "advice": [
        "A practical approach is to {a}. In addition, {b}. Consistency usually matters more than intensity, so track small progress and adjust your routine weekly.",
        "You can improve results by combining structure and recovery: {a}. It also helps to {b}. Clear priorities and adequate rest support long-term motivation.",
        "Start with manageable steps. First, {a}. Next, {b}. Review what worked at the end of each week and refine your plan without harsh self-judgment.",
        "Effective strategies often include planning and accountability. Try to {a}, and whenever possible {b}. Sustainable habits outperform short bursts of pressure.",
        "Consider a balanced routine: {a}. Supporting habits such as sleep and reflection also help when you {b}. Progress compounds through steady practice.",
    ],
    "factual": [
        "In simple terms, {a}. Equally important, {b}. Understanding both the core mechanism and its broader impact makes the topic clearer for beginners.",
        "The concept can be summarized as follows: {a}. Furthermore, {b}. These points provide a concise foundation before exploring advanced details.",
        "At a high level, {a}. Relatedly, {b}. This explanation focuses on clarity rather than technical depth.",
        "A useful definition begins with the idea that {a}. Additionally, {b}. Together, these elements explain why the topic is widely discussed.",
        "Essentially, {a}. It is also useful to note that {b}. Examples and structured explanations usually improve retention.",
    ],
    "creative": [
        "In a short scene, {a}. Moments later, {b}. The ending leaves a quiet sense of resolution without needing dramatic twists.",
        "Imagine a setting where {a}. As the moment unfolds, {b}. The tone stays gentle, focusing on small human details.",
        "A compact narrative begins when {a}. Eventually, {b}. The story emphasizes emotion and atmosphere over complex plot.",
        "Consider this vignette: {a}. Then, unexpectedly yet softly, {b}. Such scenes work best when imagery remains concrete.",
        "The piece opens with {a}. By the close, {b}. A brief creative response can still feel complete through clear imagery.",
    ],
    "opinion": [
        "A balanced view is that {a}. At the same time, {b}. The stronger position usually depends on context, goals, and individual circumstances.",
        "One reasonable perspective is that {a}. However, it is also true that {b}. Careful evaluation of trade-offs leads to a more nuanced conclusion.",
        "In many cases, {a}. Still, {b}. Therefore, absolute claims are less useful than conditional recommendations.",
        "From an evidence-informed standpoint, {a}. Equally, {b}. Policy and personal choices should reflect both benefits and risks.",
        "Overall, {a}. Yet practical constraints mean {b}. A moderate position often serves students and institutions better.",
    ],
}

AI_FILLERS = {
    "advice": [
        ("set specific daily targets", "review progress with a simple checklist"),
        ("protect consistent sleep hours", "use short focused study intervals"),
        ("reduce distractions during work blocks", "ask for feedback from peers"),
        ("break large tasks into smaller actions", "schedule deliberate rest periods"),
        ("prepare materials the night before", "maintain a calm and realistic pace"),
    ],
    "factual": [
        ("the process follows identifiable stages", "outcomes depend on surrounding conditions"),
        ("key components interact in a structured way", "real-world applications vary by context"),
        ("the underlying mechanism is cause and effect", "broader consequences affect communities"),
        ("definitions emphasize function over jargon", "examples make abstract ideas concrete"),
        ("core principles remain relatively stable", "details become clearer with practice"),
    ],
    "creative": [
        ("rain gathers on quiet rooftops", "a familiar object returns to its owner"),
        ("two strangers share a delayed journey", "conversation turns cautious hope into connection"),
        ("an ordinary evening feels unusually vivid", "a small act of kindness closes the scene"),
        ("city lights reflect on wet pavement", "someone rediscovers what mattered most"),
        ("a soft breeze moves through an empty corridor", "the final image settles into calm"),
    ],
    "opinion": [
        ("responsible use creates clear benefits", "unrestricted use can reduce focus"),
        ("structured learning supports deeper understanding", "flexibility remains valuable for access"),
        ("skills development matters beyond scores", "assessment still provides useful signals"),
        ("technology can support learning effectively", "integrity guidelines remain essential"),
        ("individual preference shapes the best option", "shared standards help maintain fairness"),
    ],
}


def build_side(prompts: pd.DataFrame, templates: dict, fillers: dict, source: str, prefix: str, per_prompt: int = 4):
    rows = []
    n = 1
    for _, row in prompts.iterrows():
        cat = row["category"]
        temps = templates[cat]
        fills = fillers[cat]
        pairs = list(itertools.islice(itertools.cycle(fills), per_prompt))
        for i in range(per_prompt):
            a, b = pairs[i]
            text = temps[i % len(temps)].format(a=a, b=b)
            rows.append({
                "response_id": f"{prefix}_{n:04d}",
                "prompt_id": row["prompt_id"],
                "prompt": row["prompt"],
                "response_text": text,
                "source_type": source,
            })
            n += 1
    return pd.DataFrame(rows)


def main():
    prompts = pd.read_csv(PROMPTS)
    human = build_side(prompts, HUMAN_TEMPLATES, HUMAN_FILLERS, "human", "H", per_prompt=4)
    ai = build_side(prompts, AI_TEMPLATES, AI_FILLERS, "AI", "AI", per_prompt=4)
    corpus = pd.concat([human, ai], ignore_index=True)

    RAW.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    human.to_csv(RAW / "human_responses.csv", index=False)
    ai.to_csv(RAW / "ai_responses.csv", index=False)
    corpus.to_csv(PROC / "corpus.csv", index=False)

    print(human["source_type"].value_counts().to_string() if False else "")
    print(f"Human: {len(human)} | AI: {len(ai)} | Total: {len(corpus)}")
    print(f"Saved -> {PROC / 'corpus.csv'}")


if __name__ == "__main__":
    main()
