# ============================================================
# WIDER TEST — SamLowe on the same 18 sentences, for direct comparison
# ============================================================
!pip install -q transformers

from transformers import pipeline

go_emotions = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None
)

test_sentences = [
    "Had coffee with an old friend today, it felt so good to catch up.",
    "Finally finished the project I've been putting off for weeks.",
    "I keep replaying that conversation in my head and it's driving me crazy.",
    "Nothing seems to be going right this week.",
    "Proud of myself for finishing the race but my legs are killing me.",
    "Excited about the new job but honestly a little scared too.",
    "Did laundry, made dinner, watched some TV. Pretty normal day.",
    "Woke up late again, need to fix my sleep schedule.",
    "Can't stop worrying about the presentation tomorrow.",
    "My heart's been racing all day for no real reason.",
    "So annoyed that they cancelled last minute again.",
    "I'm furious about how that meeting went.",
    "Really grateful for my sister checking in on me today.",
    "Small things like a good cup of tea make such a difference.",
    "It's been a year since he passed and it still hits me sometimes.",
    "Tired. Just tired.",
    "Okay day I guess.",
    "Actually kind of a good one today :)",
]

for text in test_sentences:
    results = go_emotions(text)[0]
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)[:3]
    print(f"\n{text}")
    for r in results_sorted:
        bar = "█" * int(r["score"] * 30)
        print(f"    {r['label']:<15} {r['score']:.3f}  {bar}")
