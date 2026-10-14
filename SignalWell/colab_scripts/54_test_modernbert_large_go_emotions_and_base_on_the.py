# ============================================================
# TEST — ModernBERT-large-go-emotions (and base) on the same 18 sentences
# ============================================================
!pip install -q transformers

from transformers import pipeline

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

def run_and_print(model_name):
    print(f"\n{'='*60}\n{model_name}\n{'='*60}")
    classifier = pipeline("text-classification", model=model_name, top_k=None)
    for text in test_sentences:
        results = sorted(classifier(text)[0], key=lambda x: x["score"], reverse=True)[:3]
        print(f"\n{text}")
        for r in results:
            bar = "█" * int(r["score"] * 30)
            print(f"    {r['label']:<15} {r['score']:.3f}  {bar}")

run_and_print("cirimus/modernbert-large-go-emotions")
run_and_print("cirimus/modernbert-base-go-emotions")
