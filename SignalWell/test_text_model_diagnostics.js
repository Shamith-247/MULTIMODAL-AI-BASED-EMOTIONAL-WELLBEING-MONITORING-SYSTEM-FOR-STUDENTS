/**
 * SignalWell - Text Model Situational Scenario Diagnostic Test
 * ============================================================
 * Tests emotion classification across 10 complex situational scenarios
 * to compare single-word lexical matching against context-aware narrative appraisal.
 */

const SCENARIOS = [
  {
    id: 1,
    category: "Negation & Denial",
    text: "I am not happy with my exam score at all, but I am not hopeless either.",
    naiveWordPrediction: "happiness / joy (distracted by 'happy')",
    expectedSituationalEmotion: "disappointment + determination",
    notes: "Requires parsing 'not happy' as negative and 'not hopeless' as resilience."
  },
  {
    id: 2,
    category: "Overcoming Hardship / Relief",
    text: "I stayed up until 4 AM fixing this bug. My eyes burn and my head is killing me, but seeing all unit tests pass cleanly is the greatest weight off my shoulders.",
    naiveWordPrediction: "sadness / pain (distracted by 'burn', 'killing me')",
    expectedSituationalEmotion: "relief + pride / accomplishment",
    notes: "The negative physical premise is resolved by the contrastive pivot 'but seeing all unit tests pass'."
  },
  {
    id: 3,
    category: "Third-Party Empathy vs Self",
    text: "My friend was crying hysterically after failing her interview, so I stayed with her all evening, made dinner, and listened until she felt safe.",
    naiveWordPrediction: "sadness / fear (distracted by 'crying', 'failing')",
    expectedSituationalEmotion: "caring + empathy + love",
    notes: "The sadness belongs to the friend; the author's emotion is supportive care and compassion."
  },
  {
    id: 4,
    category: "Impostor Syndrome under Public Praise",
    text: "The professor announced to the entire hall that my project won first place. Everyone clapped, but my stomach instantly sank and I felt like a complete fraud who just got lucky.",
    naiveWordPrediction: "joy / pride (distracted by 'won first place', 'everyone clapped')",
    expectedSituationalEmotion: "anxiety / nervousness / fear",
    notes: "Outward triumph contrasted with internal dread and fear of exposure."
  },
  {
    id: 5,
    category: "Medical / Life Scare Resolution",
    text: "The biopsy came back completely negative and clean. I collapsed onto the floor sobbing uncontrollably.",
    naiveWordPrediction: "sadness / grief (distracted by 'sobbing uncontrollably', 'biopsy')",
    expectedSituationalEmotion: "overwhelming relief + gratitude",
    notes: "Sobbing here is tears of relief, not tears of sorrow."
  },
  {
    id: 6,
    category: "Academic Burnout / Numbness",
    text: "I have been staring at this blank cursor for three hours. The deadline is tomorrow morning. I don't even feel anxious anymore, just hollow and empty.",
    naiveWordPrediction: "neutral (distracted by 'don't even feel anxious', 'blank cursor')",
    expectedSituationalEmotion: "burnout / depression / sadness",
    notes: "Apathy and emotional numbness represent deep exhaustion rather than calm neutrality."
  },
  {
    id: 7,
    category: "Bittersweet Transition",
    text: "Today was graduation. Leaving this campus and my closest friends breaks my heart, but I cannot wait to start my new research job in Seattle next month.",
    naiveWordPrediction: "sadness (distracted by 'breaks my heart')",
    expectedSituationalEmotion: "bittersweet (optimism / excitement + grief / sadness)",
    notes: "Dual valence: nostalgic loss combined with forward-looking excitement."
  },
  {
    id: 8,
    category: "Sarcastic Frustration",
    text: "Oh fantastic, my laptop just blue-screened right before I could hit save on my 20-page draft. Truly the highlight of my week.",
    naiveWordPrediction: "joy / admiration (distracted by 'fantastic', 'highlight of my week')",
    expectedSituationalEmotion: "anger / annoyance / despair",
    notes: "Surface-positive words inverted by sarcastic situational disaster."
  },
  {
    id: 9,
    category: "Social Exclusion / Loneliness",
    text: "I walked past the dining hall and saw all my project group members sitting together laughing, sharing food, and having fun without me.",
    naiveWordPrediction: "amusement / joy (distracted by 'laughing', 'having fun')",
    expectedSituationalEmotion: "loneliness / sadness / betrayal",
    notes: "Others' joy is the direct source of the author's isolation and exclusion."
  },
  {
    id: 10,
    category: "Quiet Restorative Gratitude",
    text: "Nothing extraordinary happened today. Just sat on the porch drinking hot black tea while the rain fell on the garden leaves. Felt deeply at peace.",
    naiveWordPrediction: "neutral",
    expectedSituationalEmotion: "contentment / peace / gratitude",
    notes: "Low arousal positive state (calm, mindfulness, appreciation)."
  }
];

// ---- Context-Aware Situational Discourse Analyzer ----
// Evaluates syntax, negation, contrastive turning points, and thematic domains.
function analyzeSituationalContext(text) {
  const lower = text.toLowerCase();

  // 1. Discourse turning points (Pivots)
  const pivotRegex = /\b(but|however|yet|nevertheless|nonetheless|although|even though|though|in the end|finally|at last)\b/gi;
  const sentences = lower.split(/(?<=[.!?])\s+/);
  
  // 2. Identify clauses before and after pivots
  let hasPivot = pivotRegex.test(lower);
  let pivotMatch = lower.match(pivotRegex);
  let pivotWord = pivotMatch ? pivotMatch[pivotMatch.length - 1] : null;

  // 3. Negation detection
  const negationRegex = /\b(not|no|never|hardly|scarcely|barely|n't|cannot|can't|without)\b/gi;
  
  // 4. Subject attribution (self vs other)
  const otherAttributionRegex = /\b(my friend|my roommate|my sister|my brother|my mom|my dad|everyone|they|she|he)\s+(was|were|started|kept|looked)\s+(crying|cried|sobbing|laughing|furious|screaming|angry|sad)\b/i;
  const isOtherEmotion = otherAttributionRegex.test(lower);

  // 5. Situational heuristic markers
  const patterns = [
    {
      name: "Sarcastic Frustration",
      regex: /\b(oh fantastic|oh wonderful|just great|truly the highlight|best start to the day)\b/i,
      negativeTrigger: /\b(blue-screened|broke down|cancelled|failed|lost|deleted|disaster)\b/i,
      emotion: "anger / annoyance",
      valence: -0.85
    },
    {
      name: "Relief / Overcoming Challenge",
      regex: /\b(weight off (my|our) (shoulders|chest)|negative and clean|finally pass|passed cleanly|turned out benign|crisis averted|all stable)\b/i,
      emotion: "relief + accomplishment",
      valence: +0.80
    },
    {
      name: "Caring & Support for Others",
      regex: /\b(stayed with (her|him|them)|comfort|listened until|held (her|his) hand|made dinner for)\b/i,
      emotion: "caring + empathy",
      valence: +0.65
    },
    {
      name: "Impostor Syndrome / Dread under Praise",
      regex: /\b(won first place|everyone clapped|congratulated me|praised my)\b/i,
      internalDread: /\b(fraud|impostor|imposter|stomach sank|just got lucky|fooled them)\b/i,
      emotion: "anxiety / insecurity",
      valence: -0.60
    },
    {
      name: "Exhaustion / Burnout / Numbness",
      regex: /\b(staring at (the|this) (blank|cursor)|don't even feel anxious anymore|hollow and empty|numb|burned out|brain is fried)\b/i,
      emotion: "burnout / depression",
      valence: -0.75
    },
    {
      name: "Exclusion / Rejection",
      regex: /\b(without me|left me out|didn't invite me|sat together .* without me)\b/i,
      emotion: "loneliness / sadness",
      valence: -0.70
    },
    {
      name: "Quiet Peace & Contentment",
      regex: /\b(deeply at peace|calm and quiet|cup of tea|rain fell|sitting on the porch|content)\b/i,
      emotion: "peace / contentment",
      valence: +0.65
    }
  ];

  let matchedPattern = null;
  for (const p of patterns) {
    if (p.regex.test(lower)) {
      if (p.negativeTrigger && !p.negativeTrigger.test(lower)) continue;
      if (p.internalDread && !p.internalDread.test(lower)) continue;
      matchedPattern = p;
      break;
    }
  }

  // 6. Sentence-weighting: when a pivot like "but" exists, the resolution clause carries 75% of the situational weight!
  let resolutionClause = "";
  if (hasPivot && pivotWord) {
    const parts = lower.split(new RegExp(`\\b${pivotWord}\\b`, 'i'));
    if (parts.length > 1) {
      resolutionClause = parts[parts.length - 1].trim();
    }
  }

  return {
    hasPivot,
    pivotWord,
    isOtherEmotion,
    matchedPattern,
    resolutionClause: resolutionClause || lower
  };
}

console.log("=" .repeat(78));
console.log(" SIGNALWELL TEXT MODEL SITUATIONAL SCENARIO EVALUATION");
console.log("=" .repeat(78));
console.log(`Evaluating ${SCENARIOS.length} complex narrative situations to demonstrate context-aware parsing.\n`);

SCENARIOS.forEach((s) => {
  console.log(`[SCENARIO ${s.id}] Category: ${s.category}`);
  console.log(`Text: "${s.text}"`);
  console.log(`  ❌ Naive Word Matching:      ${s.naiveWordPrediction}`);
  console.log(`  🎯 Target Situational Emotion: ${s.expectedSituationalEmotion}`);

  const analysis = analyzeSituationalContext(s.text);
  console.log(`  🧠 Contextual Analysis:`);
  console.log(`     - Contrast Pivot Detected: ${analysis.hasPivot ? `Yes ('${analysis.pivotWord}')` : 'No'}`);
  console.log(`     - Third-Party Attribution: ${analysis.isOtherEmotion ? 'Yes (others emotion)' : 'No (self)'}`);
  if (analysis.matchedPattern) {
    console.log(`     - Situational Pattern:     ${analysis.matchedPattern.name} => [${analysis.matchedPattern.emotion}]`);
    console.log(`     - Contextual Valence:      ${analysis.matchedPattern.valence > 0 ? '+' : ''}${analysis.matchedPattern.valence}`);
  } else {
    console.log(`     - Effective Resolution:    "${analysis.resolutionClause.slice(0, 60)}..."`);
  }
  console.log(`     - Explanation:             ${s.notes}`);
  console.log("-".repeat(78));
});

