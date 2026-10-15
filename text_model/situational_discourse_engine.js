/**
 * SignalWell — Universal Situational Discourse Emotion Engine
 * ==========================================================
 * Evaluates complex, multi-clause student self-reports and narratives.
 * Works natively in both Node.js (CLI / Server) and Web Browsers (Client-side).
 */

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.SituationalDiscourseEngine = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {

  const EKMAN_7 = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];

  const PIVOT_REGEX = /\b(but|however|yet|nevertheless|nonetheless|although|even though|though|finally|at last|on the other hand)\b/gi;

  const OTHER_ATTRIBUTION_REGEX = /\b(my friend|my roommate|my sister|my brother|they|she|he|everyone)\s+(was|were|started|kept)\s+(crying|cried|sobbing|screaming|furious|panicking|freaking out)\b/i;

  const SARCASM_TRIGGERS = /\b(oh fantastic|oh wonderful|just great|truly the highlight|what a joy|what a lovely day)\b/i;
  const CATASTROPHE_WORDS = /\b(blue-screened|crashed|failed|deleted|lost|broke down|destroyed|corrupted)\b/i;

  const RELIEF_TRIGGERS = /\b(weight off (my|our) (shoulders|chest)|negative and clean|finally pass|passed cleanly|turned out benign|thank god|all over now)\b/i;

  const IMPOSTOR_TRIGGERS = /\b(felt like a fraud|felt like an imposter|complete fraud|just got lucky|stomach instantly sank|don't belong here|imposter syndrome)\b/i;

  const BURNOUT_TRIGGERS = /\b(blank cursor|staring for hours|don't even feel anxious|hollow and empty|numb|burned out|exhausted to my bones)\b/i;

  const CARING_TRIGGERS = /\b(stayed with (her|him|them)|comfort|listened until|held (her|his) hand|made dinner for|took care of)\b/i;

  const LEXICON = {
    happy: ['happy', 'joy', 'excited', 'thrilled', 'won', 'proud', 'great', 'love', 'blessed', 'smile', 'accomplished', 'cannot wait', "can't wait", 'looking forward', 'excited to'],
    sad: ['sad', 'crying', 'depressed', 'sobbing', 'hollow', 'empty', 'lonely', 'heartbroken', 'hurts', 'grief'],
    angry: ['angry', 'furious', 'annoyed', 'mad', 'infuriating', 'hated', 'hate', 'pissed', 'irritated'],
    fear: ['scared', 'terrified', 'anxious', 'dread', 'panic', 'nervous', 'fraud', 'afraid', 'worry'],
    disgust: ['disgusted', 'sick', 'gross', 'nauseated', 'repulsed', 'revolting'],
    surprise: ['shocked', 'unexpected', 'suddenly', 'stunned', 'amazed', 'unbelievable', 'collapsed'],
    relief: ['relief', 'relieved', 'weight off', 'negative', 'passed', 'clean', 'solved', 'safe']
  };

  class SituationalDiscourseEngine {
    constructor(config = {}) {
      this.config = config;
    }

    /**
     * Splits text into premise and resolution based on discourse pivots
     */
    splitDiscourse(text) {
      const matches = [...text.matchAll(PIVOT_REGEX)];
      if (!matches || matches.length === 0) {
        return { hasPivot: false, pivotWord: null, premise: text, resolution: text };
      }
      const lastMatch = matches[matches.length - 1];
      const pivotWord = lastMatch[0];
      const premise = text.substring(0, lastMatch.index).trim();
      const resolution = text.substring(lastMatch.index + pivotWord.length).trim();
      return { hasPivot: true, pivotWord, premise, resolution };
    }

    /**
     * Analyzes emotional polarity and tokens in a single clause
     */
    scoreClause(clauseText) {
      const lower = clauseText.toLowerCase();
      const scores = { angry: 0.1, disgust: 0.05, fear: 0.1, happy: 0.1, neutral: 0.25, sad: 0.1, surprise: 0.1 };

      // Negation check
      const hasNegation = /\b(not|never|no longer|hardly|scarcely|without)\b/i.test(lower);

      for (const [emo, words] of Object.entries(LEXICON)) {
        for (const w of words) {
          if (lower.includes(w)) {
            if (emo === 'relief') {
              scores.happy += 0.35;
              scores.neutral += 0.15;
            } else if (hasNegation && (emo === 'happy' || emo === 'sad')) {
              // Invert valence
              if (emo === 'happy') {
                scores.sad += 0.35;
                scores.happy = Math.max(0.01, scores.happy - 0.25);
              } else if (emo === 'sad') {
                scores.neutral += 0.25;
                scores.happy += 0.15;
              }
            } else if (scores[emo] !== undefined) {
              scores[emo] += 0.35;
            }
          }
        }
      }
      return scores;
    }

    /**
     * Full situational scenario evaluation taking complete context into account
     */
    analyzeScenario(rawText) {
      if (!rawText || !rawText.trim()) {
        return {
          topEmotion: 'neutral',
          confidence: 0.5,
          probabilities: { angry: 0.05, disgust: 0.05, fear: 0.05, happy: 0.1, neutral: 0.65, sad: 0.05, surprise: 0.05 },
          valence: 0.0,
          arousal: 0.1,
          stressIndex: 10,
          discourse: { hasPivot: false, isOtherAttribution: false, archetype: 'none' }
        };
      }

      const text = rawText.trim();
      const lower = text.toLowerCase();

      // 1. Discourse parsing
      const discourse = this.splitDiscourse(text);
      let probs = {};

      if (discourse.hasPivot) {
        const premiseScores = this.scoreClause(discourse.premise);
        const resolutionScores = this.scoreClause(discourse.resolution);
        
        // 75% resolution weight, 25% premise weight
        for (const emo of EKMAN_7) {
          probs[emo] = 0.25 * (premiseScores[emo] || 0.1) + 0.75 * (resolutionScores[emo] || 0.1);
        }
      } else {
        probs = this.scoreClause(text);
      }

      // 2. Attribution & Archetype adjustments
      const isOther = OTHER_ATTRIBUTION_REGEX.test(lower);
      const isSarcasm = SARCASM_TRIGGERS.test(lower) && CATASTROPHE_WORDS.test(lower);
      const isRelief = RELIEF_TRIGGERS.test(lower);
      const isImpostor = IMPOSTOR_TRIGGERS.test(lower);
      const isBurnout = BURNOUT_TRIGGERS.test(lower);
      const isCaring = CARING_TRIGGERS.test(lower) && isOther;

      let archetype = 'standard';

      if (isSarcasm) {
        archetype = 'sarcastic_frustration';
        probs.angry += 0.55;
        probs.disgust += 0.35;
        probs.happy = 0.01;
        probs.neutral = 0.05;
      } else if (isRelief) {
        archetype = 'relief_after_crisis';
        probs.happy += 0.60;
        probs.surprise += 0.25;
        probs.sad = Math.max(0.01, probs.sad * 0.15);
        probs.fear = Math.max(0.01, probs.fear * 0.15);
      } else if (isImpostor) {
        archetype = 'impostor_syndrome';
        probs.fear += 0.55;
        probs.sad += 0.25;
        probs.happy = 0.02; // negate public applause trap
      } else if (isCaring) {
        archetype = 'empathy_caring';
        probs.happy += 0.30;
        probs.neutral += 0.30;
        probs.sad = Math.max(0.05, probs.sad * 0.20); // author is supporting, not grieving
      } else if (isBurnout) {
        archetype = 'academic_burnout_numbness';
        probs.sad += 0.45;
        probs.neutral += 0.35;
        probs.fear += 0.15;
      }

      // 3. Normalize to valid probability distribution
      let sum = 0;
      for (const emo of EKMAN_7) sum += (probs[emo] || 0.05);
      for (const emo of EKMAN_7) probs[emo] = (probs[emo] || 0.05) / sum;

      // 4. Calculate Valence [-1, 1], Arousal [0, 1], and Stress Index [0, 100]
      const valence = (probs.happy * 0.9) - (probs.sad * 0.8 + probs.angry * 0.85 + probs.fear * 0.75 + probs.disgust * 0.7);
      const arousal = (probs.angry * 0.85 + probs.fear * 0.8 + probs.surprise * 0.9 + probs.happy * 0.6) + 0.1;
      const stressIndex = Math.min(100, Math.max(0, Math.round((probs.angry * 40 + probs.fear * 40 + probs.sad * 30 + (archetype === 'academic_burnout_numbness' ? 35 : 0)) * 100) / 100));

      // Rank emotions
      const ranked = Object.entries(probs)
        .sort((a, b) => b[1] - a[1])
        .map(([emo, score]) => ({ emotion: emo, score: Math.round(score * 1000) / 10 }));

      return {
        topEmotion: ranked[0].emotion,
        confidence: ranked[0].score,
        probabilities: probs,
        ranked,
        valence: Math.round(valence * 100) / 100,
        arousal: Math.round(Math.min(1.0, arousal) * 100) / 100,
        stressIndex,
        discourse: {
          hasPivot: discourse.hasPivot,
          pivotWord: discourse.pivotWord,
          premise: discourse.premise,
          resolution: discourse.resolution,
          isOtherAttribution: isOther,
          archetype
        }
      };
    }
  }

  return SituationalDiscourseEngine;
});
