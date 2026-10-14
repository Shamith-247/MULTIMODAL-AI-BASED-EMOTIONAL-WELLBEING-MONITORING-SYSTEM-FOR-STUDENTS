# PATENT SPECIFICATION DISCLOSURE
## System and Method for Evidential Cross-Modal Attention and Multi-Scale Conformer Acoustic Modeling for Robust Speech Emotion Recognition in Student Wellbeing Assessment

---

### INVENTORS:
* **Siva Sethamanyu**
* **Shamith** (Project Lead)
* Multimodal Student Emotional Wellbeing Monitoring Project Team

---

### 1. FIELD OF THE INVENTION
The present invention relates generally to affective computing, deep learning, and digital acoustic signal processing. More specifically, the invention relates to a multimodal neural network architecture, loss formulation, and inference system for speech emotion recognition (SER) that incorporates **Bidirectional Cross-Modal Co-Attention**, **Multi-Scale Conformer Temporal Dynamics**, and **Evidential Deep Learning (EDL) via Dirichlet Prior Distributions** to achieve high-accuracy emotion classification and calibrated epistemic-aleatoric uncertainty quantification across massive heterogeneous speech corpora (>55,000 recordings).

---

### 2. BACKGROUND AND PRIOR ART DEFICIENCIES
Speech Emotion Recognition in continuous student wellbeing monitoring faces critical engineering bottlenecks that prevent real-world clinical and educational deployment:

1. **Acoustic-Prosodic Domain Disconnect:** 
   Prior art systems either pass raw spectrograms through standard CNNs (losing semantic context) or pass speech through self-supervised transformers (e.g., WavLM, wav2vec 2.0) while neglecting explicit acoustic prosody (fundamental frequency $F_0$, jitter, shimmer, formant resonance, vocal energy). Prior fusion attempts rely on simple vector concatenation (`torch.cat`), which forces the network to learn inter-modal relationships through high-dimensional linear projections without explicit cross-attentive grounding.

2. **Temporal Multi-Scale Dilemma:**
   Emotional cues manifest at multiple distinct time scales:
   * **Micro-prosodic phonemic shifts (50 ms – 200 ms):** Vocal tract tremor, pitch glides, micro-pauses.
   * **Macro-prosodic utterance contours (1 s – 6 s):** Declarative vs. interrogative intonation, sustained emotional valence arcs.
   Standard Transformer Encoders excel at global pairwise dependencies but suffer from weak inductive bias for localized, high-frequency temporal tremors.

3. **Uncalibrated Point-Estimate Uncertainty:**
   Standard Softmax classification outputs overconfident probability distributions, even when given out-of-distribution (OOD) background noise, non-speech vocalizations, or masked emotional distress. Existing uncertainty estimates rely on heuristic post-hoc entropy or arbitrary auxiliary sigmoid heads, which fail to separate:
   * **Aleatoric Uncertainty:** Natural emotional ambiguity (e.g., nervous laughter, bitter sarcasm).
   * **Epistemic Uncertainty:** Lack of evidence due to unseen acoustic environments or unfamiliar speaker demographics.

---

### 3. SUMMARY OF THE INVENTION AND NOVEL TECHNICAL CLAIMS

The present invention solves these technical challenges through an integrated architecture designated **`ECAN-SER` (Evidential Cross-Attention Conformer Network)**.

```
       RAW AUDIO INPUT (6-Second Window @ 16 kHz)
                 │
       ┌─────────┴─────────────────────────────┐
       ▼                                       ▼
 [Stream 1: Pretrained WavLM Large]   [Stream 2: Multi-Scale Prosody Engine]
  - 1024-dim Temporal Sequence         - 37 Acoustic Descriptors (F0, Energy,
  - 6 Temporal Chunk Embeddings          ZCR, Spectral Flux, Formants, MFCC)
       │                                       │
       └──────────────┬────────────────────────┘
                      ▼
 ┌────────────────────────────────────────────────────────┐
 │ CLAIM 1: Bidirectional Cross-Modal Co-Attention (BC-CA)│
 │  - WavLM queries Prosody (Acoustic Salience Guidance)  │
 │  - Prosody queries WavLM (Phonetic Context Grounding)  │
 │  - Squeeze-and-Excitation Gated Residual Fusion        │
 └────────────────────────────┬───────────────────────────┘
                              ▼
 ┌────────────────────────────────────────────────────────┐
 │ CLAIM 2: Multi-Scale Conformer Temporal Backbone       │
 │  - Depthwise Dilated Temporal Convolution (Micro-Scale)│
 │  - Relative Multi-Head Self-Attention (Macro-Scale)    │
 │  - Dual Pre-Norm Feed-Forward Half-Step Networks       │
 └────────────────────────────┬───────────────────────────┘
                              ▼
 ┌────────────────────────────────────────────────────────┐
 │ CLAIM 3: Attentive Multi-Head Temporal Pooling         │
 │  - Learns content-dependent emotional salient weights  │
 └────────────────────────────┬───────────────────────────┘
                              ▼
 ┌────────────────────────────────────────────────────────┐
 │ CLAIM 4: Evidential Deep Learning (Dirichlet Prior)    │
 │  - Evidence Parameter Vector: e_k = Softplus(z_k) >= 0 │
 │  - Dirichlet Concentration: alpha_k = e_k + 1          │
 │  - Expected Probability: p_k = alpha_k / S             │
 │  - Decoupled Epistemic Uncertainty: u = K / S          │
 │  - Evidential Loss with KL Prior Divergence Penalty    │
 └────────────────────────────────────────────────────────┘
```

---

### 4. DETAILED DESCRIPTION OF NOVEL CLAIMS

#### CLAIM 1: Bidirectional Cross-Modal Co-Attention (BC-CA)
Let $\mathbf{H}_w \in \mathbb{R}^{T \times d}$ be the projected self-supervised speech representations from WavLM, and $\mathbf{H}_p \in \mathbb{R}^{T \times d}$ be the projected acoustic prosodic sequence across $T=6$ temporal windows.
The co-attention maps are computed as:
$$\mathbf{A}_{w \to p} = \text{Softmax}\left(\frac{(\mathbf{H}_w \mathbf{W}_Q)(\mathbf{H}_p \mathbf{W}_K)^T}{\sqrt{d_k}}\right)(\mathbf{H}_p \mathbf{W}_V)$$
$$\mathbf{A}_{p \to w} = \text{Softmax}\left(\frac{(\mathbf{H}_p \mathbf{U}_Q)(\mathbf{H}_w \mathbf{U}_K)^T}{\sqrt{d_k}}\right)(\mathbf{H}_w \mathbf{U}_V)$$

The fused representations are dynamically modulated by a channel-wise Squeeze-and-Excitation Gating mechanism:
$$\mathbf{G} = \sigma\left(\mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \cdot \text{GAP}([\mathbf{A}_{w \to p}; \mathbf{A}_{p \to w}]))\right)$$
$$\mathbf{Z} = \mathbf{G} \odot [\mathbf{A}_{w \to p}; \mathbf{A}_{p \to w}] + \mathbf{H}_w$$

#### CLAIM 2: Multi-Scale Conformer Temporal Backbone
Each Conformer block processes $\mathbf{Z}$ via a Macaron-style sandwich structure:
1. First Feed-Forward Half-Step: $\tilde{\mathbf{Z}}_1 = \mathbf{Z} + \frac{1}{2} \text{FFN}(\mathbf{Z})$
2. Multi-Head Self-Attention: $\tilde{\mathbf{Z}}_2 = \tilde{\mathbf{Z}}_1 + \text{MHSA}(\tilde{\mathbf{Z}}_1)$
3. Dilated Depthwise Convolutional Block:
   $$\tilde{\mathbf{Z}}_3 = \tilde{\mathbf{Z}}_2 + \text{Conv}(\text{GLU}(\text{DepthwiseConv}_{k=3, d=2}(\text{PointwiseConv}(\tilde{\mathbf{Z}}_2))))$$
4. Second Feed-Forward Half-Step: $\mathbf{Z}_{\text{out}} = \text{LayerNorm}(\tilde{\mathbf{Z}}_3 + \frac{1}{2} \text{FFN}(\tilde{\mathbf{Z}}_3))$

This enables simultaneous capture of sub-second vocal tremor dynamics (depthwise conv) and multi-second emotional prosodic shifts (self-attention).

#### CLAIM 3: Evidential Dirichlet Uncertainty Formulation
Rather than optimizing a standard Categorical cross-entropy distribution, the network parameterizes a **Dirichlet distribution** over class probabilities:
$$p(\mathbf{p} \mid \boldsymbol{\alpha}) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{k=1}^K p_k^{\alpha_k - 1}$$
The network output $\mathbf{z} \in \mathbb{R}^K$ produces non-negative emotional evidence:
$$e_k = \text{Softplus}(z_k) = \ln(1 + \exp(z_k))$$
$$\alpha_k = e_k + 1, \quad S = \sum_{k=1}^K \alpha_k$$

The expected probability of emotion $k$ is:
$$\hat{p}_k = \frac{\alpha_k}{S}$$

The total epistemic uncertainty $u$ satisfies:
$$u = \frac{K}{S} \in (0, 1]$$
When evidence $\sum e_k \to 0$ (e.g. ambient noise or out-of-distribution acoustic input), $S \to K \implies u \to 1.0$ (complete uncertainty). When evidence is strong, $S \gg K \implies u \to 0.0$.

#### CLAIM 4: Evidential Focal Regularized Loss ($\mathcal{L}_{\text{patent}}$)
The overall loss function minimizes the Bayes risk under the Dirichlet prior, augmented with a continuous KL divergence penalty against a uniform prior to suppress misleading evidence:
$$\mathcal{L}_{\text{evidence}} = \sum_{k=1}^K y_k \left( \psi(S) - \psi(\alpha_k) \right)$$
$$\mathcal{L}_{\text{KL}} = \text{KL}\left[\text{Dir}(\tilde{\boldsymbol{\alpha}}) \parallel \text{Dir}(\mathbf{1})\right] = \ln \left( \frac{\Gamma(\tilde{S})}{\prod_{k=1}^K \Gamma(\tilde{\alpha}_k)} \right) + \sum_{k=1}^K (\tilde{\alpha}_k - 1) \left( \psi(\tilde{\alpha}_k) - \psi(\tilde{S}) \right)$$
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{evidence}} + \lambda_t \mathcal{L}_{\text{KL}}$$
where $\tilde{\alpha}_k = y_k + (1 - y_k) \alpha_k$, $\psi(\cdot)$ is the digamma function, and $\lambda_t = \min\left(1.0, \frac{t}{10}\right)$ is an annealing scheduler.

---

### 5. INDUSTRIAL & CLINICAL APPLICATION
The disclosed invention is directly integrated into student wellbeing monitoring dashboards, educational stress-detection platforms, and tele-mental health triage, providing verifiable trust scores alongside real-time emotion classifications.

---

### 6. FORMAL PATENT CLAIMS

#### WE CLAIM:

1. **An automated speech signal processing apparatus for affective computing and clinical triage**, comprising:
   - an acoustic input interface configured to capture digital speech audio samples at a predetermined sampling frequency;
   - a multi-scale feature decomposition processor configured to:
     - extract a sequence of deep self-supervised contextual speech representations across a plurality of temporal chunks utilizing a multi-layer speech foundation model; and
     - extract a synchronized time-varying sequence of acoustic-prosodic descriptor vectors across said temporal chunks;
   - a bidirectional cross-modal co-attention engine coupled to said feature decomposition processor, configured to compute an acoustic-to-semantic attention map and a semantic-to-acoustic attention map, and fuse said attention maps via a channel-wise squeeze-and-excitation gating circuit to produce a cross-modally grounded acoustic sequence;
   - a multi-scale conformer temporal backbone comprising a plurality of stacked conformer blocks, each block comprising a depthwise dilated temporal convolution module configured with a receptive dilation factor to model phonemic micro-tremors, a relative multi-head self-attention module to model utterance intonation contours, and dual pre-norm feed-forward half-step networks;
   - an attentive temporal pooling processor configured to generate a compact affective embedding via content-dependent multi-head importance weighting across said temporal chunks;
   - an evidential Dirichlet inference head configured to project said affective embedding through a strictly non-negative continuous Softplus activation function to generate a class evidence vector $\mathbf{e} \in \mathbb{R}^K$, parameterizing a Dirichlet probability distribution having concentration vector $\boldsymbol{\alpha} = \mathbf{e} + \mathbf{1}$, and outputting an expected emotion class probability vector $\hat{\mathbf{p}} = \boldsymbol{\alpha} / S$ and a decoupled epistemic uncertainty metric $u = K / S$, where $S = \sum_{k=1}^K \alpha_k$; and
   - an automated clinical triage actuator configured to evaluate said epistemic uncertainty metric $u$ against an empirical threshold $\tau$, wherein when $u < \tau$, an automated affective assessment is verified, and when $u \ge \tau$, automated assessment is suppressed and an alert is dispatched to an electronic clinical counselor workstation.

2. **The apparatus of claim 1**, wherein said multi-scale feature decomposition processor extracts said deep self-supervised speech representations by taking an ensemble blend of intermediate paralinguistic layers and a top semantic layer of a WavLM foundation model.

3. **The apparatus of claim 1**, wherein said synchronized acoustic-prosodic descriptor vectors comprise fundamental pitch ($F_0$) statistics derived via a fast YIN algorithm, root-mean-square vocal energy dynamics, zero-crossing rates, spectral centroid, spectral bandwidth, spectral rolloff, pause ratio, and mel-frequency cepstral coefficients (MFCCs) evaluated independently for each temporal chunk.

4. **The apparatus of claim 1**, wherein said squeeze-and-excitation gating circuit computes a channel modulation vector $\mathbf{G} = \sigma(\mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \cdot \text{GAP}([\mathbf{A}_{w \to p}; \mathbf{A}_{p \to w}])))$, wherein $\mathbf{GAP}$ denotes global average pooling across the temporal dimension.

5. **The apparatus of claim 1**, wherein said depthwise dilated temporal convolution module comprises a kernel size of 3 and a temporal dilation factor of 2, providing a localized receptive field spanning 50 ms to 300 ms.

6. **The apparatus of claim 1**, wherein said evidential Dirichlet inference head is trained using a composite balanced evidential focal loss function comprising:
   - a Dirichlet Bayes risk expected log loss term;
   - a smoothed class-frequency weighted focal cross-entropy regularization term; and
   - an annealed Kullback-Leibler (KL) divergence regularizer penalizing misleading evidence with respect to a uniform Dirichlet prior distribution.

7. **A computer-implemented method for robust speech emotion recognition and calibrated uncertainty estimation**, the method comprising:
   - receiving, by a digital signal processor, a digital speech audio signal;
   - segmenting said audio signal into a plurality of uniform chronological temporal chunks;
   - extracting a contextual speech representation sequence and a synchronized acoustic-prosodic sequence across said temporal chunks;
   - applying bidirectional cross-modal co-attention between said contextual representations and said acoustic-prosodic sequence;
   - fusing the co-attended representations utilizing a squeeze-and-excitation channel gating mechanism to generate a fused temporal feature sequence;
   - propagating said fused temporal feature sequence through a conformer temporal backbone combining dilated depthwise temporal convolutions and multi-head self-attention;
   - aggregating temporal features across said chunks via content-dependent multi-head attention pooling;
   - transforming the pooled representation via a Softplus activation function to obtain non-negative emotion evidence parameters parameterizing a Dirichlet prior distribution;
   - computing an expected emotion classification distribution and an epistemic uncertainty metric; and
   - conditionally triggering an automated clinical counselor triage alert when said epistemic uncertainty metric exceeds a predefined threshold.

8. **The method of claim 7**, wherein said digital speech audio signal is further transcribed using an automated speech recognition (ASR) engine, and the resulting linguistic token representations are co-attended with said acoustic-prosodic sequence.

9. **The method of claim 7**, wherein training said conformer temporal backbone includes applying temporal chunk masking by randomly zeroing a selected chunk representation during backward optimization.

10. **A non-transitory computer-readable storage medium** storing instructions that, when executed by one or more processors, cause the one or more processors to perform operations comprising:
    - capturing raw speech audio from an input transducer;
    - decomposing said speech audio into dual-stream contextual phonetic representations and dynamic prosodic descriptors;
    - performing bidirectional cross-modal co-attention with squeeze-and-excitation residual gating;
    - modeling micro-tremor and macro-intonation dynamics through a dilated depthwise conformer;
    - generating Dirichlet evidence parameters via Softplus activation to decouple epistemic uncertainty from classification confidence; and
    - rendering, on an interactive graphical user interface, a predicted emotion badge, an epistemic uncertainty score, and an automated clinical triage state.
