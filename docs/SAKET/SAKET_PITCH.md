# SAKET Project Pitch

## Title

SAKET: A Controllable Indian Classical Music Generation System Built with Two Complementary AI Architectures

## 1. Short Pitch Version

SAKET is a research-driven music generation platform for Indian classical music. The project is built around two distinct but connected model families: a Hybrid CVAE and a V2 Transformer-Flow model. The first model focuses on global musical structure, latent creativity, and controlled generation. The second model pushes further into expressive performance modeling by combining token prediction, flow-matching, and grammar-aware learning.

In simple terms, SAKET is not just a music generator. It is a structured AI system that understands musical identity, rhythmic cycles, raga rules, and expressive contours. That is what makes it suitable for research, academic presentation, and future product development.

## 2. What Problem Does It Solve?

Indian classical music is difficult to generate because it is not only about notes. It requires:
- Correct raga identity
- Rhythmic alignment with taal cycles
- Controlled mood and tempo
- Musical phrase continuity
- Expressive performance features such as dynamics and pitch contour

Most generic music models are weak in at least one of these areas. They may produce pleasant sequences, but they often fail to respect domain-specific musical grammar. SAKET is designed to address that gap.

## 3. Why This Project Matters

This project is valuable for three reasons:

1. Academic value
- It combines symbolic music modeling, latent learning, and flow-based expression modeling in one research pipeline.
- It can be presented as a serious AI/music research project.

2. Technical value
- It includes two advanced model architectures.
- It uses domain-aware constraints, training objectives, and controllable generation.

3. Product value
- It can evolve into a music creation tool for students, composers, researchers, and performers.
- The same framework can later support interactive composition, practice tools, and assisted music generation.

## 4. The Two Model Families

### 4.1 Hybrid CVAE Model

The Hybrid CVAE is the earlier large-scale model family. It works by encoding a token sequence into a latent distribution, sampling a latent vector, and decoding it back into music.

What it does well:
- Learns high-level musical structure
- Supports controlled generation through mood, raga, taal, tempo, and duration
- Uses a latent space to represent musical variation
- Can optionally incorporate expression features

In simple terms, this model behaves like a structured creative engine. It learns the overall musical plan and then reconstructs or generates the sequence from that plan.

### 4.2 V2 Transformer-Flow Model

The V2 model is the newer, more expressive architecture. It uses a decoder-only Transformer backbone and adds three important learning signals:
- Token prediction for symbolic music
- Flow matching for expression dynamics
- Grammar-aware loss for raga compliance

It also includes:
- RoPE attention for better sequence handling
- RMSNorm for stable optimization
- Taal position encoding for rhythmic structure
- Expression prediction head for performance-like contour generation

In simple terms, this model is more direct and more expressive. It does not just reconstruct music; it actively learns how music should move, sound, and obey Indian classical rules.

## 5. How the Transition Between the Two Models Works

The transition from Hybrid CVAE to V2 is an important part of the project story.

### Stage 1: Hybrid CVAE

This stage established the core idea:
- Music can be conditioned on raga, mood, taal, tempo, and duration.
- A latent model can represent musical style and structure.
- Expression can be treated as an additional layer of meaning.

### Stage 2: V2 Transformer-Flow

The second stage improved the method:
- Instead of relying mainly on latent reconstruction, the model now learns multiple objectives together.
- Expression is modeled more explicitly using flow matching.
- Raga grammar is enforced more directly through the loss function.
- Taal-aware rhythmic structure is embedded into the architecture itself.

### Why this transition is important

This transition shows maturity in the research design. It demonstrates that the project did not stay at one idea. It evolved from a latent generative model into a more refined hybrid framework that is better aligned with musical control, expressiveness, and domain validity.

## 6. Technical Explanation in Simple Language

### 6.1 Inputs

The system accepts musical control signals:
- Mood
- Raga
- Taal
- Tempo
- Duration

It may also use:
- MIDI token sequences
- Audio-derived expression features such as pitch contour, amplitude, voiced/unvoiced state, and spectral centroid

### 6.2 Symbolic Layer

The symbolic layer is the discrete music brain.
It handles note-level structure, melody progression, and phrase generation.

### 6.3 Expression Layer

The expression layer is the performance brain.
It handles how the note should feel in time, including contour and dynamics.

### 6.4 Grammar Layer

The grammar layer is the domain intelligence.
It teaches the system which pitch classes are preferred, which are discouraged, and how raga rules should shape output.

### 6.5 Rhythmic Layer

The rhythmic layer gives the model an understanding of the cycle structure of Indian classical music.
This is essential because taal is not just tempo; it is a structural cycle with musical meaning.

## 7. Key Technical Terms Explained Simply

- CVAE: A model that learns a compressed latent representation and can generate from that learned space.
- Transformer: A sequence model that uses attention to understand context.
- Flow matching: A method that learns how to move noise toward data through a vector field.
- RoPE: Rotary positional embedding, used to help the model understand sequence order better.
- RMSNorm: A stable normalization method that helps training.
- Tokenization: Converting MIDI events into discrete machine-readable symbols.
- Conditioning: Giving the model control variables such as raga or tempo.
- Latent space: A hidden internal space where the model stores abstract musical patterns.

## 8. Why SAKET Is Entrepreneurial

SAKET has entrepreneurial potential because it is not only a research model. It is a platform concept.

Possible future directions include:
- AI composition assistant for Indian classical music
- Educational tool for students learning ragas and taals
- Performance analysis and generation assistant
- Creative tool for composers and researchers
- A domain-specific platform that can differentiate from generic music AI products

From a product perspective, the value is in specialization. Generic tools can create music. SAKET is designed to create culturally aligned music.

## 9. Why This Can Be Explained Well to Teachers

Teachers usually want three things:
- A clear problem statement
- A technically valid solution
- A meaningful contribution

SAKET gives all three.

You can explain it like this:
- The problem is that Indian classical music is highly structured and difficult for general AI to model.
- The solution is a two-stage research system with one latent model and one flow-based expressive model.
- The contribution is a controllable, domain-aware generative architecture that respects raga grammar and rhythmic structure.

## 10. What Makes the Project Unique

The uniqueness comes from the combination of:
- Indian classical domain knowledge
- Two complementary model families
- Structural and expressive learning in one ecosystem
- Grammar-aware and rhythm-aware generation
- A research-to-product pathway

This is not a random AI model. It is a domain-specific generative system with a clear technical identity.

## 11. Pitch Narrative You Can Speak Aloud

SAKET is my research project for controllable Indian classical music generation. The idea is to build AI that does not just generate random music, but understands raga, taal, mood, tempo, duration, and expression. I first developed a Hybrid CVAE model to learn global structure and latent creativity. Then I evolved the system into a V2 Transformer-Flow model that combines token prediction, flow matching, and grammar-aware learning. This transition made the model stronger in terms of expression, rhythmic awareness, and raga compliance. In effect, SAKET is a domain-specific music intelligence system, and it can be positioned both as a research contribution and as the base for a future product.

## 12. Teacher-Friendly One-Minute Version

SAKET is an AI music research project focused on Indian classical music generation. It uses two models: a Hybrid CVAE for learning global musical structure, and a V2 Transformer-Flow model for expressive and grammar-aware generation. The project is special because it understands raga rules, rhythmic taal cycles, and performance expression instead of just generating notes. That makes it both technically strong and culturally meaningful. It can be presented as a research contribution now, and later it can become an educational or creative AI product.

## 13. What To Emphasize in a Pitch

When presenting to teachers, emphasize:
- The problem is complex and domain-specific.
- The method is technically serious.
- The two-model transition shows research progression.
- The project has both academic and product potential.
- The system is interpretable in simple language but grounded in advanced AI methods.

## 14. Closing Statement

SAKET is a structured AI music generation system for Indian classical music that combines creativity, control, and domain intelligence. Its strength is not just that it generates music, but that it generates music with awareness of musical rules, performance expression, and rhythmic tradition. That is what makes it a strong research project and a strong pitch.
