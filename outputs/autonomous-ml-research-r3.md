# Contemporary Perspectives on Human-AI Collaboration in Scientific Research (2024-2026)

*Research brief for the Turing autonomous ML harness. Focused on what leading practitioners say about the role of human judgment when AI can run experiments autonomously.*

---

## 1. Andrej Karpathy -- Autoresearch and the Creativity Ceiling

**Context:** In March 2026, Karpathy open-sourced "autoresearch," an autonomous experiment loop where an AI agent modifies a training script, runs experiments under a fixed compute budget (~5 min/GPU), evaluates results, and iterates. In one overnight session it ran 126 experiments, driving loss from 0.9979 to 0.9697. A later run conducted 700 experiments over two days, discovering 20 optimizations that produced an 11% speedup on a larger model.

**Key insight -- the human writes the research agenda, not the code:**

> "You're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the program.md Markdown files that provide context to the AI agents and set up your autonomous research org."
> -- Karpathy, autoresearch README (March 2026) [1]

**On what the human must still provide:**

> "Writing a good program.md requires having done the research yourself. You need to know which directions are worth trying, what 'better' means for your problem, and when incremental gains have run their course."
> -- DataCamp guide to autoresearch [3]

**The creativity ceiling:**

> "Autoresearch works only where a scalar metric can serve as a reliable proxy for quality. In domains where evaluation requires human judgment, creativity, or subjective assessment, the approach falls apart."
> -- Analysis of autoresearch limitations [3, 9]

> "Researchers working on alignment, interpretability, or user experience still operate in territory where no single number captures success. For those domains, human researchers remain necessary, not as bottlenecks but as the only evaluators available."
> -- Winbuzzer analysis (March 2026) [9]

**On scaling and collaboration:**

> "You spin up a swarm of agents, you have them collaborate to tune smaller models, you promote the most promising ideas to increasingly larger scales, and humans (optionally) contribute on the edges."
> -- Karpathy, Fortune interview (March 2026) [2]

**Assessment:** Karpathy's formulation is structurally identical to the insight this project encodes: the agent handles execution, but the *judgment behind the research agenda remains human*. The program.md is the human's contribution -- it embodies taste, direction, stopping criteria, and the definition of "better." Without it, the loop is directionless optimization.

---

## 2. Demis Hassabis -- AI as the Ultimate Tool, Not the Scientist

**Context:** Hassabis received the 2024 Nobel Prize in Chemistry for AlphaFold. As of late 2025, AlphaFold is used by over 3 million researchers in 190+ countries.

**On the irreducible human contribution:**

> "The human ingenuity comes in first -- asking the question, developing the hypothesis -- and AI systems can't do any of that. It just sort of analyses data right now."
> -- Hassabis, Nobel Prize press conference (October 2024) [4]

> "It can't figure out what the right question is to ask, or the right hypothesis, or the right conjecture. All of that's got to come from the human scientist."
> -- Hassabis, Nobel Prize interview (2024) [4]

**On what's still missing from AI:**

> "One thing that's clearly missing... was the ability for these systems to invent their own hypotheses or conjectures about science, not just prove existing ones."
> -- Hassabis, lessons compilation [6]

> "They can play a game of Go at a world champion level. But could a system invent Go?"
> -- Hassabis, on independent scientific invention [6]

**On jagged intelligence:**

> "You'd want an AGI to have pretty consistent, robust behavior across the board for all cognitive tasks."
> -- Hassabis, noting current systems exhibit "jagged intelligence" -- excelling in some areas while failing at basic tasks [6]

**On collaboration:**

> "The best AI will be created by humans and machines working together."
> -- Hassabis [6]

**Assessment:** Hassabis draws a clean division: AI analyzes, humans hypothesize. This maps directly onto the autoresearch architecture -- the agent optimizes within a space the human defined, but cannot define the space itself.

---

## 3. Yann LeCun -- Architectural Limitations of Current AI

**Context:** LeCun, Turing Award winner and (until 2026) Meta's Chief AI Scientist, has been the most vocal critic of LLM-based approaches to general intelligence. In early 2026, he left Meta to co-found AMI Labs ($1.03B raised at $3.5B valuation) to build world models.

**On what LLMs fundamentally lack:**

> "An LLM produces one token after another... that's clearly System 1 -- it's reactive, right? There's no reasoning."
> -- LeCun (2025) [7]

> "We have these language systems that can pass the bar exam, can solve equations, compute integrals, but where is our domestic robot?"
> -- LeCun, on the Moravec's Paradox problem [7]

**On why text-only training is insufficient:**

> "We're never going to get to human-level intelligence by just training on text. It's never going to happen."
> -- LeCun (2025) [7]

**On the missing capability:**

> "A 17-year-old can learn to drive a car in about 20 hours... but we still don't have self-driving cars. So that means we're missing something really, really big."
> -- LeCun [7]

**The four deficits:** LeCun identifies four capabilities current AI structurally lacks: (1) understanding of the physical world, (2) persistent memory, (3) reasoning, and (4) complex planning. These are "not bugs to be fixed with more data but fundamental architectural limitations."

**Assessment:** LeCun's critique explains *why* autoresearch works for narrow optimization but fails for open-ended research. The agent can iterate within a defined search space (System 1), but cannot plan multi-step research strategies, revise its own approach, or reason about what experiment to try when the metric landscape is unknown (System 2). This is not a scaling problem -- it is an architecture problem.

---

## 4. Yoshua Bengio -- Scientist AI vs. Agentic AI

**Context:** In June 2025, Bengio (the world's most-cited computer scientist) launched LawZero, a nonprofit with $35M+ from the Gates Foundation and Schmidt Sciences, to build "safe by design" AI. His central proposal: "Scientist AI" -- a non-agentic system that understands rather than acts.

**The core distinction:**

> "The Scientist AI is trained to understand, explain and predict, like a selfless idealized and platonic scientist."
> -- Bengio, LawZero announcement (June 2025) [10]

> "Instead of an actor trained to imitate or please people (including sociopaths), imagine an AI that is trained like a psychologist."
> -- Bengio [10]

**On AI as guardrail rather than agent:**

> Scientist AI provides "the key ingredient of a safety guardrail: is this proposed action from the AI agent likely to cause harm?"
> -- Bengio [10]

**Assessment:** Bengio's Scientist AI formulation is the most philosophically precise framing of the human-AI research boundary. An AI that *understands* the world can accelerate science as a tool. An AI that *acts* in the world introduces agency risks. The autoresearch loop sits at this boundary -- the agent acts (modifies code, runs experiments), but the human constrains the action space via program.md. Bengio would likely argue this is safe only because the action space is narrow and the consequences reversible.

---

## 5. Sakana AI's "AI Scientist" -- A Cautionary Example

**Context:** In 2024, Sakana AI released "The AI Scientist," an automated research agent that generates hypotheses, writes code, runs experiments, and produces manuscripts. It was published in Nature and generated significant controversy. An independent evaluation (Stahlberg & Goldberg, February 2025) provided a rigorous assessment.

**Failure rate:**

> "Five out of twelve proposed experiments (42%) failed due to coding errors, and those that did run often produced logically flawed or misleading results."
> -- Stahlberg & Goldberg, arXiv 2502.14297 (February 2025) [11]

**On novelty assessment:**

> The system relies on "simplistic keyword searches rather than profound synthesis, which leads to poor novelty assessments." Well-established concepts (like micro-batching for SGD) were incorrectly classified as novel.
> -- Stahlberg & Goldberg [11]

**On output quality:**

> "Each iteration adding only 8% more characters on average, suggesting limited adaptability."
> -- Stahlberg & Goldberg [11]

> Generated manuscripts had "a median of just five citations per paper -- most of which were outdated (only five out of 34 citations were from 2020 or later)."
> -- Stahlberg & Goldberg [11]

> Manuscripts contained "missing figures, repeated sections, and placeholder text such as 'Conclusions Here'" plus hallucinated numerical results.
> -- Stahlberg & Goldberg [11]

**The bottom line:**

> The AI Scientist "currently operates at the level of an unmotivated undergraduate student rushing to meet a deadline."
> -- Stahlberg & Goldberg [11]

> The system required "substantial human setup (25 hours excluding initial installation)" and "cannot independently assess its own methodological flaws, making it unsuitable for truly autonomous scientific inquiry at present."
> -- Stahlberg & Goldberg [11]

**Assessment:** The AI Scientist is autoresearch without taste. It runs the loop, but nobody wrote a good program.md. The 42% failure rate and hallucinated results demonstrate what happens when you remove human judgment from the pipeline entirely: you get volume without validity. This is the negative example that justifies the Turing harness's immutable evaluation infrastructure.

---

## 6. Self-Driving Laboratories -- The Wet Lab Frontier

**Context:** The movement toward fully automated ("self-driving") wet labs is led by Alan Aspuru-Guzik (Toronto) and others. These systems use AI to design experiments, robots to execute them, and ML to analyze results.

**On what gets automated:**

> "The mundane decisions of what the experiment to be done next, the next conditions to try, are actually optimized using artificial intelligence."
> -- Aspuru-Guzik, University of Toronto [13]

> "A robot doesn't have feelings -- at least, not yet. A human isn't willing to give up two or three hours of their life, pipetting and doing experiments unless there's a reasonably good chance of getting a positive result."
> -- Aspuru-Guzik [14]

**On the human role:**

> Scientists must "give up the driver's seat" to a computer, which allows them to "work on more important things," such as deciding *what research projects to pursue*.
> -- Aspuru-Guzik [14]

**On the limits of automation:**

> Finding a promising molecule is "only the first step toward developing a drug that can actually be used to treat human diseases; that process can take many years and cost billions of dollars."
> -- Aspuru-Guzik [14]

**The consensus from the SDL community (2025):**

> "Human oversight of SDLs is a key element of their safety and security policies... it is prudent to ensure that whenever an autonomous experiment is run, humans with knowledge of the experiment are held responsible and accountable for its safe and secure execution."
> -- Royal Society Open Science review (2025) [15]

> "SDLs can suggest optimal experiments, but science often advances through intuition and creativity, and maintaining human oversight is essential to avoid blind spots and ensure discoveries remain meaningful."
> -- Oak Ridge National Laboratory workshop report (2024) [16]

**Assessment:** The SDL community has converged on a "levels of autonomy" framework analogous to self-driving cars. Full autonomy (Level 5) remains aspirational. Current best practice is Level 3-4: the machine handles execution and local optimization, the human handles goal-setting, interpretation, and safety oversight. This is exactly the architecture of the Turing harness.

---

## 7. Additional Notable Voices

### Fei-Fei Li (Stanford HAI, World Labs)

At the AI Safety Summit in Paris (February 2025), Li stated that AI governance should be based on "science rather than science fiction," urging a scientific approach to assessing AI capabilities and limitations. Her startup World Labs (founded 2024, $230M raised) focuses on spatial intelligence -- AI that understands the 3D physical world -- implicitly acknowledging that current AI lacks this foundational capability.

### Nature Editorial Consensus (2025-2026)

A Nature feature on self-driving labs (2026) emphasized that "to achieve their full potential, human oversight is required." The emerging editorial position treats AI-for-science as a collaboration story, not a replacement story. A separate Nature article on biomedical research acceleration found that while "current AI could deliver around 2x speed increase, and future AI could facilitate acceleration of up to 25x for physical tasks and 100x for cognitive tasks," these gains "may be severely limited by irreducible biological constraints, research infrastructure, data access, and the need for human oversight." [17]

### The "AI Advisor" Model

A December 2025 paper proposed an "AI advisor" approach for self-driving labs where humans and machines "share the driver's seat," with AI handling data processing but "keeping decisions in the hands of experienced researchers accustomed to making real-time choices using limited datasets." [18]

---

## Synthesis: The Emerging Consensus (and Disagreements)

### Where there is strong consensus (6/6 voices agree):

1. **AI accelerates execution, humans provide direction.** Every source -- from Karpathy to Hassabis to the SDL community -- agrees that AI is transformatively good at running experiments within a defined search space, but cannot define the search space itself.

2. **The human contribution is taste, not labor.** Karpathy's program.md, Hassabis's "asking the question," Aspuru-Guzik's "deciding what research projects to pursue" -- all describe the same irreducible human role: judgment about what matters.

3. **Removing humans entirely produces garbage.** The AI Scientist's 42% failure rate and hallucinated results are the canonical negative example. Volume without validity is worse than useless -- it is actively misleading.

4. **Current AI lacks self-assessment capability.** No source claims AI can reliably evaluate whether its own results are meaningful. This is why immutable evaluation infrastructure (the Turing harness's core design) is not optional.

### Where there is productive disagreement:

1. **Timeline for creative AI.** Hassabis suggests 5-10 years for "true innovation and creativity" in AI. LeCun believes it requires a fundamentally new architecture (world models). Bengio frames it as a safety question rather than a capability question.

2. **Degree of human involvement.** Karpathy says humans "optionally contribute on the edges" -- suggesting the loop can run with minimal oversight. The SDL community and Bengio argue for mandatory human-in-the-loop for any high-stakes decision. The difference may be stakes: training loss is low-stakes (you can always revert); drug synthesis is not.

3. **Agentic vs. non-agentic AI for science.** Bengio explicitly argues for non-agentic "Scientist AI" that understands but does not act. Karpathy's autoresearch is maximally agentic within its sandbox. The resolution may be that agency is safe when the action space is narrow and consequences are reversible.

### The meta-pattern:

Every voice in this survey, without exception, arrives at a version of the same insight: **autonomous AI research is a force multiplier for human judgment, not a replacement for it.** The human's job shifts from doing experiments to choosing which experiments matter -- from execution to evaluation, from labor to taste. This is not a temporary limitation waiting for better models. It reflects something structural about what research *is*: the hard part was never running the experiment.

---

## Sources

1. [Karpathy autoresearch GitHub repository](https://github.com/karpathy/autoresearch) -- README and program.md documentation (March 2026)
2. [Fortune: "The Karpathy Loop: 700 experiments, 2 days"](https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/) (March 2026)
3. [DataCamp: Guide to AutoResearch](https://www.datacamp.com/tutorial/guide-to-autoresearch) (2026)
4. [Nobel Prize: Hassabis interview](https://www.nobelprize.org/prizes/chemistry/2024/hassabis/interview/) (October 2024)
5. [MIT Technology Review: What's next for AlphaFold](https://www.technologyreview.com/2025/11/24/1128322/whats-next-for-alphafold-a-conversation-with-a-google-deepmind-nobel-laureate/) (November 2025)
6. [Lessons from Demis Hassabis](https://www.antoinebuteau.com/lessons-from-demis-hassabis/) -- compilation of quotes on AI limitations
7. [Economist Writing Every Day: LeCun on LLM Limits](https://economistwritingeveryday.com/2025/07/22/meta-ai-chief-yann-lecun-notes-limits-of-large-language-models-and-path-towards-artificial-general-intelligence/) (July 2025)
8. [Royfactory: LeCun Declares LLMs Useless in Five Years](https://royfactory.net/posts/ai/202510/yann-lecun-2025-llm-doomed-jepa-world-model/) (October 2025)
9. [Winbuzzer: Karpathy -- Humans Are the Bottleneck in AI Research](https://winbuzzer.com/2026/03/23/karpathy-humans-bottleneck-ai-research-xcxwbn/) (March 2026)
10. [Yoshua Bengio: Introducing LawZero](https://yoshuabengio.org/2025/06/03/introducing-lawzero/) (June 2025)
11. [Stahlberg & Goldberg: Evaluating Sakana's AI Scientist](https://arxiv.org/abs/2502.14297) -- arXiv 2502.14297 (February 2025)
12. [IEEE Spectrum: The AI Scientist Stirs Up Controversy](https://spectrum.ieee.org/ai-for-science-2) (2024)
13. [University of Toronto: Aspuru-Guzik on self-driving labs](https://www.utoronto.ca/news/u-t-s-al-n-aspuru-guzik-self-driving-laboratories-use-robotics-ai-automate-routine-parts) (2024)
14. [UofT Magazine: Aspuru-Guzik -- The Lab of the Future](https://magazine.utoronto.ca/research-ideas/science/alan-aspuru-guzik-has-seen-the-lab-of-the-future-and-its-very-very-fast-ai-artificial-intelligence/) (2024)
15. [Royal Society Open Science: Autonomous self-driving laboratories review](https://royalsocietypublishing.org/rsos/article/12/7/250646/235354/Autonomous-self-driving-laboratories-a-review-of) (2025)
16. [Oak Ridge National Lab: Shaping the Future of Self-Driving Autonomous Laboratories Workshop](https://info.ornl.gov/sites/publications/Files/Pub227078.pdf) (2024)
17. [Nature: What are the limits to biomedical research acceleration](https://www.nature.com/articles/s41598-025-32583-w.pdf) (2025)
18. [Phys.org: AI advisor helps self-driving labs share control](https://phys.org/news/2025-12-ai-advisor-labs-creation-generation.html) (December 2025)
19. [Sakana AI: The AI Scientist -- published in Nature](https://sakana.ai/ai-scientist-nature/) (2026)
20. [VentureBeat: Karpathy's autoresearch](https://venturebeat.com/technology/andrej-karpathys-new-open-source-autoresearch-lets-you-run-hundreds-of-ai) (2026)
21. [Nature: Inside the self-driving lab revolution](https://www.nature.com/articles/d41586-026-00974-2) (2026)
