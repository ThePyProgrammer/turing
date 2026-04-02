# Philosophy of Science: Taste vs. Execution in Research

**Research dimension:** The intellectual roots of the distinction between problem selection (taste/judgment) and mechanical execution in scientific research.

**Scope:** Philosophy of science foundations only. Excludes Goodhart's Law, self-evaluation, and contemporary 2024-2026 AI voices (covered by other researchers).

---

## 1. Michael Polanyi: Tacit Knowledge and Personal Knowledge

**Primary sources:**
- Polanyi, M. (1958). *Personal Knowledge: Towards a Post-Critical Philosophy*. University of Chicago Press.
- Polanyi, M. (1966). *The Tacit Dimension*. University of Chicago Press.

**Core argument:** Scientific discovery depends on a form of knowledge that cannot be articulated, formalized, or transmitted as explicit rules. Polanyi called this "tacit knowledge" -- the dimension of knowing that exceeds what can be stated propositionally.

**Key quotes:**

> "We can know more than we can tell."
> -- *The Tacit Dimension* (1966), p. 4

> "His act of knowing exercises a personal judgement in relating evidence to an external reality."
> -- *The Tacit Dimension* (1966), pp. 24-25

> "To hold such knowledge is an act deeply committed to the conviction that there is something there to be discovered. It is personal."
> -- *The Tacit Dimension* (1966), pp. 24-25

> "Connoisseurship, like skill, can be communicated only by example, not by precept."
> -- *Personal Knowledge* (1958)

> "Yet personal knowledge in science is not made but discovered, and as such it claims to establish contact with reality beyond the clues on which it relies. It commits us, passionately and far beyond our comprehension, to a vision of reality...Like love, to which it is akin, this commitment is a 'shirt of flame', blazing with passion and, also like love, consumed by devotion to a universal demand."
> -- *Personal Knowledge* (1958)

**Synthesis -- connection to taste vs. execution:** Polanyi provides the strongest philosophical foundation for the claim that research "taste" is real and irreducible. His argument is not merely that tacit knowledge is *currently* hard to formalize, but that it is *in principle* resistant to formalization. The scientist's ability to sense that a problem is important, that an anomaly is significant rather than noise, that a line of inquiry is promising -- these are acts of connoisseurship transmitted through apprenticeship, not through rules. An autonomous ML agent can execute experiments (the "telling" part), but the deeper knowing that guides problem selection -- what Polanyi calls "personal knowledge" -- is precisely the dimension that resists automation. The "informed guesses, hunches and imaginings" that drive discovery are not expressible in the propositional form that algorithms require.

**URLs:**
- [Michael Polanyi and tacit knowledge -- infed.org](https://infed.org/dir/welcome/michael-polanyi-and-tacit-knowledge/)
- [Personal Knowledge -- University of Chicago Press](https://press.uchicago.edu/ucp/books/book/chicago/P/bo19722848.html)
- [The Tacit Dimension -- University of Chicago Press](https://press.uchicago.edu/ucp/books/book/chicago/T/bo6035368.html)
- [Personal Knowledge -- EBSCO Research Starters](https://www.ebsco.com/research-starters/history/personal-knowledge-michael-polanyi)

---

## 2. Karl Popper (and Hans Reichenbach): Context of Discovery vs. Context of Justification

**Primary sources:**
- Popper, K. (1934/1959). *The Logic of Scientific Discovery* (*Logik der Forschung*). Hutchinson.
- Reichenbach, H. (1938). *Experience and Prediction*. University of Chicago Press.

**Core argument:** There is a fundamental asymmetry between how scientific hypotheses are *generated* (context of discovery) and how they are *tested* (context of justification). The context of discovery -- where hypotheses come from -- is not susceptible to logical analysis. Only the context of justification -- how hypotheses are tested against evidence -- admits of formal, rule-based treatment.

**Key quotes:**

> "The act of discovery escapes logical analysis; there are no logical rules in terms of which a 'discovery machine' could be constructed that would take over the creative function of the genius."
> -- Hans Reichenbach, *Experience and Prediction* (1938)

Popper approvingly cites Einstein:

> "There is no logical path leading to [the highly universal laws of science]. They can only be reached by intuition, based upon something like an intellectual love of the objects of experience."
> -- Albert Einstein, cited in Popper, *The Logic of Scientific Discovery*

On the method of science:

> "The only logical technique which is an integral part of scientific method is that of the deductive testing of theories which are not themselves the product of any logical operation."
> -- Popper, *The Logic of Scientific Discovery*

Popper holds that hypotheses require "a leap of the imagination" and that "there is no unique way, no single method such as induction, which functions as the route to scientific theory."

**Synthesis -- connection to taste vs. execution:** The Popper/Reichenbach distinction maps almost perfectly onto the taste-vs-execution divide in autonomous ML research. The "context of justification" -- testing hypotheses, running experiments, evaluating statistical significance -- is precisely the domain where automated agents excel. It is rule-based, deductive, and mechanical. The "context of discovery" -- choosing what to investigate, formulating the hypothesis worth testing, sensing which anomaly matters -- is the domain Popper explicitly places *outside* the reach of logic. Reichenbach's phrase is striking: there are no rules for a "discovery machine." An autonomous training loop is, in Reichenbach's terms, a justification machine. The discovery part -- what to test, why it matters, which direction to push -- remains the province of human taste. The irony of Popper's book title (*The Logic of Scientific Discovery*) is instructive: even Popper, who wrote the book on it, acknowledged that discovery itself has no logic.

**URLs:**
- [Karl Popper -- Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/popper/)
- [Hans Reichenbach -- Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/reichenbach/)
- [History of the Discovery/Justification Distinction -- MPIWG Berlin](https://www.mpiwg-berlin.mpg.de/research/projects/DeptII_AufrechtMonica_History)
- [Popper, Karl: Philosophy of Science -- IEP](https://iep.utm.edu/pop-sci/)

---

## 3. Thomas Kuhn: Normal Science, Puzzle-Solving, and Paradigm Choice

**Primary source:**
- Kuhn, T. S. (1962/1970). *The Structure of Scientific Revolutions*. University of Chicago Press.

**Core argument:** Most scientific work ("normal science") is puzzle-solving within an established paradigm -- highly constrained, rule-following activity where the framework of acceptable problems and methods is given. But the *choice* of paradigm -- deciding which framework to work within, and especially recognizing when to abandon one paradigm for another -- cannot be reduced to rules or algorithms. It requires judgment based on perceived similarity to exemplars, not on the application of explicit criteria.

**Key quotes:**

> "There are no rules for deciding the significance of a puzzle and for weighing puzzles...against one another."
> -- *The Structure of Scientific Revolutions* (1962/1970)

> "Perception of similarity cannot be reduced to rules, and a fortiori cannot be reduced to rules of rationality."
> -- *The Structure of Scientific Revolutions* (1962/1970)

> "The functioning of exemplars is intended explicitly to contrast with the operation of rules."
> -- *The Structure of Scientific Revolutions* (1962/1970)

On criteria for theory choice, Kuhn identifies five values -- accuracy, consistency, scope, simplicity, and fruitfulness -- but argues these "cannot determine scientific choice" because they are imprecise, disputable in application, and weighted differently by different scientists.

**Synthesis -- connection to taste vs. execution:** Kuhn's framework yields the sharpest version of the taste-vs-execution split. Normal science -- puzzle-solving within a paradigm -- is the most automatable part of research. The rules are given, the methods are known, the criteria for success are defined within the paradigm. This is exactly what an autonomous ML training loop does: it solves puzzles (hyperparameter optimization, architecture search, loss function tweaking) within a given paradigm. But *paradigm selection* -- choosing the framework, recognizing anomalies as significant, deciding when the current approach is exhausted -- is explicitly non-algorithmic. Kuhn's insistence that similarity perception "cannot be reduced to rules of rationality" is a direct philosophical challenge to any claim that research taste can be automated. The human researcher's role is not to solve puzzles faster but to choose which puzzles matter and to recognize when the puzzle-framework itself needs replacing.

**URLs:**
- [Thomas Kuhn -- Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/thomas-kuhn/)
- [The Structure of Scientific Revolutions -- Wikipedia](https://en.wikipedia.org/wiki/The_Structure_of_Scientific_Revolutions)
- [Normal Science -- Wikipedia](https://en.wikipedia.org/wiki/Normal_science)

---

## 4. Imre Lakatos: Research Programmes and the Hard Core

**Primary source:**
- Lakatos, I. (1978). *The Methodology of Scientific Research Programmes: Philosophical Papers Volume 1*. Cambridge University Press.

**Core argument:** Scientific research is organized into "research programmes" with a two-layer structure. The **hard core** contains the fundamental assumptions that define the programme and are shielded from falsification by methodological decision. The **protective belt** of auxiliary hypotheses absorbs empirical challenges and is freely modified. The **negative heuristic** forbids directing criticism at the hard core; the **positive heuristic** provides suggestions for developing the protective belt.

**Key quotes:**

> "The negative heuristic of the programme forbids us to direct the modus tollens at this 'hard core'. Instead, we must use our ingenuity to articulate or even invent 'auxiliary hypotheses', which form a protective belt around this core."
> -- *FMSRP* (1978), p. 48

> "The positive heuristic consists of a partially articulated set of suggestions or hints on how to change, develop the 'refutable variants'."
> -- *FMSRP* (1978), p. 50

A programme is **progressive** when it generates novel predictions that are corroborated; it is **degenerating** when it only produces post-hoc rationalizations.

> "A research programme is progressive if it is both theoretically and empirically progressive, and degenerating if it is not."
> -- *FMSRP* (1978), p. 34

**Synthesis -- connection to taste vs. execution:** Lakatos provides the most architecturally precise model for the division of labor between human and machine in research. The protective belt -- modifying auxiliary hypotheses, running experiments, adjusting parameters to accommodate new data -- is mechanical work. It is the domain of the positive heuristic, which can be partially formalized as "hints on how to change the refutable variants." An autonomous ML agent operates squarely in the protective belt: tweaking hyperparameters, trying architectural variants, adjusting training procedures. But the *hard core* -- the fundamental commitments of the research programme, the decision about what to protect and what to sacrifice -- requires human judgment. The decision that a programme is degenerating (producing only post-hoc rationalizations rather than novel predictions) is itself a judgment call that cannot be reduced to a metric. Lakatos's framework suggests that the most dangerous failure mode of autonomous research is not getting the wrong answer, but unknowingly operating within a degenerating programme -- endlessly adjusting the protective belt while the hard core is rotten.

**URLs:**
- [Imre Lakatos -- Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/lakatos/)
- [Lakatos's Methodology of Scientific Research Programmes -- Medium](https://thethinkinglane.medium.com/lakatos-methodology-of-scientific-research-programmes-b0c8297ce98a)
- [Research Program -- Wikipedia](https://en.wikipedia.org/wiki/Research_program)

---

## 5. Donald Schon: The Reflective Practitioner and Knowing-in-Action

**Primary source:**
- Schon, D. A. (1983). *The Reflective Practitioner: How Professionals Think in Action*. Basic Books.

**Core argument:** Professional expertise -- in engineering, architecture, medicine, and research -- cannot be understood as the application of technical rules to well-defined problems. Schon distinguishes **technical rationality** (applying known methods to given problems) from **reflection-in-action** (the practitioner's ability to reframe problems, improvise, and learn through a "reflective conversation with the situation"). The best professionals "know more than they can put into words" and "rely less on formulas learned in graduate school than on the kind of improvisation learned in practice."

**Key quotes:**

> "Increasingly we have become aware of the importance to professional practice of phenomena -- complexity, uncertainty, instability, uniqueness, and value conflict -- which do not fit the model of Technical Rationality."
> -- *The Reflective Practitioner* (1983)

> "When the phenomenon at hand eludes the normal categories of knowledge-in-practice, presenting itself as unique or unstable, the practitioner may surface and criticize his initial understanding of the phenomenon, construct a new description of it, and test the new description by an on-the-spot experiment."
> -- *The Reflective Practitioner* (1983)

The practitioner "shapes the situation, but in conversation with it, so that his own models and appreciations are also shaped by the situation."

Professionals judge a "problem-setting by the quality and direction of the reflective conversation to which it leads. This judgment rests, at least in part, on his perception of potentials for coherence and congruence which he can realize through his further inquiry."

**Synthesis -- connection to taste vs. execution:** Schon complements Polanyi by focusing not on the epistemology of knowledge but on the *practice* of expertise. His key contribution is the concept of **problem-setting** as distinct from **problem-solving**. Technical rationality -- applying known methods to defined problems -- is the automatable part; it maps to what an ML training loop does. But the harder, prior act is problem-setting: deciding what the problem *is*, framing it productively, recognizing when the current framing is failing. Schon's "reflection-in-action" is the researcher mid-experiment realizing that the question is wrong, that the loss function is measuring the wrong thing, that the entire experimental setup needs reframing. This is not a step that can be inserted into a pipeline; it is a continuous, situated, improvisational process. Schon's framework predicts that autonomous research systems will be competent problem-solvers but poor problem-setters -- capable of optimizing within a frame but unable to recognize when the frame itself needs changing.

**URLs:**
- [Notes on Schon's Reflective Practitioner -- Dan Saffer, Medium](https://odannyboy.medium.com/notes-on-donald-sh%C3%B6ns-the-reflective-practitioner-e67f753879d8)
- [The Reflective Practitioner -- Amazon](https://www.amazon.com/Reflective-Practitioner-Professionals-Think-Action/dp/0465068782)
- [Donald Schon: The Reflective Practitioner -- Andy Matuschak's notes](https://notes.andymatuschak.org/z5wZoGy72FafNGd1AHgtghs)

---

## 6. Hans Reichenbach: The Discovery Machine That Cannot Be Built

**Primary source:**
- Reichenbach, H. (1938). *Experience and Prediction: An Analysis of the Foundations and the Structure of Knowledge*. University of Chicago Press.

**Core argument:** Reichenbach codified the distinction between the context of discovery and the context of justification. His formulation is the most explicit statement in philosophy of science that discovery is not mechanizable.

**Key quote:**

> "The act of discovery escapes logical analysis; there are no logical rules in terms of which a 'discovery machine' could be constructed that would take over the creative function of the genius."
> -- *Experience and Prediction* (1938)

**Synthesis -- connection to taste vs. execution:** Reichenbach's quote is arguably the single most relevant statement in all of philosophy of science for the autonomous ML research question. Written in 1938, it directly names what an autonomous training loop aspires to be -- a "discovery machine" -- and declares it impossible in principle. His argument is not about computational resources but about the logical structure of discovery itself: there are no rules to follow. The autonomous ML agent is, in Reichenbach's terms, operating entirely within the context of justification. It tests hypotheses efficiently, but it does not -- cannot, on Reichenbach's account -- perform the creative act of generating the hypotheses worth testing. The human researcher's "taste" is precisely what Reichenbach calls "the creative function of the genius," which escapes logical analysis and therefore escapes automation.

**URLs:**
- [Hans Reichenbach -- Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/reichenbach/)
- [History of the Discovery/Justification Distinction -- MPIWG Berlin](https://www.mpiwg-berlin.mpg.de/research/projects/DeptII_AufrechtMonica_History)

---

## Cross-Cutting Synthesis

These six thinkers converge on a single structural claim that maps directly onto the taste-vs-execution divide in autonomous ML research:

| Thinker | "Taste" (Human) | "Execution" (Automatable) |
|---------|-----------------|---------------------------|
| **Polanyi** | Tacit knowledge, connoisseurship, personal commitment | Explicit, propositional, formalizable knowledge |
| **Reichenbach** | Context of discovery ("the creative function of the genius") | Context of justification (logical analysis) |
| **Popper** | Hypothesis generation ("a leap of the imagination") | Hypothesis testing (deductive falsification) |
| **Kuhn** | Paradigm selection, anomaly recognition | Normal science puzzle-solving within a paradigm |
| **Lakatos** | Hard core commitments, programme evaluation | Protective belt adjustment, auxiliary hypothesis modification |
| **Schon** | Problem-setting, reflection-in-action | Problem-solving via technical rationality |

The philosophical consensus is striking: every major 20th-century philosopher of science drew a line between a creative/judgmental dimension and a mechanical/rule-based dimension of research. They disagreed about almost everything else -- Popper vs. Kuhn on falsification, Lakatos vs. Kuhn on rationality, Polanyi vs. all of them on the role of personal commitment -- but they *all* agreed that the creative dimension resists formalization.

For autonomous ML research, the implication is architectural: the training loop automates the *justification* side (testing, evaluation, comparison), but the *discovery* side (what to test, why it matters, when to abandon the approach) remains the province of human judgment. The correct design is not "full autonomy" but a division of labor that mirrors the philosophical structure: humans set paradigms, select problems, and evaluate programme health; machines solve puzzles, test hypotheses, and adjust protective belts.

The deepest risk identified by this literature is not that the machine will get the wrong answer, but that it will efficiently optimize within a degenerating research programme -- solving puzzles with great precision in a paradigm that should have been abandoned. This is Kuhn's "normal science" pathology: the puzzles keep getting solved, but the paradigm is exhausted. Detecting that condition requires exactly the kind of judgment that Polanyi, Kuhn, and Schon argue cannot be formalized.

---

*Research compiled: 2026-04-01*
*Dimension: Philosophy of Science -- Taste vs. Execution*
*Sources: 6 primary thinkers, 7 primary texts, 15+ secondary sources*
