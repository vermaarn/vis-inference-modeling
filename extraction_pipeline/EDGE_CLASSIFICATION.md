# Dependency Edge Categories

This document describes the edge types used in the dependency graphs produced by `3_dependency_classification.py`.

Valid edge types:

- `Inferential`
- `Elaborative`
- `Evaluative`
- `Contrastive`
- `Uncategorized`

These four substantive categories (plus a fallback) replace the previous eight-category system. The consolidation is motivated by two observations: (1) the node-level ACE sentence classification (12 categories) already encodes *what kind* of information each statement contains, so the edge types need only capture the *rhetorical move* between statements; and (2) the analytical power comes from the `(node_type_A, edge_type, node_type_B)` triple, not from the edge type alone.

**Mapping from previous categories:**

| Old label | New label |
|---|---|
| Causal | Inferential |
| Conditional | Inferential |
| Elaboration | Elaborative |
| Narrative/Referential | Elaborative |
| Evaluative | Evaluative |
| Questioning | Evaluative |
| Contrastive | Contrastive |
| Uncategorized | Uncategorized |

---

## Differentiation at a glance

|  | Same direction as A | Against / orthogonal to A |
|---|---|---|
| **Factual/propositional** | **Elaborative** (refine, specify, continue) | **Contrastive** (counterpoint, tension, exception) |
| **Reasoning beyond A** | **Inferential** (cause, effect, prediction) | — |
| **Subjective orientation** | **Evaluative** (judgment, reaction, question) | — |

---

## 1. Inferential

**JSON label:** `Inferential`

**Description**

A provides a premise, observation, or condition; B states a consequence, explanation, prediction, or hypothetical outcome that reasons *beyond* what A alone states. Covers causal reasoning ("X caused Y"), abductive explanation ("this can be explained by"), conditional/hypothetical chains ("if X then Y"), and predictions ("X may lead to Y").

**Typical source-text cues:** "because," "so," "thus," "therefore," "this means that," "suggesting that," "can be explained by," "may be causing," "if…then," "may," "could," "depending on," "which would," "leads to," "due to."

**How it differs:**
- Unlike Elaborative: introduces *new propositional content* (a cause, effect, prediction) rather than refining A.
- Unlike Evaluative: makes a factual/explanatory claim rather than expressing a subjective stance.
- Unlike Contrastive: follows *from* A rather than introducing tension *against* it.

**Examples**

1. **Causal consequence**
   - A: "Those with legacy have a higher chance of getting in."
   - B: "This lowers my own chance of acceptance."
   - Source cue: "which **then** lowers" — B is a consequence of A.

2. **Abductive explanation**
   - A: "The percentage is more balanced for players and head coaches."
   - B: "This can be explained by the large number of countries playing soccer."
   - Source cue: "**This probably can be explained by**"

3. **Logical derivation**
   - A: "The energy usage of the United States was only about twice as less as China's."
   - B: "The average energy usage per person in the US is significantly greater."
   - Source cue: "**This means that**"

4. **Conditional chain**
   - A: "The U.S. may be unable to purchase fossil fuels from other countries."
   - B: "If the U.S. is unable to purchase fossil fuels then people may start to struggle."
   - Source cue: "**If**…**then**"

5. **Conditional with predicted outcome**
   - A: "Students with less access to resources need to be addressed."
   - B: "If we can address this issue, we can reduce the disparity."
   - Source cue: "**If we can**…**we can**"

---

## 2. Elaborative

**JSON label:** `Elaborative`

**Description**

B refines, specifies, continues, or contextualizes A without introducing a new causal, evaluative, or contrastive claim. Covers: general-to-specific, specific-to-general, encoding interpretation (visual→meaning), parallel enumeration, narrative continuation, temporal sequence, anaphoric reference, and headline framing.

**Typical source-text cues:** apposition (commas, parenthetical detail), "like [specific]," "such as," "for example," "for instance," "this" + same-topic continuation, "whether through," repeated entity names, "at that time," "during the," "these events," headline framing.

**How it differs:**
- Unlike Inferential: stays within A's informational territory rather than reasoning beyond it.
- Unlike Evaluative: adds factual detail or continuation, not a subjective stance.
- Unlike Contrastive: develops A in the same direction rather than introducing tension.

**Examples**

1. **Quantification of a qualitative claim**
   - A: "Legacy students who are also wealthy see an even more significant boost."
   - B: "This boost is around 7x than the average."
   - Source cue: "**around 7x**" quantifies the "significant boost."

2. **Visual encoding → data meaning**
   - A: "Southern states tended to be shaded orange."
   - B: "The shading orange means that abortion is or will be banned."
   - Source cue: decodes the color encoding.

3. **General → specific instance**
   - A: "The use of clean energy sources has grown."
   - B: "Wind power has experienced much growth recently."
   - Source cue: wind power is a specific instance of clean energy.

4. **Narrative continuation / anaphoric reference**
   - A: "Religious students were exempt from this type of education."
   - B: "A lot more of these students got pregnant accidentally."
   - Source cue: "**these students**" anaphorically references the group in A.

5. **Headline framing**
   - A: "Young women tend to reach out to their parents more often than young men."
   - B: "A catchy headline would be 'Young Adults are Children Too.'"
   - Source cue: headline restates A's observation.

---

## 3. Evaluative

**JSON label:** `Evaluative`

**Description**

B registers the viewer's subjective orientation toward A — a value judgment, emotional reaction, normative claim, or expressed curiosity/uncertainty. Whether the stance is assertive ("this is unfair") or interrogative ("I wonder why"), the edge captures the same structural move: content prompts a subjective response. The *kind* of response (prescriptive, reactive, or curiosity) is captured by B's node type, not the edge type.

**Typical source-text cues:** "it seems unfair," "is alarming," "I am deeply concerned," "is puzzling," "it is important," "I hope that," "I feel like," "should," "need to," "I wonder why/if/how," "I am curious," "I would like to know."

**How it differs:**
- Unlike Inferential: registers a reaction rather than building a reasoning chain.
- Unlike Elaborative: introduces subjectivity rather than developing A's factual content.
- Unlike Contrastive: responds *to* A rather than opposing it factually.

**Examples**

1. **Fairness judgment**
   - A: "Applicants with a higher income family and a legacy have a higher chance."
   - B: "It seems unfair to accept higher income applicants more."
   - Source cue: "**it seems unfair**"

2. **Emotional reaction**
   - A: "States with heavily restricted abortions are more likely to stress abstinence."
   - B: "I am deeply concerned for my female friends in those states."
   - Source cue: "**I am deeply concerned**"

3. **Normative claim**
   - A: "There is only a certain amount of time with our loved ones."
   - B: "It is important that we spend that time."
   - Source cue: "**it's important**"

4. **Curiosity / questioning**
   - A: "The younger generation is more optimistic about the future."
   - B: "I wonder why older people are less optimistic."
   - Source cue: "**I wonder why**"

5. **Questioning a connection**
   - A: "Most states have heavy restrictions on abortion."
   - B: "I wonder if there is a connection between abstinence and restrictions."
   - Source cue: "**I wonder if there is a connection**"

---

## 4. Contrastive

**JSON label:** `Contrastive`

**Description**

B introduces a counterpoint, qualification, tension, exception, or surprising juxtaposition against A. The viewer holds both in mind and highlights that they pull in different directions.

**Typical source-text cues:** "but," "however," "despite," "on the other hand," "while," "meanwhile," "as opposed to," "doesn't mean," "pales in comparison to," "even during," violated-expectation juxtaposition.

**How it differs:**
- Unlike Inferential: does not chain A to a consequence; places A and B side-by-side.
- Unlike Elaborative: pushes *against* A rather than developing it in the same direction.
- Unlike Evaluative: stays in the factual/observational register rather than expressing subjectivity.

**Examples**

1. **Explicit adversative**
   - A: "The majority of players are players of color."
   - B: "However, there are fewer coaches of color."
   - Source cue: "**However**"

2. **Tension between magnitude and usage**
   - A: "India's population is greater than China's."
   - B: "Compared to China, India's power usage is significantly less."
   - Source cue: tension between large population and low usage.

3. **Violated expectation**
   - A: "I noticed that many more states have abortion bans than I had anticipated."
   - B: "For some reason, I thought that only a few states did this."
   - Source cue: A is reality; B is the violated prior belief.

4. **Exception to a trend**
   - A: "As your parent's income rank increases, your chance of getting accepted increases."
   - B: "At the 90th rank and beyond, there is a dip in acceptance."
   - Source cue: "**dip**" contrasts with the upward trend.

5. **Qualification**
   - A: "The majority of major sports leagues are made up of players of color."
   - B: "This does not mean that we have accomplished increasing diversity."
   - Source cue: "**Though this doesn't mean**"

---

## 5. Uncategorized

**JSON label:** `Uncategorized`

**Description**

The dependency between A and B is present but does not clearly fit any of the four substantive categories. Use sparingly — prefer a more specific type when possible.
