# Dependency Edge Categories

This document describes the edge types used in the dependency graphs produced by `3_dependency_classification.py`.

Valid edge types:

- `Causal`
- `Elaboration`
- `Conditional`
- `Evaluative`
- `Questioning`
- `Contrastive`
- `Narrative/Referential`
- `Uncategorized`

---

## 1. Causal

**JSON label:** `Causal`

**Description**

Use `Causal` when one sentence gives a *cause or reason* for another sentence, or when it states that one event *leads to* another.

**Typical patterns**

- “X caused Y…”
- “Because X, Y…”
- “X explains why Y…”

**Examples**

1. **Cause of policy action**

- **Source (cause):**  
  “The rate of wind energy and solar energy has increased since 2000.”
- **Target (effect):**  
  “The governments of various countries promote renewable energy.”
- **Edge:** `Causal` (0 → 3)
- **Why this label:** The increase in wind and solar energy is presented as the reason governments promote renewable energy, so the target sentence is an effect of the source.

2. **Cause of economic slump**

- **Source (cause):**  
  “The Lehman Shock happened in 2008.”
- **Target (effect):**  
  “The Lehman Shock caused an economic slump.”
- **Edge:** `Causal` (8 → 10)
- **Why this label:** The second sentence explicitly states that the Lehman Shock “caused” the economic slump, making the relationship straightforwardly causal.

3. **Cause of decreased electricity demand**

- **Source (cause):**  
  “The Lehman Shock caused an economic slump.”  
  “The spread of coronavirus caused an economic slump.”
- **Target (effect):**  
  “An economic slump reduces the demand for electric power generation.”
- **Edge:** `Causal` (10 → 12, 11 → 12)
- **Why this label:** Both source sentences describe events that lead to an economic slump, which in turn explains the reduced demand for electricity in the target sentence.

4. **Cause of global warming**

- **Source (cause):**  
  “The use of fossil fuels has been consistently higher than the use of clean energy sources.”  
  “Everyone is impacted by the negative effects of climate change.”
- **Target (effect):**  
  “The expenditure of fossil fuels contributes to global warming.”
- **Edge:** `Causal` (2 → 20, 19 → 20)
- **Why this label:** The target sentence claims that burning fossil fuels contributes to global warming, using the high fossil fuel use and widespread climate impacts as causal background.

---

## 2. Elaboration

**JSON label:** `Elaboration`

**Description**

Use `Elaboration` when a sentence *adds detail, examples, clarification, or supporting information* to another sentence without changing its basic meaning.

**Typical patterns**

- “For example, …”
- “In more detail, …”
- “This shows that…”
- Listing specific instances of a general claim.

**Examples**

1. **Detailing fossil fuel rate**

- **Source (base idea):**  
  “However, fossil fuels occupy more than half of energy materials.”
- **Target (detail):**  
  “The rate of fossil fuels may remain steady.”
- **Edge:** `Elaboration` (1 → 15)
- **Why this label:** The target sentence adds a more specific characterization (the rate remaining steady) of the already-mentioned dominance of fossil fuels, enriching rather than changing the original claim.

2. **Detailing variance of stats**

- **Source (base idea):**  
  “During the canvassing, I noticed a fact.”
- **Target (detail):**  
  “The fact is that clean energy was a very important issue.”
- **Edge:** `Elaboration` (17 → 18)
- **Why this label:** The target spells out what the “fact” is, supplying concrete content to the vague reference in the source sentence.

3. **Detailing clean energy growth**

- **Source (base idea):**  
  “The use of clean energy sources in the world has grown during the last 20 years.”
- **Targets (detail / examples):**  
  “Wind power has experienced much growth recently.”  
  “Solar power has experienced much growth recently.”
- **Edge:** `Elaboration` (1 → 5, 1 → 6)
- **Why this label:** The wind and solar sentences serve as concrete examples that flesh out the general statement about clean energy growth.

4. **Detailing resource usage**

- **Sources (base idea):**  
  “Coal is a popular energy source.”  
  “Gas is a popular energy source.”
- **Targets (detail):**  
  “The use of coal has steadily increased.”  
  “The use of gas has steadily increased.”
- **Edge:** `Elaboration` (3 → 9, 4 → 8)
- **Why this label:** The target sentences specify how coal and gas are popular (their usage has increased), elaborating on the more general popularity statements.

5. **Strengthening a claim**

- **Source (base claim):**  
  “The expenditure of fossil fuels contributes to global warming.”
- **Target (stronger version / support):**  
  “It has been proven that the expenditure of fossil fuels greatly contributes to global warming.”
- **Edge:** `Elaboration` (20 → 21)
- **Why this label:** The target rephrases and strengthens the same claim by adding that it has been “proven” and “greatly” contributes, deepening the original assertion without introducing a new relation.

---

## 3. Conditional

**JSON label:** `Conditional`

**Description**

Use `Conditional` when a sentence explicitly states an *if–then* relationship: a condition and its potential consequence.

**Typical patterns**

- “If X, then Y…”
- “If X happens, Y will/may/could happen…”

**Examples**

1. **Fossil fuels and global warming**

- **Source (condition):**  
  “The rate of fossil fuels may remain steady.”
- **Targets (conditional outcomes):**  
  “If the rate of fossil fuels remains steady then global warming will worsen.”  
  “If the rate of fossil fuels remains steady then air pollution will worsen.”
- **Edge:** `Conditional` (15 → 16, 15 → 17)
- **Why this label:** The targets explicitly use “If … then …” structure to link the steady fossil fuel rate to potential worsening of global warming and air pollution.

2. **Global warming and local environment**

- **Source (condition):**  
  “If the rate of fossil fuels remains steady then global warming will worsen.”
- **Targets (conditional outcomes):**  
  “If global warming worsens then the natural environment in Akita could be negatively affected.”  
  “If global warming worsens then the growth of Akita cedars may deteriorate.”
- **Edge:** `Conditional` (16 → 20, 16 → 22)
- **Why this label:** Both targets state what may happen *if* global warming worsens, maintaining a clear conditional structure tied to the source sentence.

---

## 4. Evaluative

**JSON label:** `Evaluative`

**Description**

Use `Evaluative` when a sentence expresses a *value judgment, evaluation, or opinion* about another sentence or situation.

**Typical patterns**

- “This is a problem / good / bad / important / beneficial / harmful…”
- “It is surprising / concerning that…”

**Example**

1. **Evaluating a situation**

- **Source (situation described):**  
  “However, fossil fuels occupy more than half of energy materials.”
- **Target (evaluation):**  
  “This situation is a problem.”
- **Edge:** `Evaluative` (1 → 2)
- **Why this label:** The target sentence explicitly judges the described situation as “a problem,” expressing a value-laden evaluation rather than adding new factual content.

---

## 5. Questioning

**JSON label:** `Questioning`

**Description**

Use `Questioning` when a sentence *asks about, interrogates, or expresses curiosity* about another sentence or fact.

**Typical patterns**

- “I wonder why X…”  
- “I wonder how much X affects Y…”  
- “Why is X…?”  

**Examples**

1. **Questioning increases in gas/coal**

- **Sources (facts):**  
  “The use of gas has steadily increased.”  
  “The use of coal has steadily increased.”
- **Target (question):**  
  “I wonder why the use of gas and coal has steadily increased.”
- **Edge:** `Questioning` (8 → 10, 9 → 10)
- **Why this label:** The target shows the author asking why the earlier facts are true, turning the preceding statements into the object of a question.

2. **Questioning population–electricity link**

- **Sources (facts):**  
  “Electricity generation has increased over time.”  
  “The global population has grown.”
- **Target (question):**  
  “I wonder how much global population growth affects how electricity is generated.”
- **Edge:** `Questioning` (0 → 13, 12 → 13)
- **Why this label:** The target sentence explicitly wonders about the effect of population growth on electricity generation, questioning the relationship between the two earlier facts.

3. **Questioning cross-country variation**

- **Source (fact):**  
  “Statistics about electricity generation vary from country to country.”
- **Target (question):**  
  “I wonder how greatly these statistics vary from country to country.”
- **Edge:** `Questioning` (14 → 15)
- **Why this label:** The target takes the stated fact and turns it into an inquiry about *how much* variation there is, so the relationship is driven by a question about the source statement.

---

## 6. Contrastive

**JSON label:** `Contrastive`

**Description**

Use `Contrastive` when a sentence *contrasts, opposes, or highlights a difference* with another sentence.

**Typical patterns**

- “However, …”  
- “In contrast, …”  
- “On the other hand, …”  
- “X, but Y…”

**Examples**

1. **Clean vs fossil energy**

- **Source (clean energy growth):**  
  “The use of clean energy sources in the world has grown during the last 20 years.”
- **Target (contrast: fossil fuels higher):**  
  “The use of fossil fuels has been consistently higher than the use of clean energy sources.”
- **Edge:** `Contrastive` (1 → 2)
- **Why this label:** The target explicitly compares fossil fuel use to clean energy use and emphasizes that it is “higher,” putting the two trends in contrast.

2. **Contrast between new and traditional sources**

- **Source (new ways):**  
  “New ways to generate electricity have proliferated.”
- **Target (questioning increased gas/coal use, contrasting with proliferation):**  
  “I wonder why the use of gas and coal has steadily increased.”
- **Edge:** `Contrastive` (7 → 10)
- **Why this label:** The coexistence of proliferating new methods and increasing gas/coal use sets up a tension between old and new sources, so the question implicitly contrasts these trends.

---

## 7. Narrative/Referential

**JSON label:** `Narrative/Referential`

**Description**

Use `Narrative/Referential` when a sentence *refers back to a previously mentioned event, entity, or time*, or continues a personal or story-like narrative. The connection is about *reference or narrative flow*, not necessarily causal or evaluative content.

**Typical patterns**

- “At that time, …” (referring to an earlier event)  
- “The decrease… relates to these events.”  
- “During X, I noticed…”  

**Examples**

1. **Linking problem to earlier situation**

- **Source (reference situation):**  
  “This situation is a problem.”
- **Target (narrative continuation / reference to that situation):**  
  “At that time, I thought about a solution to improve this situation.”
- **Edge:** `Narrative/Referential` (2 → 5)
- **Why this label:** The target refers back to “this situation” and “that time,” continuing the story about the problem rather than introducing a new causal or evaluative relation.

2. **Linking personal experience**

- **Source (earlier life event):**  
  “When I was a junior high school student, I had an opportunity.”
- **Target (later thought, tied to that time):**  
  “At that time, I thought about a solution to improve this situation.”
- **Edge:** `Narrative/Referential` (4 → 5)
- **Why this label:** The target anchors itself in the same time frame (“that time”) as the source, advancing a personal narrative linked by reference rather than cause or contrast.

3. **Referring to earlier events and outcome**

- **Sources (events and outcome):**  
  “The electric power generation in the world decreased around 2008.”  
  “The electric power generation in the world also decreased around 2021.”  
  “The demand for electric power generation decreased.”
- **Target (reference to them):**  
  “The decrease in electric power generation relates to these events.”
- **Edges:**  
  `Narrative/Referential` (6 → 14, 7 → 14), plus `Causal` (13 → 14)
- **Why this label:** The target sentence explicitly says the decrease “relates to these events,” tying back to previously mentioned episodes and outcomes in a referential way, while also gesturing at causality from the demand decrease.

4. **Personal location / environment**

- **Source (location):**  
  “I live in Akita.”
- **Targets (narrative / referential ties):**  
  “If global warming worsens then the natural environment in Akita could be negatively affected.”  
  “Akita cedars grow in Akita.”
- **Edge:** `Narrative/Referential` (19 → 20, 19 → 21)
- **Why this label:** Both targets refer back to Akita, using the shared place to connect the speaker’s location with the potential environmental consequences there.

5. **Narrative of canvassing**

- **Source (earlier action):**  
  “I canvassed for a local democratic campaign.”
- **Target (subsequent event in same story):**  
  “During the canvassing, I noticed a fact.”
- **Edge:** `Narrative/Referential` (16 → 17)
- **Why this label:** The target situates itself “during the canvassing,” clearly continuing the same narrative episode introduced in the source.

---

## 8. Uncategorized

**JSON label:** `Uncategorized`

**Description**

Use `Uncategorized` when a dependency between sentences is present but *does not clearly fit* any of the other categories, or when you are uncertain about the best label.

**When to use**

- The relationship is vague or mixed in type.  
- The sentence is related, but not clearly causal, elaborative, conditional, evaluative, questioning, contrastive, or narrative/referential.

_No concrete examples are enforced in the current script; use this label sparingly and prefer a more specific type when possible._