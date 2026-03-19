# LaTeX Writing Checklists

## Purpose of This Document

This is a personal reference document & was written to reduce repeatedly explaining the same points when reviewing graduate students’ papers in the future. It will be continuously updated with writing-related knowledge accumulated over time.

This is strictly a **writing checklist reflecting personal preferences**. Some items are *right/wrong* (must follow), while others are *better/worse* (recommended). The more you know, the more you notice.

Read this once before writing your paper, and once again before submitting it to your advisor. Carefully revise it yourself before submitting version 1—then you might just manage to graduate in three years (or not).

---

# Mindset When Writing a Paper

1. A paper can **never reach a perfect “100-point” state** like undergraduate exams.

   * Do not aim for acceptance directly; aim to avoid rejection.

2. A paper follows a **strict template**.

   * If you are new, do **not try to be creative** with writing or figures.
   * Most likely, you will create representations only you can understand.
   * **Solution:** Follow expressions and structures used in highly cited papers (~100 citations).

     * Example: Studying the ResNet paper line-by-line.
     * Robotics researchers are encouraged to read works from Cyrill Stachniss’s group.

3. Everyone makes mistakes when writing manuscripts.

   * Use LaTeX to reduce human error.
   * As a co-author, contribute to proofreading.

     * Use tools like Grammarly or ChatGPT.

4. The goal is to ensure others can **clearly understand new knowledge without misunderstanding**.

5. If your English proficiency is not very high, **follow your advisor’s writing style**.

   * Analyze previous papers from your lab: structure, tone, expressions.
   * If you disagree with feedback, politely justify your reasoning.

---

## Practical English Writing Tips

* Focus first on understanding **what a paper is**, not just English writing.
* Each paragraph should contain **one key message**.
* English tends to favor **noun-based expressions**.
* Verify unfamiliar words using dictionaries and tools like Ludwig.
* Use lowercase for full terms in abbreviations:

  * “simultaneous localization and mapping (SLAM)” (correct)
* Avoid vague words like *ensure* or *facilitate*; prefer specific verbs.
* Do **not use “outperform”** (can sound aggressive).

  * Use “showed lower error,” “higher success rate,” etc.
* Use “significant” only after statistical validation (e.g., t-test).

  * Otherwise, use “substantial.”
* Using “we” is acceptable (papers are argumentative writing).
* Proper nouns (e.g., Kalman filter, Fourier transform) must be capitalized.
* Use correct spacing and hyphenation rules.
* Place adverbs **before verbs** for clarity.

---

## Basic Rules

### 1. One sentence per line in LaTeX

* Treat LaTeX like coding, not Word-style writing.

### 2. Always use `\newcommand`

* Prevent human errors by defining variables consistently.
* Makes revisions easier and safer.

### 3. Use useful LaTeX packages

* `cite`: compresses and orders citations automatically

* `cleveref`: simplifies references (e.g., `\Cref{}`)

* Figures, tables, algorithms should be placed at **top or bottom**:

  * `\begin{figure}[t!]`

* Formatting customization examples:

  * “Fig.” vs “Table”
  * Avoid period after “Table”

---

## Miscellaneous

* Distinguish between **% and percentage points (%p)**
* Add space between numbers and units (e.g., `20\,m`)
* No space for percentages (e.g., `10%`)
* Use `--` for ranges (e.g., `10--30%`)
* Use “user-defined” for parameters set externally

---

# Figures & Tables

* Always include **units for numerical values**
* The meaning of numbers is more important than the numbers themselves

---

# Structure of a Manuscript

## Abstract

* Single paragraph, independent of the main text
* Define abbreviations again
* Suggested structure:

  * **WHY**
  * **PROBLEM**
  * **HOW & WHAT**
  * **RESULTS**

---

## Introduction

Structure:

1. **WHY** – Why is this research important?
2. **PROBLEM** – Define the problem and limitations of prior work

   * Avoid overclaims
3. **HOW & WHAT** – General approaches and background
4. **MAIN CONTRIBUTION** – Clearly state contributions

---

## Related Works

* Do **not omit this section**
* Keep it separate from Introduction
* Include:

  * Baseline methods
  * Recent works
* End with a summary of differences from prior work

---

## Methodology

* Avoid generic titles like “Method”
* Provide:

  * Overview first (the “forest”)
  * Details later (the “trees”)

---

## Experimental Results

* Use **diverse visualizations**:

  * Tables (for precision)
  * Graphs (for intuition)

---

## Conclusion

* Less critical for acceptance
* Follow advisor’s tense preference
* Include a safe, non-critical future work statement

---

## Acknowledgments

* Use starred section in LaTeX:

```
\section*{Acknowledgments}
```
