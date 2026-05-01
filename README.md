# Advanced Data Structure Based Intelligent Drug Alternative Recommendation System

A fully from-scratch implementation of four advanced data structures powering a real-time drug recommendation engine, with a premium Tkinter UI.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [File Structure](#3-file-structure)
4. [Data Structures (Implemented from Scratch)](#4-data-structures-implemented-from-scratch)
   - 4.1 B-Tree — Disease Database
   - 4.2 AVL Tree — Medicine Storage
   - 4.3 Splay Tree — MRU Cache
   - 4.4 Fibonacci Heap — Alternative Ranking
5. [Recommendation Engine](#5-recommendation-engine)
   - 5.1 Path A — Direct Alternative Finder
   - 5.2 Path B — Symptom-Based Predictor
6. [Scoring Formulae](#6-scoring-formulae)
7. [Dataset Generator](#7-dataset-generator)
8. [UI Guide](#8-ui-guide)
9. [Known Bug Fixed](#9-known-bug-fixed)
10. [Setup & Running](#10-setup--running)
11. [Dependencies](#11-dependencies)
12. [Medical Disclaimer](#12-medical-disclaimer)

---

## 1. Project Overview

This system answers two clinical queries:

| Query Type | Inputs | Output |
|---|---|---|
| **Path A** — Direct Alternative | Disease + Baseline Medicine + Patient Age | Ranked list of alternative medicines |
| **Path B** — Symptom-Based | Symptom description + Patient Age | Inferred disease + Ranked medicine list |

**Core constraint**: All four data structures (`BTree`, `AVLTree`, `SplayTree`, `FibonacciHeap`) are implemented entirely by hand — no `collections`, `heapq`, or equivalent stdlib shortcuts are used for their logic.

---

## 2. Architecture Diagram

```
User Input (UI)
     │
     ▼
RecommendationEngine
     │
     ├── Path A: find_alternatives(disease, medicine, age)
     │       │
     │       ├─① SplayTree.access(disease)       ← O(1) MRU cache hit / O(log n) miss
     │       ├─② BTree.search(disease)            ← O(log n) primary lookup
     │       ├─③ AVLTree.filter_by_age_group()    ← O(n) traversal
     │       ├─④ score_path_a(candidate, baseline)← O(|composition|) per medicine
     │       ├─⑤ FibHeap.insert_max(score, med)   ← O(1) per insert
     │       └─⑥ FibHeap.extract_max() × top_n   ← O(log n) amortised
     │
     └── Path B: predict_from_symptoms(symptoms, age)
             │
             ├─① symptoms_to_disease(text)         ← O(k) keyword voting (NLP layer)
             ├─② SplayTree + BTree lookup          ← same as above
             ├─③ AVLTree.filter_by_age_group()
             ├─④ score_path_b(candidate)
             ├─⑤ FibHeap.insert_max()
             └─⑥ FibHeap.extract_max() × top_n
```

---

## 3. File Structure

```
drug_system/
├── b_tree.py               ← B-Tree (Disease Database Index)
├── avl_tree.py             ← AVL Tree (Medicine Storage per Disease)
├── splay_tree.py           ← Splay Tree (MRU Cache)
├── fibonacci_heap.py       ← Fibonacci Heap (Priority Ranking)
├── recommendation_engine.py← Core Engine + MedicineNode + NLP Mapper
├── generate_dataset.py     ← 2200-row CSV generator (61 diseases)
├── ui_app.py               ← Tkinter premium UI
├── medicines_dataset.csv   ← Pre-generated dataset (auto-created on first run)
└── README.md               ← This file
```

---

## 4. Data Structures (Implemented from Scratch)

### 4.1 B-Tree — `b_tree.py`

**Role**: Primary index for the disease database.

| Operation | Time Complexity |
|---|---|
| `insert_disease(name, medicine)` | O(t · log_t n) |
| `search(disease_name)` | O(t · log_t n) |
| `get_all_diseases()` | O(n) |

**How it works**:

- Minimum degree `t = 3`: each internal node holds 2–5 keys.
- Keys are disease name strings (lexicographic BST order).
- Each key is paired with an `AVLTree` object storing all medicines for that disease.
- On overflow (root full), a new root is created and the old root is split — this is the only way tree height increases, guaranteeing all leaves stay at the same depth.
- `_split_child`: the median key is promoted to the parent; left and right halves form two child nodes of `t-1` keys each. O(t).
- `_insert_non_full`: splits children proactively on the way *down*, so no backtracking is needed.

```
B-Tree (t=3) example — 5 diseases:
              [Hypertension | Migraine]
             /            |            \
  [Arthritis|ADHD]  [Insomnia|GERD]  [Type2DM|Vertigo]
```

### 4.2 AVL Tree — `avl_tree.py`

**Role**: Stores and organises all medicines for one disease. One `AVLTree` lives inside every B-Tree leaf node.

| Operation | Time Complexity |
|---|---|
| `insert(medicine)` | O(log n) |
| `search(name)` | O(log n) |
| `get_all_medicines()` | O(n) |
| `filter_by_age_group(group)` | O(n) |

**How it works**:

- Keyed by `medicine.name` (string), standard BST ordering.
- Balance factor BF = height(left) − height(right). |BF| ≤ 1 at all times.
- Four rotation cases after each insert:

| Case | Condition | Fix |
|---|---|---|
| LL | BF > 1, new key < left child's key | Single right rotation |
| RR | BF < -1, new key > right child's key | Single left rotation |
| LR | BF > 1, new key > left child's key | Left-rotate left child, then right-rotate node |
| RL | BF < -1, new key < right child's key | Right-rotate right child, then left-rotate node |

### 4.3 Splay Tree — `splay_tree.py`

**Role**: MRU (Most Recently Used) cache. Diseases queried frequently rise to the root, giving near-O(1) re-access.

| Operation | Amortised Complexity |
|---|---|
| `access(disease)` | O(log n) |
| `get_mru()` | O(1) |
| `get_top_k(k)` | O(n log n) |

**How it works**:

- Top-down splay using an auxiliary `header` node with left and right sub-trees assembled in-place.
- Three zig-step types:
  - **Zig**: target is child of root → one rotation.
  - **Zig-Zig**: target and its parent are both left (or both right) → rotate grandparent first, then parent.
  - **Zig-Zag**: target is left of parent but parent is right of grandparent (or vice versa) → two opposite-direction rotations.
- After splaying, the accessed node is the new root, giving O(1) next access.
- Each `SplayNode` also carries an `access_count` for the MRU panel display.

### 4.4 Fibonacci Heap — `fibonacci_heap.py`

**Role**: Priority queue used to rank medicine candidates. The medicine with the highest recommendation score is extracted first.

| Operation | Amortised Complexity |
|---|---|
| `insert_max(score, med)` | O(1) |
| `extract_max()` | O(log n) |
| `decrease_key_max(node, score)` | O(1) |

**Max-heap simulation**: All scores are stored negated (`stored_key = −actual_score`). The min-heap's `extract_min` then yields the highest-scoring medicine.

**Key operations**:

- **`_insert`**: Creates a `FibNode`, links it into the circular doubly-linked root list. O(1).
- **`_extract_min`**: Promotes all children of the min node to the root list, then calls `_consolidate`. O(log n) amortised.
- **`_consolidate`**: Merges trees of equal degree using an auxiliary array `A[degree]`. Maximum degree ≤ log_φ(n) where φ ≈ 1.618. Guarantees the root list has at most O(log n) trees after consolidation.
- **`_decrease_key`**: Updates a node's key, then calls `_cut` (removes from parent, adds to root) and `_cascading_cut` (propagates up if parent was already marked). Marks track how many children each node has lost; a node losing its second child is cut free. This bounds the tree degree at O(log n).

---

## 5. Recommendation Engine

### `MedicineNode` Schema

```python
MedicineNode(
    name              = "Sumatriptan XR",
    disease_target    = "Migraine",
    composition       = [{"ingredient": "Sumatriptan", "mg": 100, "percentage": 100.0}],
    suitable_for      = ["Adult", "Senior"],
    price             = 349.50,
    effectiveness_score = 0.88,
    availability      = True
)
```

### 5.1 Path A — Direct Alternative Finder

```
find_alternatives(disease, baseline_name, age_group, top_n)
```

1. **Splay cache lookup** → O(1) if MRU, else **B-Tree search** → O(log n).
2. Retrieve the `AVLTree` for the disease.
3. Find baseline medicine in AVL Tree → O(log n).
4. `filter_by_age_group(age_group)` → O(n). Excludes baseline itself.
5. Compute `max_price` for normalisation.
6. Score each candidate with `score_path_a`.
7. Push all into `FibonacciHeap`.
8. Extract top-N with `extract_max`.

### 5.2 Path B — Symptom-Based Predictor

```
predict_from_symptoms(symptom_string, age_group, top_n)
```

1. **NLP Mapping** (`symptoms_to_disease`): Tokenises the input, matches against 70+ curated symptom phrases, votes for the most likely disease. Falls back to "Seasonal Flu".
2. Splay cache → B-Tree lookup.
3. `filter_by_age_group`.
4. Score each with `score_path_b`.
5. Fibonacci Heap rank + extract.

---

## 6. Scoring Formulae

### Path A — Relative Score (with baseline)

```
Score = (CompositionMatch × 0.40)
      + (Effectiveness   × 0.30)
      − (PricePenalty    × 0.20)
      + (Availability    × 0.10)
```

**CompositionMatch** compares active ingredient lists:

| Match quality | Points |
|---|---|
| Exact mg match (within 1%) | 1.0 |
| Close mg match (within 20%) | 0.5 |
| Ingredient present, wrong dose | 0.2 |
| Ingredient absent | 0.0 |

`PricePenalty = |candidate.price − baseline.price| / max_price`  
`Availability = 1.0 if in-stock else 0.0`

### Path B — Absolute Score (no baseline)

```
Score = (Effectiveness      × 0.50)
      + (Availability       × 0.30)
      − (NormalisedPrice    × 0.20)
```

`NormalisedPrice = candidate.price / max_price`

---

## 7. Dataset Generator

**File**: `generate_dataset.py`

- **61 diseases** across: Infectious, Cardiovascular, Metabolic/Endocrine, Neurological, Mental Health, GI, Musculoskeletal, Respiratory, Skin, Urological, Oncology supportive, Eye/ENT, Women's Health, Pain.
- **2,200 rows** by default; configurable via `target_rows`.
- Each brand generates 1–3 formulation variants (XR, SR, Forte, OD, Generic, etc.).
- Prices are disease-specific and clinically plausible (INR).
- `suitable_for` lists are randomly distributed across `["Child", "Adult", "Senior"]`.

### Bug Fixed: 0 mg values

**Root cause**: The original rounding logic `round(x / 5) * 5` mapped small mg values like `1.5 mg → 0 mg` because `round(0.3) * 5 = 0`.

**Fix** (`_smart_round_mg`):

```python
def _smart_round_mg(value):
    if value < 1.0:    return max(0.5, round(value, 2))   # e.g. 0.25 mg
    elif value < 20.0: return max(0.5, round(value*2)/2)  # nearest 0.5
    elif value < 100.0:return max(0.5, round(value))      # nearest 1
    else:              return max(0.5, round(value/5)*5)  # nearest 5
```

Hard floor of `0.5 mg` ensures no ingredient ever displays as `0 mg`. Verified: **0 zero-mg values** in the regenerated dataset.

---

## 8. UI Guide

The UI (`ui_app.py`) uses a dark charcoal / amber / emerald palette with zero blue.

### Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Rx  Drug Alternative Recommendation System        ✓ Ready — 61 diseases │
├─────────────────────────────────────────────────────────────────────┤
│  ⚠  Medical Disclaimer                                              │
├──────────────────────┬──────────────────────────────────────────────┤
│  QUERY PARAMETERS    │  RECOMMENDATIONS                             │
│  ┌──────────────┐    │  ◆ #1  Sumatriptan XR              Available│
│  │Path A│Path B │    │  Score  0.7241 ████████░░           ₹ 349.50│
│  └──────────────┘    │  Composition:                               │
│  Disease ▼           │    Sumatriptan  100mg  (100%)               │
│  Baseline Med ▼      │  Effectiveness: 88%  [Adult] [Senior]       │
│  Age Group ▼         ├─────────────────────────────────────────────┤
│                      │  ◇ #2  Rizact SR                   Available│
│  Results: 5 ────     │  ...                                        │
│  [▶ Run]             │                                              │
│                      │                                              │
│  SPLAY CACHE         │                                              │
│  1. Migraine (3×)    │                                              │
│  2. Hypertension (1×)│                                              │
└──────────────────────┴──────────────────────────────────────────────┘
```

### Features

- **Path A tab**: Selecting a disease auto-populates the medicine dropdown (cascaded filter).
- **Path B tab**: Free-text symptom entry. Inferred disease shown after query.
- **Score bar**: Visual progress bar coloured green (≥0.60), amber (≥0.35), or red (<0.35).
- **Demographic badges**: Colour-coded pills — lavender (Child), emerald (Adult), amber (Senior).
- **Splay Cache panel**: Shows the top-6 most queried diseases with hit counts.
- **Engine loading**: Background thread — UI stays responsive during CSV load.

---

## 9. Setup & Running

### Step 1 — Install dependencies

```bash
pip install faker
```
> Note: `faker` is only used in `generate_dataset.py` for one variant name suffix. The core data structures and engine have zero external dependencies.

### Step 2 — Generate the dataset (optional)

The dataset is auto-generated on first UI launch. To generate manually:

```bash
python generate_dataset.py
```

Outputs `medicines_dataset.csv` (~312 KB, 2200 rows, 61 diseases).

### Step 3 — Launch the UI

```bash
python ui_app.py
```

### Step 4 — Test the engine directly

```python
from recommendation_engine import RecommendationEngine

eng = RecommendationEngine(t=3)
eng.load_csv("medicines_dataset.csv")

# Path A
results = eng.find_alternatives("Migraine", "Sumatriptan", "Adult", top_n=5)
for r in results:
    print(r["medicine"].name, r["score"])

# Path B
out = eng.predict_from_symptoms("fever, headache, runny nose", "Child", top_n=5)
print("Inferred:", out["inferred_disease"])
for r in out["recommendations"]:
    print(r["medicine"].name, r["score"])
```

---

## 10. Dependencies

| Package | Purpose | Required for |
|---|---|---|
| `tkinter` | UI framework | `ui_app.py` |
| `faker` | Brand name suffixes | `generate_dataset.py` (optional) |
| Standard library only | Data structures, engine | All core files |

**Python version**: 3.8+

---

## 11. Medical Disclaimer

> **This system provides algorithmic suggestions based on general medical data and is NOT a substitute for professional medical advice. The recommendations are computed from a dataset and carry no clinical validity. Always consult a qualified physician or pharmacist before changing any prescribed medication.**
