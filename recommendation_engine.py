"""
recommendation_engine.py
========================
Core engine: MedicineNode, NLP symptom mapper, scoring, and
RecommendationEngine class integrating all four data structures.

NLP Architecture (7 layers, all voting, highest total wins):
  L0 — Token normalisation  : lowercase, strip punctuation, contract whitespace
  L1 — Spell / variant fix  : 200+ common misspellings & double-letter variants
  L2 — Synonym expansion    : 300+ lay→medical term translations
  L3 — Suffix stripping     : -itis,-algia,-uria,-opathy,-osis patterns
  L4 — Jargon translation   : 120+ Latin/Greek medical terms → plain English
  L5 — Phrase dictionary    : 500+ scored phrases (longer = higher weight)
  L6 — Regex sentence pats  : 120+ patterns for full-sentence natural language
  L7 — Token overlap        : fuzzy match against all 61 disease name tokens
"""

import csv
import os
import re as _re

from b_tree import BTree
from avl_tree import AVLTree
from splay_tree import SplayTree
from fibonacci_heap import FibonacciHeap

# Populated by RecommendationEngine.load_csv() for Layer-7 overlap matching
_ENGINE_DISEASE_SET: set = set()


# ══════════════════════════════════════════════════════════════════════
# MEDICINE NODE
# ══════════════════════════════════════════════════════════════════════

class MedicineNode:
    """Rich representation of a single medicine/drug."""

    def __init__(self, name, disease_target, composition,
                 suitable_for, price, effectiveness_score, availability):
        self.name = name
        self.disease_target = disease_target
        self.composition = composition          # list of {"ingredient","mg","percentage"}
        self.suitable_for = suitable_for        # list of "Child"/"Adult"/"Senior"
        self.price = float(price)
        self.effectiveness_score = float(effectiveness_score)
        self.availability = bool(availability)

    def __repr__(self):
        return (f"<Medicine: {self.name} | {self.disease_target} | "
                f"₹{self.price:.2f} | Eff:{self.effectiveness_score:.2f}>")


# ══════════════════════════════════════════════════════════════════════
# NLP SYMPTOM → DISEASE MAPPER
# ══════════════════════════════════════════════════════════════════════

# ── L1: Spell / variant corrections ───────────────────────────────────
# Keys: misspelled / variant forms.  Values: corrected form.
# Covers double-letter typos, common misspellings, British/American variants.
_SPELL = {
    # Vomiting variants (the reported bug)
    "vomitting": "vomiting", "vommiting": "vomiting", "vomiting": "vomiting",
    "vomit": "vomiting", "vomits": "vomiting", "vomitted": "vomiting",
    "threw up": "vomiting", "throwing up": "vomiting", "thrown up": "vomiting",
    "been sick": "vomiting nausea", "feel sick": "nausea",
    "feeling sick": "nausea", "sick to stomach": "nausea",
    # FIX: "throwing up and loose motions" was double-replacing "vomiting" + "ing"
    # The spell table must not match partial words in longer replacements.
    # This is handled by longer-first matching, but we add explicit combo entries:
    "throwing up and loose motions": "vomiting diarrhea gastroenteritis",
    "vomiting and diarrhea": "vomiting diarrhea gastroenteritis",
    "vomiting and loose motions": "vomiting diarrhea gastroenteritis",
    "vomitting and loose motions": "vomiting diarrhea gastroenteritis",

    # Diarrhoea
    "diarrhea": "diarrhea", "diarrhoea": "diarrhea", "diarea": "diarrhea",
    "diarrea": "diarrhea", "diarhea": "diarrhea", "diarrhoea": "diarrhea",
    "loose motion": "diarrhea", "loose motions": "diarrhea",
    "loose stool": "diarrhea", "loose stools": "diarrhea",
    "watery stool": "diarrhea", "runny stool": "diarrhea",

    # Headache / head
    "headche": "headache", "haedache": "headache", "heache": "headache",
    "head hurts": "headache", "head is hurting": "headache",
    "head is killing me": "headache", "head pain": "headache",
    "head ache": "headache", "my head hurts": "headache",

    # Stomach / tummy / belly
    "stomache": "stomach", "stomack": "stomach",
    "tummy ache": "stomach pain", "tummy pain": "stomach pain",
    "tummy hurts": "stomach pain", "my tummy hurts": "stomach pain",
    "belly pain": "stomach pain", "belly ache": "stomach pain",
    "belly hurts": "stomach pain", "gut pain": "stomach pain",
    "stomach ache": "stomach pain", "abdo pain": "abdominal pain",

    # Breathing
    "breathless": "breathlessness", "breathlesness": "breathlessness",
    "cant breathe": "difficulty breathing", "can't breathe": "difficulty breathing",
    "trouble breathing": "difficulty breathing",
    "hard to breathe": "difficulty breathing", "hard to breath": "difficulty breathing",
    "out of breath": "breathlessness", "short of breath": "shortness of breath",
    "sob": "shortness of breath", "breathe": "breathing",
    "breathig": "breathing",

    # Diabetes
    "diebetes": "diabetes", "diabetis": "diabetes", "diabeties": "diabetes",
    "diabetic": "diabetes", "dibetes": "diabetes",

    # Arthritis / joints
    "artritis": "arthritis", "athritis": "arthritis", "artheritis": "arthritis",
    "achy joints": "joint pain", "achy joint": "joint pain",
    "stiff knee": "joint pain", "stiff knees": "joint pain",
    "joint aches": "joint pain", "joints ache": "joint pain",
    "joints hurt": "joint pain", "my joints hurt": "joint pain",

    # Anxiety
    "anexity": "anxiety", "anxeity": "anxiety", "anixety": "anxiety",
    "anxious": "anxiety", "anxiousness": "anxiety",

    # Migraine
    "migrene": "migraine", "migraene": "migraine", "migrane": "migraine",
    "migraines": "migraine",

    # Psoriasis
    "psoriassis": "psoriasis", "psorisis": "psoriasis", "psoriases": "psoriasis",
    "skin flaking": "skin flaking psoriasis", "skin peeling": "skin peeling psoriasis",
    "skin peel": "psoriasis", "peeling skin": "psoriasis",
    "red patches skin": "psoriasis", "red patches on skin": "psoriasis",
    "skin patches": "psoriasis",

    # Insomnia
    "insomnea": "insomnia", "insomina": "insomnia", "insomania": "insomnia",
    "cant sleep": "insomnia", "can't sleep": "insomnia",
    "trouble sleeping": "insomnia", "trouble to sleep": "insomnia",
    "hard to sleep": "insomnia", "sleepless": "insomnia",

    # Depression
    "depresion": "depression", "deppression": "depression", "depresson": "depression",
    "depressed": "depression", "feel depressed": "depression",
    "feeling depressed": "depression", "crying a lot": "depression sadness",
    "emotional": "depression mood", "mood changes": "mood swings",

    # Other diseases
    "epilepsey": "epilepsy", "epilepcy": "epilepsy",
    "ashma": "asthma", "asthema": "asthma", "ashtma": "asthma",
    "alzhiemer": "alzheimer", "alzheimers": "alzheimer",
    "parkinsons": "parkinson", "parkinsion": "parkinson",
    "schitzophrenia": "schizophrenia",
    "fibromialgia": "fibromyalgia", "fibromyaglia": "fibromyalgia",
    "conjuntivitis": "conjunctivitis", "conjuctivitis": "conjunctivitis",
    "hypertenshion": "hypertension", "hypertenion": "hypertension",
    "cholestrol": "cholesterol", "colesterol": "cholesterol",
    "osteoporsis": "osteoporosis", "osteopourosis": "osteoporosis",
    "rheumatiod": "rheumatoid arthritis", "rhuematoid": "rheumatoid arthritis",

    # Symptom spelling variants
    "sore eyes": "eye pain", "burning eyes": "eye burning",
    "watery eyes": "watery eye", "itchy eyes": "eye itching",
    "blurry vision": "blurred vision", "blurry": "blurred",
    "cant see clearly": "blurred vision", "cant see": "vision loss",
    "cloudy vision": "blurred vision",
    "hair falling out": "hair loss", "losing hair": "hair loss",
    "bald patches": "hair loss alopecia", "balding": "hair loss",
    "always hungry": "excessive hunger polydipsia",
    "excessive hunger": "polyphagia diabetes",
    "thirsty all time": "excessive thirst polydipsia",
    "thirsty all the time": "excessive thirst polydipsia",
    "sweating at night": "night sweats", "waking up sweating": "night sweats",
    "sweating profusely": "excessive sweating hyperthyroidism",
    "excessive sweating": "hyperhidrosis sweating",
    "night sweat": "night sweats",
    "poor concentration": "difficulty concentrating adhd",
    "cant concentrate": "difficulty concentrating adhd",
    "can't concentrate": "difficulty concentrating adhd",
    "cant focus": "difficulty focusing adhd",
    "brain fog": "difficulty concentrating cognitive decline",
    "loss of appetite": "appetite loss depression anorexia",
    "no appetite": "appetite loss depression anorexia",
    "not hungry": "appetite loss depression",
    "lost appetite": "appetite loss depression anorexia",
    "cant eat": "appetite loss depression anorexia",
    "no interest in food": "appetite loss depression",
    "cant stop coughing": "persistent cough bronchitis",
    "keep coughing": "persistent cough bronchitis",
    "coughing constantly": "persistent cough bronchitis",
    "coughing non stop": "persistent cough bronchitis",
    "non stop cough": "persistent cough bronchitis",
    "constant cough": "persistent cough bronchitis",
    "my skin is peeling badly": "skin peeling psoriasis scaling skin",
    "skin is peeling": "skin peeling psoriasis",
    "skin peeling": "skin peeling psoriasis scaling skin",
    "skin peels": "skin peeling psoriasis",
    "peeling skin badly": "skin peeling psoriasis scaling",
    "itchy flaky scalp": "scalp itching flaking psoriasis",
    "flaky scalp": "scalp flaking psoriasis",
    "scalp itchy": "scalp itching psoriasis",
    "scalp flaky": "scalp flaking psoriasis",
    "dandruff itchy": "scalp itching psoriasis",
    "losing weight quickly": "weight loss rapid hyperthyroidism",
    "weight loss sudden": "weight loss rapid hyperthyroidism",
    "yellow urine": "dark urine urinary",
    "dark urine": "dark urine urinary kidney",
    "smelly urine": "urinary infection urine",
    "cloudy urine": "cloudy urine urinary tract infection",
    "blood in pee": "hematuria blood urine urinary",
    "pain when peeing": "painful urination dysuria urinary",
    "burning pee": "burning urination urinary",
    "burning when peeing": "burning urination urinary",
    "pain while peeing": "painful urination urinary",
    "hearing loss": "hearing loss ear otitis",
    "ringing ears": "tinnitus ear",
    "ringing in ears": "tinnitus ear",
    "ear blocked": "ear pain congestion otitis",
    "blocked ear": "ear congestion otitis",
    "eye crust": "eye discharge conjunctivitis",
    "eye gunk": "eye discharge conjunctivitis",
    "eyes stuck together": "eye discharge conjunctivitis",
    "crusted eyes": "eye discharge conjunctivitis",
    "swollen face": "facial swelling heart failure edema",
    "puffy face": "facial swelling edema heart failure",
    "face swelling": "facial swelling edema heart failure",
    "face swollen": "facial swelling edema heart failure",
    "puffiness face": "facial swelling edema",
    "throat swollen": "swollen throat pharyngitis cold",
    "swollen throat": "swollen throat pharyngitis cold",
    "gassy": "gas bloating irritable bowel",
    "bloated stomach": "bloating irritable bowel",
    "stomach bloating": "bloating irritable bowel",
    "constipated": "constipation irritable bowel",
    "hard stool": "constipation irritable bowel",
    "fluttery heart": "palpitations arrhythmia",
    "heart skips": "palpitations arrhythmia",
    "missed heartbeat": "palpitations arrhythmia",
    "skin burning": "burning skin eczema",
    "burning sensation skin": "skin burning eczema",
    "burning sensation": "burning pain neuropathic",
    "electric feeling": "electric shock pain neuropathic",
    "muscle cramps": "muscle cramp pain",
    "leg cramps": "leg cramp lower back pain",
    "calf cramps": "leg cramp",
    "numbness": "numb numbness neuropathic",
    "tingling sensation": "tingling neuropathic pins needles",
    "bloated": "bloating irritable bowel",
    "runny nose": "runny nose cold flu",
    "blocked nose": "nasal congestion cold",
    "stuffy nose": "nasal congestion cold",
    "cant smell": "nasal congestion cold",
    "tight chest": "chest tightness asthma",
    "racing heart": "heart racing palpitations arrhythmia",
    "cant breathe properly": "difficulty breathing",
    "mouth ulcers": "mouth ulcer peptic ulcer",
    "canker sores": "mouth ulcer peptic ulcer",
    "tongue sores": "mouth ulcer",
    "breathless climbing stairs": "breathlessness exertion copd",
}

# ── L2: Synonym expansion ──────────────────────────────────────────────
# Single-word lay terms → medical/canonical equivalents
_SYN = {
    # Body parts
    "eye": "eye", "eyes": "eye", "ocular": "eye", "optic": "eye",
    "ear": "ear", "ears": "ear", "auricular": "ear",
    "nose": "nose", "nasal": "nose",
    "throat": "throat", "pharyngeal": "throat", "laryngeal": "throat",
    "stomach": "stomach", "gastric": "stomach", "tummy": "stomach", "belly": "stomach",
    "gut": "stomach", "abdomen": "stomach", "abdominal": "stomach",
    "bowel": "bowel", "intestinal": "bowel", "colonic": "bowel", "rectal": "bowel",
    "liver": "liver", "hepatic": "liver",
    "kidney": "kidney", "renal": "kidney", "nephric": "kidney",
    "bladder": "bladder", "urinary": "urinary", "urethra": "urethra", "urethral": "urethra",
    "lung": "lung", "pulmonary": "lung", "bronchial": "lung",
    "heart": "heart", "cardiac": "heart", "myocardial": "heart",
    "chest": "chest", "thoracic": "chest",
    "skin": "skin", "dermal": "skin", "cutaneous": "skin",
    "joint": "joint", "articular": "joint",
    "bone": "bone", "osseous": "bone",
    "muscle": "muscle", "muscular": "muscle",
    "nerve": "nerve", "neural": "nerve",
    "brain": "brain", "cerebral": "brain", "cranial": "brain",
    "head": "head",
    "scalp": "scalp",
    "back": "back", "lumbar": "back", "spinal": "back",
    "toe": "toe", "toes": "toe",
    "prostate": "prostate", "prostatic": "prostate",
    "thyroid": "thyroid", "thyroidal": "thyroid",
    "pelvic": "pelvic", "pelvis": "pelvic",
    "uterine": "pelvic", "ovarian": "pelvic", "vaginal": "pelvic",
    "period": "period", "menstrual": "period", "menstruation": "period",
    "face": "face", "facial": "face",

    # Symptom synonyms
    "irritation": "inflammation", "irritated": "inflammation",
    "inflamed": "inflammation", "inflaming": "inflammation",
    "swollen": "swelling", "swells": "swelling", "swelled": "swelling",
    "painful": "pain", "aching": "pain", "aches": "pain", "ache": "pain",
    "sore": "pain", "sores": "ulcer",
    "hurts": "pain", "hurting": "pain", "hurt": "pain",
    "burning": "burning", "burns": "burning", "burnt": "burning",
    "itchy": "itching", "itches": "itching", "itch": "itching",
    "bleeding": "blood", "bleed": "blood", "bleeds": "blood",
    "infected": "infection", "infects": "infection",
    "discharge": "discharge", "discharging": "discharge",
    "red": "redness", "redness": "redness",
    "yellow": "yellow", "yellowing": "yellow", "yellowish": "yellow",
    "puffy": "swelling", "puffiness": "swelling",
    "tender": "pain", "tenderness": "pain",
    "cramp": "cramping", "cramping": "abdominal pain",
    "cramps": "cramping",
    "twitch": "tremor", "twitching": "tremor",
    "shaking": "tremor", "shakes": "tremor", "shake": "tremor",
    "trembling": "tremor", "trembles": "tremor", "tremble": "tremor",
    "shivering": "chills", "shiver": "chills", "shivers": "chills",
    "stiff": "stiffness", "stiffness": "stiffness",
    "numb": "numbness", "numbing": "numbness",
    "tingly": "tingling", "tingle": "tingling",
    "cloudy": "cloudy",
    "frequent": "frequent",
    "urgency": "urgency",
    "pee": "urination", "peing": "urination", "peeing": "urination",
    "wee": "urination", "weeing": "urination",
    "urinate": "urination", "urinating": "urination",
    "ulcer": "ulcer", "ulcers": "ulcer", "ulceration": "ulcer",
    "lesion": "ulcer", "lesions": "ulcer",
    "blister": "blister", "blisters": "blister",
    "flaking": "flaking", "flaky": "flaking",
    "scaling": "scaling", "scaly": "scaling", "scales": "scaling",
    "plaque": "plaque", "plaques": "plaque",
    "forgetful": "memory loss", "forgetting": "memory loss",
    "confused": "confusion", "confusing": "confusion",
    "disoriented": "confusion",
    "hallucinating": "hallucination",
    "voices": "hearing voices",
    "delusions": "delusion", "delusional": "delusion",
    "paranoid": "paranoia",
    "impulsive": "impulsivity",
    "restless": "restlessness",
    "worrying": "worry", "worried": "worry",
    "hopeless": "hopelessness",
    "suicidal": "suicidal thoughts",
    "snore": "snoring",
    "wheeze": "wheezing", "wheezes": "wheezing",
    "breathless": "breathlessness",
    "breath": "breathing",
    "breathe": "breathing",
    "palpitation": "palpitations",
    "flutter": "palpitations",
    "racing": "palpitations",
    "cholesterol": "cholesterol",
    "lipid": "cholesterol",
    "triglyceride": "triglycerides",
    "gassy": "gas bloating",
    "bloated": "bloating",
    "constipated": "constipation",
    "diarrhea": "diarrhea",
    "vomit": "vomiting",
    "vomiting": "vomiting",
    "nauseous": "nausea",
    "sick": "nausea",
    "regurgitate": "regurgitation",
    "burp": "belching",
    "belch": "belching",
    "acidity": "acid reflux",
    "heartburn": "heartburn",
    "reflux": "acid reflux",
    "dizzy": "dizziness",
    "giddy": "dizziness",
    "spinning": "dizziness",
    "shaky": "tremor",
    "slow": "slow movement",
    "memory": "memory",
    "forget": "memory loss",
    "appetite": "appetite",
    "hungry": "hunger",
    "thirsty": "thirst",
    "tired": "fatigue",
    "exhausted": "fatigue",
    "lethargic": "fatigue",
    "drowsy": "fatigue",
    "sweating": "sweating",
    "sweaty": "sweating",
}

# ── L3: Suffix rules ───────────────────────────────────────────────────
_SUFFIX_RULES = [
    # -itis → <root> inflammation  (urethritis → urethra inflammation)
    (_re.compile(r'\b(\w+?)itis\b', _re.I),      r'\1 inflammation'),
    # -algia → <root> pain         (arthralgia → arthr pain → joint pain)
    (_re.compile(r'\b(\w+?)algia\b', _re.I),     r'\1 pain'),
    # -opathy → <root> disease
    (_re.compile(r'\b(\w+?)opathy\b', _re.I),    r'\1 disease'),
    # -osis → <root> condition
    (_re.compile(r'\b(\w+?)osis\b', _re.I),      r'\1 condition'),
    # -emia / -aemia → blood condition
    (_re.compile(r'\b(\w+?)a?emia\b', _re.I),    r'\1 blood'),
    # -uria → urine <root>
    (_re.compile(r'\b(\w+?)uria\b', _re.I),      r'urine \1'),
    # -rrhea → <root> discharge/flow
    (_re.compile(r'\b(\w+?)rr?h[eo]a\b', _re.I), r'\1 discharge'),
]

# ── L4: Medical jargon → plain English ────────────────────────────────
_JARGON = {
    # Vomiting / GI
    "emesis": "vomiting", "nausea": "nausea", "emetic": "vomiting",
    "dyspepsia": "indigestion stomach pain", "pyrosis": "heartburn acid reflux",
    "eructation": "belching", "flatulence": "gas bloating",
    "melena": "blood stool", "hematemesis": "vomiting blood stomach",
    "dysphagia": "difficulty swallowing throat gerd acid reflux",
    "odynophagia": "painful swallowing throat gerd",
    "hematochezia": "blood stool bowel",
    "tenesmus": "bowel urgency",
    "gastroesophageal": "acid reflux gerd",
    "constipation": "constipation",
    "diarrhea": "diarrhea",

    # Urinary
    "hematuria": "blood in urine urinary",
    "haematuria": "blood in urine urinary",
    "dysuria": "painful urination burning urinary",
    "pyuria": "pus in urine urinary",
    "nocturia": "frequent night urination prostate",
    "oliguria": "low urine kidney",
    "anuria": "no urine kidney failure",
    "polyuria": "frequent urination diabetes",
    "cystitis": "bladder infection urinary",
    "urethritis": "urethra inflammation urinary",
    "pyelonephritis": "kidney infection",
    "nephritis": "kidney inflammation",
    "nephropathy": "kidney disease",

    # Respiratory
    "dyspnea": "difficulty breathing breathlessness copd exertion",
    "dyspnoea": "difficulty breathing breathlessness copd exertion",
    "orthopnea": "breathlessness lying down heart failure",
    "tachypnea": "fast breathing respiratory",
    "hemoptysis": "coughing blood tuberculosis",
    "rhinorrhea": "runny nose cold",
    "rhinitis": "nose inflammation allergy cold",
    "sinusitis": "sinus inflammation nasal congestion",
    "pharyngitis": "throat inflammation sore throat cold",
    "laryngitis": "voice inflammation throat",
    "bronchiectasis": "chronic cough bronchitis",

    # Cardiovascular
    "tachycardia": "heart racing fast palpitations arrhythmia",
    "bradycardia": "slow heartbeat",
    "palpitations": "palpitations arrhythmia",
    "arrhythmia": "irregular heartbeat arrhythmia",
    "fibrillation": "irregular heartbeat arrhythmia",
    "hypertension": "high blood pressure hypertension",
    "hypotension": "low blood pressure",
    "angina": "chest pain angina",
    "edema": "swelling fluid retention",
    "oedema": "swelling fluid retention",
    "ascites": "abdominal fluid liver",

    # Neurological
    "vertigo": "dizziness spinning vertigo",
    "syncope": "fainting heart",
    "ataxia": "loss of balance coordination vertigo",
    "aphasia": "speech problem brain",
    "tinnitus": "ringing in ear otitis",
    "photophobia": "light sensitivity migraine",
    "phonophobia": "sound sensitivity migraine",
    "aura": "visual disturbance migraine",
    "neuropathy": "nerve pain numbness neuropathic",
    "neuralgia": "nerve pain neuropathic",
    "paresthesia": "tingling numbness neuropathic",
    "seizure": "seizure epilepsy",
    "convulsion": "seizure epilepsy",

    # Skin
    "pruritus": "itching skin eczema",
    "erythema": "skin redness inflammation",
    "urticaria": "hives rash allergy",
    "onychomycosis": "nail fungus fungal",
    "tinea": "ringworm fungal",
    "candidiasis": "thrush fungal",
    "seborrhea": "oily skin scalp psoriasis",
    "psoriasiform": "scaling psoriasis",
    "dermatitis": "skin inflammation eczema",

    # Eyes / ENT
    "conjunctivitis": "eye redness infection conjunctivitis",
    "keratitis": "cornea inflammation eye pain",
    "glaucoma": "eye pressure vision loss glaucoma",
    "epistaxis": "nosebleed nose hypertension",
    "otitis": "ear inflammation infection",
    "tinnitus": "ringing ear",

    # Musculoskeletal
    "myalgia": "muscle pain",
    "arthralgia": "joint pain arthritis",
    "osteoporosis": "bone loss osteoporosis",
    "gout": "uric acid joint pain toe gout",
    "sciatica": "leg nerve back pain",

    # Metabolic / endocrine
    "polydipsia": "excessive thirst diabetes",
    "polyphagia": "excessive hunger diabetes",
    "hyperglycemia": "high blood sugar diabetes",
    "hypoglycemia": "low blood sugar diabetes",
    "hypothyroidism": "underactive thyroid weight gain",
    "hyperthyroidism": "overactive thyroid weight loss",
    "alopecia": "hair loss thyroid hypothyroidism",
    "obesity": "overweight obesity",

    # Mental health
    "anhedonia": "no interest depression",
    "insomnia": "sleep problems insomnia",
    "somnolence": "excessive sleepiness",

    # Women's health
    "dysmenorrhea": "painful periods menstrual endometriosis",
    "amenorrhea": "missed periods pcos",
    "menorrhagia": "heavy periods",
}

# ── L5: Phrase dictionary ──────────────────────────────────────────────
# Format: phrase → disease.  Scored by word count (longer = more specific = higher weight).
_PHRASES = {

    # ── Seasonal Flu ──────────────────────────────────────────────────
    "fever and body ache": "Seasonal Flu", "flu symptoms": "Seasonal Flu",
    "influenza": "Seasonal Flu", "flu like": "Seasonal Flu",
    "flu": "Seasonal Flu", "fever": "Seasonal Flu",

    # ── Common Cold ───────────────────────────────────────────────────
    "common cold": "Common Cold", "runny nose": "Common Cold",
    "blocked nose": "Common Cold", "stuffy nose": "Common Cold",
    "nasal congestion": "Common Cold", "sore throat": "Common Cold",
    "sneezing cold": "Common Cold", "cold symptoms": "Common Cold",

    # ── Bronchitis ────────────────────────────────────────────────────
    "productive cough": "Bronchitis", "cough with phlegm": "Bronchitis",
    "cough with mucus": "Bronchitis", "wet cough": "Bronchitis",
    "persistent cough": "Bronchitis", "chest cough": "Bronchitis",
    "phlegm": "Bronchitis", "bronchitis": "Bronchitis",
    "mucus cough": "Bronchitis",

    # ── Pneumonia ─────────────────────────────────────────────────────
    "pneumonia": "Pneumonia", "lung infection": "Pneumonia",
    "chest infection": "Pneumonia", "difficulty breathing fever": "Pneumonia",

    # ── Tuberculosis ──────────────────────────────────────────────────
    "tuberculosis": "Tuberculosis", "tb": "Tuberculosis",
    "coughing blood": "Tuberculosis", "blood in cough": "Tuberculosis",
    "night sweats cough": "Tuberculosis", "prolonged cough": "Tuberculosis",
    "night sweats": "Tuberculosis",
    "night sweat": "Tuberculosis",
    "sweating at night": "Tuberculosis",
    "waking up sweating": "Tuberculosis",
    "breathless on exertion": "COPD",
    "breathless exertion": "COPD",
    "difficulty breathing exertion": "COPD",
    "copd exertion": "COPD",
    "chest tight when running": "Asthma",
    "chest tight running": "Asthma",
    "tight when run": "Asthma",
    "tight chest run": "Asthma",

    # ── Malaria ───────────────────────────────────────────────────────
    "malaria": "Malaria", "cyclic fever": "Malaria",
    "chills and fever": "Malaria", "shivering fever": "Malaria",

    # ── Dengue Fever ─────────────────────────────────────────────────
    "dengue fever": "Dengue Fever", "dengue": "Dengue Fever",
    "platelet drop": "Dengue Fever", "platelet low": "Dengue Fever",
    "rash with fever": "Dengue Fever",

    # ── Gastritis ─────────────────────────────────────────────────────
    "nausea vomiting": "Gastritis", "nausea and vomiting": "Gastritis",
    "stomach pain": "Gastritis", "stomach ache": "Gastritis",
    "stomach inflammation": "Gastritis", "gastritis": "Gastritis",
    "vomiting": "Gastritis", "nausea": "Gastritis",
    "upset stomach": "Gastritis", "indigestion": "Gastritis",
    "stomach discomfort": "Gastritis", "epigastric pain": "Gastritis",
    "abdominal pain": "Gastritis",

    # ── GERD ─────────────────────────────────────────────────────────
    "acid reflux": "GERD", "heartburn": "GERD",
    "burning after eating": "GERD", "stomach burns after meals": "GERD",
    "throat burn": "GERD", "sour belching": "GERD",
    "regurgitation": "GERD", "gerd": "GERD",
    "acidity": "GERD", "acid indigestion": "GERD",
    "chest burn eating": "GERD",

    # ── Gastroenteritis ───────────────────────────────────────────────
    "diarrhea vomiting": "Gastroenteritis",
    "diarrhea and vomiting": "Gastroenteritis",
    "food poisoning": "Gastroenteritis", "stomach bug": "Gastroenteritis",
    "loose motions vomiting": "Gastroenteritis",
    "diarrhea": "Gastroenteritis", "gastroenteritis": "Gastroenteritis",
    "stomach flu": "Gastroenteritis",

    # ── Peptic Ulcer ─────────────────────────────────────────────────
    "peptic ulcer": "Peptic Ulcer", "stomach ulcer": "Peptic Ulcer",
    "gastric ulcer": "Peptic Ulcer", "duodenal ulcer": "Peptic Ulcer",
    "ulcer": "Peptic Ulcer", "h pylori": "Peptic Ulcer",
    "burning stomach empty": "Peptic Ulcer",
    "stomach bleeding": "Peptic Ulcer",
    "mouth ulcer": "Peptic Ulcer",

    # ── Irritable Bowel Syndrome ─────────────────────────────────────
    "irritable bowel": "Irritable Bowel Syndrome",
    "ibs": "Irritable Bowel Syndrome", "bloating": "Irritable Bowel Syndrome",
    "constipation": "Irritable Bowel Syndrome",
    "abdominal bloating": "Irritable Bowel Syndrome",
    "gas pain": "Irritable Bowel Syndrome",
    "alternating bowel": "Irritable Bowel Syndrome",

    # ── Crohn's Disease ───────────────────────────────────────────────
    "crohn": "Crohn's Disease", "inflammatory bowel": "Crohn's Disease",
    "blood in stool": "Crohn's Disease", "chronic diarrhea": "Crohn's Disease",
    "crohns disease": "Crohn's Disease",

    # ── Chemotherapy Nausea ───────────────────────────────────────────
    "chemo nausea": "Chemotherapy Nausea",
    "chemotherapy nausea": "Chemotherapy Nausea",
    "cancer treatment vomiting": "Chemotherapy Nausea",

    # ── Hypertension ─────────────────────────────────────────────────
    "high blood pressure": "Hypertension", "hypertension": "Hypertension",
    "elevated blood pressure": "Hypertension",
    "blood pressure high": "Hypertension",
    "nosebleed hypertension": "Hypertension",
    "epistaxis": "Hypertension",

    # ── Angina ────────────────────────────────────────────────────────
    "angina": "Angina", "chest pain": "Angina",
    "chest pressure": "Angina", "chest discomfort": "Angina",
    "chest squeezing": "Angina",

    # ── Arrhythmia ────────────────────────────────────────────────────
    "palpitations": "Arrhythmia", "irregular heartbeat": "Arrhythmia",
    "heart racing": "Arrhythmia", "heart flutter": "Arrhythmia",
    "rapid heartbeat": "Arrhythmia", "fast heartbeat": "Arrhythmia",
    "heart skipping": "Arrhythmia", "arrhythmia": "Arrhythmia",
    "tachycardia": "Arrhythmia",

    # ── Heart Failure ─────────────────────────────────────────────────
    "heart failure": "Heart Failure", "swollen ankles": "Heart Failure",
    "swollen legs": "Heart Failure", "leg swelling": "Heart Failure",
    "ankle swelling": "Heart Failure", "edema": "Heart Failure",
    "fluid retention": "Heart Failure", "breathless lying down": "Heart Failure",

    # ── Hyperlipidemia ────────────────────────────────────────────────
    "high cholesterol": "Hyperlipidemia", "cholesterol": "Hyperlipidemia",
    "high triglycerides": "Hyperlipidemia",
    "triglycerides": "Hyperlipidemia", "hyperlipidemia": "Hyperlipidemia",

    # ── Type 2 Diabetes ───────────────────────────────────────────────
    "type 2 diabetes": "Type 2 Diabetes", "diabetes": "Type 2 Diabetes",
    "high blood sugar": "Type 2 Diabetes",
    "excessive thirst": "Type 2 Diabetes",
    "increased thirst": "Type 2 Diabetes",
    "frequent urination diabetes": "Type 2 Diabetes",
    "polydipsia": "Type 2 Diabetes", "polyuria": "Type 2 Diabetes",
    "hyperglycemia": "Type 2 Diabetes", "insulin resistance": "Type 2 Diabetes",

    # ── Type 1 Diabetes ───────────────────────────────────────────────
    "type 1 diabetes": "Type 1 Diabetes",
    "insulin dependent": "Type 1 Diabetes",
    "juvenile diabetes": "Type 1 Diabetes",

    # ── Hypothyroidism ────────────────────────────────────────────────
    "hypothyroidism": "Hypothyroidism", "underactive thyroid": "Hypothyroidism",
    "low thyroid": "Hypothyroidism", "thyroid low": "Hypothyroidism",
    "hair loss thyroid": "Hypothyroidism",
    "weight gain thyroid": "Hypothyroidism",
    "alopecia thyroid": "Hypothyroidism",

    # ── Hyperthyroidism ───────────────────────────────────────────────
    "hyperthyroidism": "Hyperthyroidism", "overactive thyroid": "Hyperthyroidism",
    "high thyroid": "Hyperthyroidism",

    # ── Obesity ───────────────────────────────────────────────────────
    "obesity": "Obesity", "overweight": "Obesity", "obese": "Obesity",

    # ── Migraine ─────────────────────────────────────────────────────
    "migraine": "Migraine", "throbbing headache": "Migraine",
    "severe headache": "Migraine", "one sided headache": "Migraine",
    "headache with nausea": "Migraine", "cluster headache": "Migraine",
    "pulsating headache": "Migraine", "headache": "Migraine",

    # ── Epilepsy ─────────────────────────────────────────────────────
    "epilepsy": "Epilepsy", "seizure": "Epilepsy",
    "convulsion": "Epilepsy", "fits": "Epilepsy",
    "epileptic fit": "Epilepsy",

    # ── Parkinson's Disease ───────────────────────────────────────────
    "parkinson": "Parkinson's Disease", "hand tremor": "Parkinson's Disease",
    "resting tremor": "Parkinson's Disease", "slow movement": "Parkinson's Disease",
    "shuffling walk": "Parkinson's Disease", "tremor": "Parkinson's Disease",
    "muscle rigidity parkinson": "Parkinson's Disease",

    # ── Alzheimer's Disease ───────────────────────────────────────────
    "alzheimer": "Alzheimer's Disease", "dementia": "Alzheimer's Disease",
    "memory loss": "Alzheimer's Disease", "forgetfulness": "Alzheimer's Disease",
    "cognitive decline": "Alzheimer's Disease", "memory problems": "Alzheimer's Disease",

    # ── Vertigo ───────────────────────────────────────────────────────
    "vertigo": "Vertigo", "dizziness": "Vertigo",
    "spinning sensation": "Vertigo", "room spinning": "Vertigo",
    "balance problem": "Vertigo", "loss of balance": "Vertigo",
    "lightheaded": "Vertigo", "giddiness": "Vertigo",

    # ── Multiple Sclerosis ────────────────────────────────────────────
    "multiple sclerosis": "Multiple Sclerosis",
    "numb limbs": "Multiple Sclerosis", "ms symptoms": "Multiple Sclerosis",

    # ── Anxiety Disorder ─────────────────────────────────────────────
    "anxiety": "Anxiety Disorder", "panic attack": "Anxiety Disorder",
    "nervousness": "Anxiety Disorder", "excessive worry": "Anxiety Disorder",
    "panic disorder": "Anxiety Disorder", "generalised anxiety": "Anxiety Disorder",

    # ── Major Depressive Disorder ─────────────────────────────────────
    "depression": "Major Depressive Disorder",
    "persistent sadness": "Major Depressive Disorder",
    "hopelessness": "Major Depressive Disorder",
    "anhedonia": "Major Depressive Disorder",
    "suicidal thoughts": "Major Depressive Disorder",
    "low mood": "Major Depressive Disorder",

    # ── Insomnia ─────────────────────────────────────────────────────
    "insomnia": "Insomnia", "sleeplessness": "Insomnia",
    "cannot sleep": "Insomnia", "trouble sleeping": "Insomnia",
    "poor sleep": "Insomnia", "sleep problems": "Insomnia",

    # ── Bipolar Disorder ─────────────────────────────────────────────
    "bipolar": "Bipolar Disorder", "mood swings": "Bipolar Disorder",
    "manic episode": "Bipolar Disorder", "mania": "Bipolar Disorder",

    # ── Schizophrenia ─────────────────────────────────────────────────
    "schizophrenia": "Schizophrenia", "hallucination": "Schizophrenia",
    "hearing voices": "Schizophrenia", "psychosis": "Schizophrenia",
    "paranoia": "Schizophrenia", "delusion": "Schizophrenia",

    # ── ADHD ─────────────────────────────────────────────────────────
    "adhd": "ADHD", "attention deficit": "ADHD",
    "hyperactivity": "ADHD", "cannot concentrate": "ADHD",
    "difficulty focusing": "ADHD", "inattention": "ADHD",

    # ── PTSD ─────────────────────────────────────────────────────────
    "ptsd": "PTSD", "flashback": "PTSD",
    "post traumatic": "PTSD", "trauma": "PTSD",

    # ── Arthritis ─────────────────────────────────────────────────────
    "arthritis": "Arthritis", "joint pain": "Arthritis",
    "joint inflammation": "Arthritis", "swollen joints": "Arthritis",
    "stiff joints": "Arthritis", "rheumatoid": "Arthritis",
    "joint swelling": "Arthritis", "joint stiffness": "Arthritis",

    # ── Osteoporosis ─────────────────────────────────────────────────
    "osteoporosis": "Osteoporosis", "bone loss": "Osteoporosis",
    "weak bones": "Osteoporosis", "brittle bones": "Osteoporosis",
    "low bone density": "Osteoporosis", "bone thinning": "Osteoporosis",

    # ── Gout ─────────────────────────────────────────────────────────
    "gout": "Gout", "uric acid": "Gout", "big toe pain": "Gout",
    "big toe swelling": "Gout", "high uric acid": "Gout",
    "gouty arthritis": "Gout",

    # ── Lower Back Pain ───────────────────────────────────────────────
    "lower back pain": "Lower Back Pain", "back pain": "Lower Back Pain",
    "lumbar pain": "Lower Back Pain", "sciatica": "Lower Back Pain",
    "slipped disc": "Lower Back Pain", "herniated disc": "Lower Back Pain",
    "lumbago": "Lower Back Pain",

    # ── Fibromyalgia ─────────────────────────────────────────────────
    "fibromyalgia": "Fibromyalgia", "widespread pain": "Fibromyalgia",
    "muscle pain all over": "Fibromyalgia", "chronic muscle pain": "Fibromyalgia",

    # ── Chronic Pain ─────────────────────────────────────────────────
    "chronic pain": "Chronic Pain", "persistent pain": "Chronic Pain",
    "long term pain": "Chronic Pain",

    # ── Neuropathic Pain ─────────────────────────────────────────────
    "neuropathic pain": "Neuropathic Pain", "nerve pain": "Neuropathic Pain",
    "burning pain": "Neuropathic Pain", "neuropathy": "Neuropathic Pain",
    "pins and needles": "Neuropathic Pain", "tingling": "Neuropathic Pain",
    "numbness": "Neuropathic Pain", "electric shock pain": "Neuropathic Pain",
    "numb fingers": "Neuropathic Pain",

    # ── Post-operative Pain ───────────────────────────────────────────
    "post surgery pain": "Post-operative Pain",
    "after surgery pain": "Post-operative Pain",
    "post operative": "Post-operative Pain",

    # ── Cancer Pain Management ────────────────────────────────────────
    "cancer pain": "Cancer Pain Management",
    "palliative pain": "Cancer Pain Management",

    # ── Asthma ────────────────────────────────────────────────────────
    "asthma": "Asthma", "wheezing": "Asthma",
    "breathlessness": "Asthma", "chest tightness": "Asthma",
    "shortness of breath": "Asthma", "asthma attack": "Asthma",

    # ── COPD ─────────────────────────────────────────────────────────
    "copd": "COPD", "chronic obstructive": "COPD",
    "emphysema": "COPD", "chronic bronchitis copd": "COPD",
    "breathless on exertion": "COPD",

    # ── Allergic Rhinitis ─────────────────────────────────────────────
    "allergic rhinitis": "Allergic Rhinitis", "hay fever": "Allergic Rhinitis",
    "sneezing": "Allergic Rhinitis", "nasal allergy": "Allergic Rhinitis",
    "pollen allergy": "Allergic Rhinitis", "dust allergy": "Allergic Rhinitis",
    "itchy nose": "Allergic Rhinitis",

    # ── Acne Vulgaris ─────────────────────────────────────────────────
    "acne": "Acne Vulgaris", "pimples": "Acne Vulgaris",
    "blackheads": "Acne Vulgaris", "whiteheads": "Acne Vulgaris",
    "cystic acne": "Acne Vulgaris", "face breakout": "Acne Vulgaris",
    "zits": "Acne Vulgaris",

    # ── Eczema ────────────────────────────────────────────────────────
    "eczema": "Eczema", "atopic dermatitis": "Eczema",
    "skin itching": "Eczema", "itchy skin": "Eczema",
    "skin rash": "Eczema", "dry itchy skin": "Eczema",
    "dermatitis": "Eczema",

    # ── Psoriasis ─────────────────────────────────────────────────────
    "psoriasis": "Psoriasis", "scaling skin": "Psoriasis",
    "skin plaques": "Psoriasis", "scaly patches": "Psoriasis",
    "silvery scales": "Psoriasis", "skin flaking": "Psoriasis",

    # ── Fungal Infection ─────────────────────────────────────────────
    "fungal infection": "Fungal Infection", "ringworm": "Fungal Infection",
    "athlete's foot": "Fungal Infection", "nail fungus": "Fungal Infection",
    "thrush": "Fungal Infection", "candida": "Fungal Infection",
    "jock itch": "Fungal Infection",

    # ── Conjunctivitis ────────────────────────────────────────────────
    "conjunctivitis": "Conjunctivitis", "pink eye": "Conjunctivitis",
    "eye infection": "Conjunctivitis", "eye redness": "Conjunctivitis",
    "eye irritation": "Conjunctivitis", "eye discharge": "Conjunctivitis",
    "watery eye": "Conjunctivitis", "itchy eye": "Conjunctivitis",
    "red eye": "Conjunctivitis", "eye inflammation": "Conjunctivitis",
    "burning eye": "Conjunctivitis",

    # ── Glaucoma ─────────────────────────────────────────────────────
    "glaucoma": "Glaucoma", "eye pressure": "Glaucoma",
    "high eye pressure": "Glaucoma", "blurred vision": "Glaucoma",
    "tunnel vision": "Glaucoma", "vision loss": "Glaucoma",

    # ── Otitis Media ─────────────────────────────────────────────────
    "otitis media": "Otitis Media", "ear infection": "Otitis Media",
    "ear pain": "Otitis Media", "earache": "Otitis Media",
    "ear discharge": "Otitis Media", "middle ear infection": "Otitis Media",
    "tinnitus": "Otitis Media", "ringing ear": "Otitis Media",

    # ── UTI ───────────────────────────────────────────────────────────
    "urinary tract infection": "Urinary Tract Infection",
    "uti": "Urinary Tract Infection",
    "burning urination": "Urinary Tract Infection",
    "painful urination": "Urinary Tract Infection",
    "urethra inflammation": "Urinary Tract Infection",
    "urethritis": "Urinary Tract Infection",
    "bladder infection": "Urinary Tract Infection",
    "blood in urine": "Urinary Tract Infection",
    "cloudy urine": "Urinary Tract Infection",
    "frequent urge to urinate": "Urinary Tract Infection",
    "cystitis": "Urinary Tract Infection",
    "dysuria": "Urinary Tract Infection",
    "hematuria": "Urinary Tract Infection",

    # ── CKD ───────────────────────────────────────────────────────────
    "kidney disease": "Chronic Kidney Disease",
    "kidney failure": "Chronic Kidney Disease",
    "chronic kidney disease": "Chronic Kidney Disease",
    "renal failure": "Chronic Kidney Disease",
    "kidney stones": "Chronic Kidney Disease",
    "flank pain": "Chronic Kidney Disease",

    # ── BPH ───────────────────────────────────────────────────────────
    "enlarged prostate": "Benign Prostatic Hyperplasia",
    "bph": "Benign Prostatic Hyperplasia",
    "difficulty urinating": "Benign Prostatic Hyperplasia",
    "nocturia": "Benign Prostatic Hyperplasia",
    "weak urine stream": "Benign Prostatic Hyperplasia",
    "frequent night urination": "Benign Prostatic Hyperplasia",

    # ── PCOS ─────────────────────────────────────────────────────────
    "pcos": "Polycystic Ovary Syndrome",
    "irregular periods": "Polycystic Ovary Syndrome",
    "polycystic ovary": "Polycystic Ovary Syndrome",
    "hormonal imbalance": "Polycystic Ovary Syndrome",

    # ── Endometriosis ─────────────────────────────────────────────────
    "endometriosis": "Endometriosis", "pelvic pain": "Endometriosis",
    "painful periods": "Endometriosis", "dysmenorrhea": "Endometriosis",
}

# Build sorted phrase list once at module load (longest first for greedy matching)
_PHRASES_SORTED = sorted(_PHRASES.keys(), key=lambda p: len(p.split()), reverse=True)

# ── L6: Regex natural language patterns ───────────────────────────────
# Format: (compiled_pattern, disease_name)
# Each fires on raw, spell-corrected, suffix-normalised, and jargon-expanded text.
_PATTERNS = [
    # ── Vomiting / Nausea / GI ────────────────────────────────────────
    (_re.compile(r'\b(vomit|vomiting|vomitted|vomits|vomitting|throwing up|threw up|been sick|feel sick)\b', _re.I), "Gastritis"),
    (_re.compile(r'\b(nausea|nauseous|nauseated|queasiness|queasy)\b', _re.I), "Gastritis"),
    (_re.compile(r'\bstomach\b.{0,30}\b(hurt|pain|ache|burn|cramp|upset|irrit|discomfort)\b', _re.I), "Gastritis"),
    (_re.compile(r'\b(hurt|pain|ache|burn)\b.{0,30}\bstomach\b', _re.I), "Gastritis"),
    (_re.compile(r'\b(tummy|belly|gut)\b.{0,30}\b(hurt|pain|ache|cramp|upset)\b', _re.I), "Gastritis"),
    (_re.compile(r'\b(indigestion|dyspepsia|upset stomach)\b', _re.I), "Gastritis"),
    # GERD
    (_re.compile(r'\b(acid reflux|heartburn|acidity|regurgitat)\b', _re.I), "GERD"),
    (_re.compile(r'\bbur(n|ns|ning|nt)\b.{0,40}\b(eat|meal|food|after)\b', _re.I), "GERD"),
    (_re.compile(r'\b(eat|meal|food)\b.{0,30}\bbur(n|ns|ning)\b', _re.I), "GERD"),
    (_re.compile(r'\bstomach\b.{0,20}\bbur(n|ns|ning)\b', _re.I), "GERD"),
    (_re.compile(r'\b(swallow|swallowing|swallowed)\b.{0,30}\b(hard|difficult|painful|trouble|pain)\b', _re.I), "GERD"),
    (_re.compile(r'\b(hard|difficult|trouble|pain)\b.{0,20}\bswallow', _re.I), "GERD"),
    (_re.compile(r'\bdifficulty.{0,10}swallow', _re.I), "GERD"),
    (_re.compile(r'\bgerd|acid reflux\b', _re.I), "GERD"),
    # Gastroenteritis — must outscore Gastritis when both vomiting+loose motions present
    (_re.compile(r'\b(vomit|vomiting)\b.{0,40}\b(diarrhea|loose|stool|motion|motions|gastroenteritis)\b', _re.I), "Gastroenteritis"),
    (_re.compile(r'\b(diarrhea|loose motions|loose stools)\b.{0,40}\b(vomit|vomiting)\b', _re.I), "Gastroenteritis"),
    (_re.compile(r'\b(loose|watery|runny)\b.{0,20}\b(stool|motion|bowel|poo|poop)\b', _re.I), "Gastroenteritis"),
    (_re.compile(r'\b(stool|motion|bowel|poo)\b.{0,20}\b(loose|watery|blood)\b', _re.I), "Gastroenteritis"),
    (_re.compile(r'\bfood.{0,5}poison(ing)?\b', _re.I), "Gastroenteritis"),
    (_re.compile(r'\b(stomach bug|stomach flu|stomach virus)\b', _re.I), "Gastroenteritis"),
    (_re.compile(r'\bgastroenteritis\b', _re.I), "Gastroenteritis"),
    # Peptic Ulcer
    (_re.compile(r'\b(stomach|gastric|duodenal|peptic)\b.{0,15}\bulcer\b', _re.I), "Peptic Ulcer"),
    (_re.compile(r'\bulcers?\b', _re.I), "Peptic Ulcer"),
    (_re.compile(r'\bmouth ulcer|canker sore|tongue sore\b', _re.I), "Peptic Ulcer"),
    # IBS / Constipation
    (_re.compile(r'\b(bloat|bloated|bloating|gassy|gas pain|flatulen)\b', _re.I), "Irritable Bowel Syndrome"),
    (_re.compile(r'\b(constipat|cant poop|cant go|no bowel)\b', _re.I), "Irritable Bowel Syndrome"),
    (_re.compile(r'\bhard stool|hard to pass|straining\b', _re.I), "Irritable Bowel Syndrome"),

    # ── Appetite / Mood ─────────────────────────────────────────────────
    (_re.compile(r'\b(loss of appetite|no appetite|lost appetite|cant eat|appetite loss)\b', _re.I), "Major Depressive Disorder"),
    (_re.compile(r'\b(no interest in food|not eating|stopped eating)\b', _re.I), "Major Depressive Disorder"),

    # ── Facial / Body swelling → Heart Failure ─────────────────────────
    (_re.compile(r'\b(face|facial)\b.{0,15}\b(swollen|swell|puffy|puff|swelling)\b', _re.I), "Heart Failure"),
    (_re.compile(r'\b(swollen|puffy)\b.{0,15}\b(face|facial)\b', _re.I), "Heart Failure"),
    (_re.compile(r'\bfacial swelling\b', _re.I), "Heart Failure"),

    # ── Scalp / Skin peeling → Psoriasis ──────────────────────────────
    (_re.compile(r'\bscalp\b.{0,25}\b(itch|itch|flak|scal|dry|peel|irritat)\b', _re.I), "Psoriasis"),
    (_re.compile(r'\b(itch|flak|scal|dry|peel)\b.{0,15}\bscalp\b', _re.I), "Psoriasis"),
    (_re.compile(r'\bitchy.{0,10}flaky.{0,10}scalp\b', _re.I), "Psoriasis"),
    (_re.compile(r'\bflaky.{0,10}scalp\b', _re.I), "Psoriasis"),
    (_re.compile(r'\bskin\b.{0,20}\bpeel(ing|s|ed|badly)?\b', _re.I), "Psoriasis"),
    (_re.compile(r'\bpeel(ing|s|ed)?\b.{0,20}\bskin\b', _re.I), "Psoriasis"),

    # ── Persistent cough → Bronchitis ─────────────────────────────────
    (_re.compile(r'\b(cant stop|cannot stop|keep|constant|persistent|non.?stop)\b.{0,10}\bcough(ing)?\b', _re.I), "Bronchitis"),
    (_re.compile(r'\bcough(ing)?\b.{0,10}\b(constant|persist|stop|all day|non.?stop)\b', _re.I), "Bronchitis"),

    # ── Chest tight on exertion → Asthma (higher weight than Angina) ──
    (_re.compile(r'\bchest\b.{0,15}\btight\b.{0,20}\b(run|running|walk|walking|exert|exercise|stair|effort)\b', _re.I), "Asthma"),
    (_re.compile(r'\b(run|running|walk|walking|exercise|exert)\b.{0,20}\bchest\b.{0,15}\btight\b', _re.I), "Asthma"),
    (_re.compile(r'\btight.{0,10}chest.{0,20}\b(run|running|walk|exercise)\b', _re.I), "Asthma"),

    # ── Night sweats → Tuberculosis (stronger than Hyperthyroidism) ────
    (_re.compile(r'\bnight\b.{0,10}\bsweat(s|ing)?\b', _re.I), "Tuberculosis"),
    (_re.compile(r'\bsweat(s|ing)?\b.{0,10}\bnight\b', _re.I), "Tuberculosis"),
    (_re.compile(r'\bwak(e|ing).{0,10}(up.{0,10})?(sweat|drench|soak)', _re.I), "Tuberculosis"),

    # ── Eye ────────────────────────────────────────────────────────────
    (_re.compile(r'\beyes?\b.{0,25}\b(red|pink|itch|burn|irrit|discharg|water|swell|infect|inflam|crust|gunk|stuck|pus)\b', _re.I), "Conjunctivitis"),
    (_re.compile(r'\b(red|pink|itch|burn|irrit|discharg|water|swell|infect|inflam|crust)\b.{0,25}\beyes?\b', _re.I), "Conjunctivitis"),
    (_re.compile(r'\beye.{0,10}(irritat|iritat|burning|itching|redne|redness|discharge|watery|crusty|gunk|stuck)\b', _re.I), "Conjunctivitis"),
    (_re.compile(r'\beyes?\b.{0,20}\b(pressure|blur|vision|tunnel)\b', _re.I), "Glaucoma"),
    (_re.compile(r'\b(blurr|blur).{0,10}vision\b', _re.I), "Glaucoma"),
    (_re.compile(r'\bcant see clearly|vision loss|peripheral vision\b', _re.I), "Glaucoma"),

    # ── Ear ────────────────────────────────────────────────────────────
    (_re.compile(r'\bears?\b.{0,25}\b(hurt|pain|ach|infect|discharg|pus|pressur|block|ring|ringing)\b', _re.I), "Otitis Media"),
    (_re.compile(r'\b(hurt|pain|ach|infect|ring|ringing)\b.{0,25}\bears?\b', _re.I), "Otitis Media"),
    (_re.compile(r'\bearache|ear ache|blocked ear|hearing loss\b', _re.I), "Otitis Media"),

    # ── Urinary / Urethra ─────────────────────────────────────────────
    (_re.compile(r'\b(urethra|urethral|bladder)\b.{0,25}\b(pain|burn|irrit|inflam|infect|discharg)\b', _re.I), "Urinary Tract Infection"),
    (_re.compile(r'\b(burn|pain|sting|hurt)\b.{0,25}\b(pee|urinat|toilet|wee|urinary|bathroom)\b', _re.I), "Urinary Tract Infection"),
    (_re.compile(r'\b(pee|urinat|toilet|wee)\b.{0,25}\b(burn|pain|sting|hurt|blood|cloud)\b', _re.I), "Urinary Tract Infection"),
    (_re.compile(r'\bblood\b.{0,15}\b(urine|pee|wee)\b', _re.I), "Urinary Tract Infection"),
    (_re.compile(r'\b(urine|pee|wee)\b.{0,15}\b(blood|cloud|dark|smelly|smell)\b', _re.I), "Urinary Tract Infection"),
    (_re.compile(r'\burethrit|cystitis\b', _re.I), "Urinary Tract Infection"),
    # BPH
    (_re.compile(r'\b(keep|always|constant|frequent)\b.{0,20}\b(toilet|pee|urinat|wee)\b.{0,20}\bnight\b', _re.I), "Benign Prostatic Hyperplasia"),
    (_re.compile(r'\bnight.{0,10}\b(toilet|pee|urinat|wee|bathroom)\b', _re.I), "Benign Prostatic Hyperplasia"),
    (_re.compile(r'\bprostat|bph\b', _re.I), "Benign Prostatic Hyperplasia"),

    # ── Chest / Heart / Breathing ─────────────────────────────────────
    (_re.compile(r'\bchest\b.{0,20}\b(pain|tight|press|squeez|discomfort|burn)\b', _re.I), "Angina"),
    (_re.compile(r'\bheart\b.{0,20}\b(pain|ache)\b', _re.I), "Angina"),
    (_re.compile(r'\bheart\b.{0,25}\b(beat|racing|fast|pound|flutter|skip|irreg|palpitat)\b', _re.I), "Arrhythmia"),
    (_re.compile(r'\b(racing|fast|pounding|fluttering|skipping|palpitat)\b.{0,15}\bheart\b', _re.I), "Arrhythmia"),
    (_re.compile(r'\b(fluttery|fluttering)\b.{0,15}\bheart\b', _re.I), "Arrhythmia"),
    (_re.compile(r'\b(hard|difficult|trouble|cannot|cant)\b.{0,20}\b(breath|breathe|breathing)\b', _re.I), "COPD"),
    (_re.compile(r'\bbreath(e|less|ing)?\b.{0,20}\b(walk|run|climb|exert|effort|stair|difficult|hard)\b', _re.I), "COPD"),
    (_re.compile(r'\b(walk|run|climb|stair|exert)\b.{0,20}\bbreath(e|less|ing)?\b', _re.I), "COPD"),
    (_re.compile(r'\bshort.{0,10}breath\b', _re.I), "Asthma"),
    (_re.compile(r'\bchest.{0,10}tight\b', _re.I), "Asthma"),
    (_re.compile(r'\bwheez(ing|es|ed)?\b', _re.I), "Asthma"),
    (_re.compile(r'\bcough\b.{0,20}\b(phlegm|mucus|sputum|blood|chronic|persist)\b', _re.I), "Bronchitis"),
    (_re.compile(r'\blung\b.{0,20}\b(infect|pain|congest|fluid)\b', _re.I), "Pneumonia"),
    # Swollen ankles → Heart Failure
    (_re.compile(r'\b(ankle|leg|feet|foot)\b.{0,15}\b(swollen|swell|puffy|puff)\b', _re.I), "Heart Failure"),
    (_re.compile(r'\b(swollen|puffy)\b.{0,15}\b(ankle|leg|feet|foot)\b', _re.I), "Heart Failure"),
    # Nosebleed → Hypertension
    (_re.compile(r'\bnosebleed|nose.?bleed|epistaxis\b', _re.I), "Hypertension"),
    (_re.compile(r'\bhigh.{0,10}\b(blood pressure|bp)\b', _re.I), "Hypertension"),
    (_re.compile(r'\b(blood pressure|bp)\b.{0,10}high\b', _re.I), "Hypertension"),

    # ── Skin ───────────────────────────────────────────────────────────
    (_re.compile(r'\bskin\b.{0,25}\b(itch|rash|red|dry|inflam|irrit|sore|burn)\b', _re.I), "Eczema"),
    (_re.compile(r'\b(itch|rash)\b.{0,15}\bskin\b', _re.I), "Eczema"),
    (_re.compile(r'\bskin\b.{0,20}\b(peel|scal|flak|plaque|silver|thick|patch)\b', _re.I), "Psoriasis"),
    (_re.compile(r'\b(peel|scal|flak|plaque|peeling)\b.{0,20}\bskin\b', _re.I), "Psoriasis"),
    (_re.compile(r'\bscalp\b.{0,20}\b(itch|flak|scal|dry|peel)\b', _re.I), "Psoriasis"),
    (_re.compile(r'\bskin\b.{0,20}\b(fungal|ring|jock|athlete|candid|thrush)\b', _re.I), "Fungal Infection"),
    (_re.compile(r'\b(pimple|zit|blackhead|whitehead|acne|breakout)\b', _re.I), "Acne Vulgaris"),

    # ── Joints / Bone / Muscle ────────────────────────────────────────
    (_re.compile(r'\bjoint\b.{0,25}\b(pain|hurt|swell|stiff|ach|inflam)\b', _re.I), "Arthritis"),
    (_re.compile(r'\b(pain|hurt|swell|stiff|aching)\b.{0,20}\bjoint\b', _re.I), "Arthritis"),
    (_re.compile(r'\bjoints?\b.{0,20}\b(hurt|sore|bad|ach|much|really|all|awful|terrible)\b', _re.I), "Arthritis"),
    (_re.compile(r'\bknee\b.{0,20}\b(pain|hurt|swell|ach|stiff)\b', _re.I), "Arthritis"),
    (_re.compile(r'\b(hip|shoulder|wrist|ankle|elbow)\b.{0,20}\b(pain|hurt|swell|stiff)\b', _re.I), "Arthritis"),
    (_re.compile(r'\bbone\b.{0,20}\b(loss|weak|thin|brittle|fracture)\b', _re.I), "Osteoporosis"),
    (_re.compile(r'\btoe\b.{0,20}\b(pain|swell|red|inflam)\b', _re.I), "Gout"),
    (_re.compile(r'\bback\b.{0,20}\b(pain|hurt|ach|sore|bad|killing|kill)\b', _re.I), "Lower Back Pain"),
    (_re.compile(r'\b(pain|hurt|ach|killing)\b.{0,20}\bback\b', _re.I), "Lower Back Pain"),
    (_re.compile(r'\bsciatica\b', _re.I), "Lower Back Pain"),
    (_re.compile(r'\b(nerve|neuropath)\b.{0,15}\bpain\b', _re.I), "Neuropathic Pain"),
    (_re.compile(r'\b(numb|tingle|pins|needle)\b.{0,25}\b(hand|finger|foot|feet|toe|arm|leg)\b', _re.I), "Neuropathic Pain"),
    (_re.compile(r'\b(hand|finger|foot|feet|arm|leg)\b.{0,25}\b(numb|tingle|pins|needle)\b', _re.I), "Neuropathic Pain"),
    (_re.compile(r'\bmuscle\b.{0,20}\bpain\b.{0,20}\b(all|every|whole|wide|everywhere)\b', _re.I), "Fibromyalgia"),

    # ── Neurological ───────────────────────────────────────────────────
    (_re.compile(r'\b(seizure|convuls|fit|epilept|fits)\b', _re.I), "Epilepsy"),
    (_re.compile(r'\b(tremble|trembling|tremor|tremors|shaking|shaky)\b.{0,20}\b(hand|arm|body|finger|limb)\b', _re.I), "Parkinson's Disease"),
    (_re.compile(r'\b(hand|arm|finger|limb)\b.{0,20}\b(tremble|trembling|tremor|shaking|shaky)\b', _re.I), "Parkinson's Disease"),
    (_re.compile(r'\b(forget|memory|remember|recall)\b.{0,30}\b(bad|poor|loss|problem|trouble|hard|things|keep|always)\b', _re.I), "Alzheimer's Disease"),
    (_re.compile(r'\b(memory|dementia|cognitive)\b', _re.I), "Alzheimer's Disease"),
    (_re.compile(r'\bkeep.{0,15}forget\b', _re.I), "Alzheimer's Disease"),
    (_re.compile(r'\b(dizzy|dizziness|spinning|vertigo|giddy|lightheaded)\b', _re.I), "Vertigo"),
    (_re.compile(r'\broom\b.{0,10}\bspin\b', _re.I), "Vertigo"),
    (_re.compile(r'\bheadache|head ache\b', _re.I), "Migraine"),
    (_re.compile(r'\bhead\b.{0,15}\b(ache|pain|pound|throb|hurt|killing)\b', _re.I), "Migraine"),
    (_re.compile(r'\b(migrain|migren)\b', _re.I), "Migraine"),
    (_re.compile(r'\bnumb\b.{0,20}\b(limb|arm|leg|body|side)\b', _re.I), "Multiple Sclerosis"),

    # ── Mental Health ──────────────────────────────────────────────────
    (_re.compile(r'\b(anxious|anxiety|panic|nervous|worry|worrying|nervousness)\b', _re.I), "Anxiety Disorder"),
    (_re.compile(r'\b(depress|depressed|hopeless|low mood|no joy|miserable)\b', _re.I), "Major Depressive Disorder"),
    (_re.compile(r'\b(cant|cannot|trouble|hard|difficult)\b.{0,15}\bsleep\b', _re.I), "Insomnia"),
    (_re.compile(r'\bsleep\b.{0,20}\b(problem|trouble|bad|poor|disturbance)\b', _re.I), "Insomnia"),
    (_re.compile(r'\b(mood swing|manic|bipolar|mania)\b', _re.I), "Bipolar Disorder"),
    (_re.compile(r'\b(hallucinat|see things|hear voices|delusion|paranoi|psychos)\b', _re.I), "Schizophrenia"),
    (_re.compile(r'\bi see things|see things that\b', _re.I), "Schizophrenia"),
    (_re.compile(r'\b(attention|hyperact|adhd|impuls|cant focus|cannot focus|cant concentrate)\b', _re.I), "ADHD"),
    (_re.compile(r'\b(trauma|flashback|nightmare|ptsd|post.traumat)\b', _re.I), "PTSD"),

    # ── Metabolic / Endocrine ─────────────────────────────────────────
    (_re.compile(r'\b(blood sugar|glucose)\b.{0,25}\b(high|elevat|up|wont|down|not|come)\b', _re.I), "Type 2 Diabetes"),
    (_re.compile(r'\bsugar.{0,15}\b(high|wont|not|down|elevat|coming)\b', _re.I), "Type 2 Diabetes"),
    (_re.compile(r'\b(thirst|thirsty)\b.{0,20}\b(excess|always|constant|extreme|lot|keep|all time)\b', _re.I), "Type 2 Diabetes"),
    (_re.compile(r'\balways.{0,10}\b(thirst|thirsty|drinking)\b', _re.I), "Type 2 Diabetes"),
    (_re.compile(r'\b(hair loss|hair falling|losing hair|bald)\b', _re.I), "Hypothyroidism"),
    (_re.compile(r'\b(tired|fatigue|exhausted)\b.{0,30}\b(gain|gaining|weight)\b', _re.I), "Hypothyroidism"),
    (_re.compile(r'\b(gain|gaining)\b.{0,15}\bweight\b', _re.I), "Hypothyroidism"),
    (_re.compile(r'\bweight.{0,10}\b(gain|gaining)\b', _re.I), "Hypothyroidism"),
    (_re.compile(r'\bthyroid\b.{0,20}\b(low|under|slow|inactive|deficien)\b', _re.I), "Hypothyroidism"),
    (_re.compile(r'\bthyroid\b.{0,20}\b(high|over|active|excess)\b', _re.I), "Hyperthyroidism"),
    (_re.compile(r'\b(losing|rapid)\b.{0,10}\bweight\b', _re.I), "Hyperthyroidism"),
    (_re.compile(r'\b(sweating profusely|night sweats|excessive sweating)\b', _re.I), "Hyperthyroidism"),
    (_re.compile(r'\bhigh.{0,10}\b(cholesterol|lipid|triglycerid)\b', _re.I), "Hyperlipidemia"),
    (_re.compile(r'\b(cholesterol|triglycerid|lipid)\b.{0,10}\b(high|elevat)\b', _re.I), "Hyperlipidemia"),

    # ── Respiratory Allergy ────────────────────────────────────────────
    (_re.compile(r'\b(sneez|pollen|dust|hay fever)\b.{0,25}\b(nose|eye|rhinit|allerg)\b', _re.I), "Allergic Rhinitis"),
    (_re.compile(r'\ballerg.{0,10}\b(nose|eye|sneez|rhinit)\b', _re.I), "Allergic Rhinitis"),
    (_re.compile(r'\bhay fever\b', _re.I), "Allergic Rhinitis"),

    # ── Infectious ─────────────────────────────────────────────────────
    (_re.compile(r'\b(malaria|cyclic fever|mosquito bite fever)\b', _re.I), "Malaria"),
    (_re.compile(r'\b(dengue|platelet|breakbone)\b', _re.I), "Dengue Fever"),
    (_re.compile(r'\b(tb|tuberculosis)\b.{0,20}\b(cough|blood|sputum|lung)\b', _re.I), "Tuberculosis"),
    (_re.compile(r'\bcough(ing)?\b.{0,10}\bblood\b', _re.I), "Tuberculosis"),
    (_re.compile(r'\bnight sweats\b.{0,20}\b(cough|weight|loss)\b', _re.I), "Tuberculosis"),

    # ── Women's Health ─────────────────────────────────────────────────
    (_re.compile(r'\b(period|menstrual)\b.{0,20}\b(irregular|miss|late|absent)\b', _re.I), "Polycystic Ovary Syndrome"),
    (_re.compile(r'\b(pcos|polycystic ovary)\b', _re.I), "Polycystic Ovary Syndrome"),
    (_re.compile(r'\b(pelvic|period|menstrual)\b.{0,20}\b(pain|cramp|severe)\b', _re.I), "Endometriosis"),
    (_re.compile(r'\bpainful.{0,10}\bperiods\b', _re.I), "Endometriosis"),

    # ── Kidney ─────────────────────────────────────────────────────────
    (_re.compile(r'\bkidney\b.{0,20}\b(pain|stone|infect|fail|disease|problem)\b', _re.I), "Chronic Kidney Disease"),
    (_re.compile(r'\bflank\b.{0,15}\b(pain|ache)\b', _re.I), "Chronic Kidney Disease"),
]


def symptoms_to_disease(symptom_string: str) -> str:
    """
    Map free-text symptom input → most likely disease.

    7 layers, all voting. Winner = highest cumulative score.
    Falls back to 'Seasonal Flu' ONLY if zero signal is found.
    """
    if not symptom_string or not symptom_string.strip():
        return "Seasonal Flu"

    scores: dict = {}

    def _vote(disease: str, pts: float):
        scores[disease] = scores.get(disease, 0.0) + pts

    # ── L0: basic normalisation ───────────────────────────────────────
    raw = symptom_string.lower().strip()
    raw = _re.sub(r"[^\w\s\-']", ' ', raw)
    raw = _re.sub(r'\s+', ' ', raw)

    # ── L1: spell / variant correction ───────────────────────────────
    # First try multi-word phrases (longest first), then word-by-word
    corrected = raw
    for phrase in sorted(_SPELL.keys(), key=len, reverse=True):
        if phrase in corrected:
            corrected = corrected.replace(phrase, _SPELL[phrase])

    # ── L2: synonym expansion (word level) ───────────────────────────
    def _expand(text):
        return ' '.join(_SYN.get(w, w) for w in text.split())

    expanded_raw = _expand(raw)
    expanded_cor = _expand(corrected)

    # ── L3: suffix stripping ─────────────────────────────────────────
    def _strip_suffixes(text):
        for pattern, repl in _SUFFIX_RULES:
            text = pattern.sub(repl, text)
        return text

    suffix_raw = _strip_suffixes(raw)
    suffix_cor = _strip_suffixes(corrected)

    # ── L4: jargon translation ────────────────────────────────────────
    def _translate_jargon(text):
        words = text.split()
        out = []
        i = 0
        while i < len(words):
            # Try 2-word jargon first
            if i + 1 < len(words):
                two = words[i] + ' ' + words[i + 1]
                if two in _JARGON:
                    out.append(_JARGON[two])
                    i += 2
                    continue
            w = words[i]
            out.append(_JARGON.get(w, w))
            i += 1
        return ' '.join(out)

    jargon_raw = _translate_jargon(raw)
    jargon_suf = _translate_jargon(suffix_raw)
    jargon_cor = _translate_jargon(suffix_cor)

    # All unique text variants to search
    variants = list(dict.fromkeys([
        raw, corrected, expanded_raw, expanded_cor,
        suffix_raw, suffix_cor, jargon_raw, jargon_suf, jargon_cor,
        _expand(suffix_raw), _expand(suffix_cor),
        _expand(jargon_raw), _expand(jargon_suf),
    ]))

    # ── L6: regex patterns (highest weight = 3) ───────────────────────
    for variant in variants:
        for pattern, disease in _PATTERNS:
            if pattern.search(variant):
                _vote(disease, 3)

    # ── L5: phrase dictionary (weight = word count, min 1) ────────────
    for variant in variants:
        for phrase in _PHRASES_SORTED:
            if phrase in variant:
                pts = max(1, len(phrase.split()))
                _vote(_PHRASES[phrase], pts)

    # ── L7: token overlap with disease names (weight = 0.5 per token) ─
    all_tokens = set()
    for v in variants:
        all_tokens.update(v.split())
    all_tokens = {t for t in all_tokens if len(t) > 3}

    for disease in _ENGINE_DISEASE_SET:
        d_tokens = set(disease.lower().split())
        overlap = all_tokens & d_tokens
        if overlap:
            _vote(disease, len(overlap) * 0.5)

    if not scores:
        return "Seasonal Flu"

    return max(scores, key=scores.get)


# ══════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════

def _composition_match(med_a: MedicineNode, med_b: MedicineNode) -> float:
    """
    Similarity between candidate (a) and baseline (b) based on mg values.
    Score per baseline ingredient:
      exact match (within 1%)  → 1.0
      close match (within 20%) → 0.5
      wrong dose               → 0.2
      absent                   → 0.0
    Normalised by max(len_a, len_b).
    """
    dict_a = {c["ingredient"].lower(): c["mg"] for c in med_a.composition}
    dict_b = {c["ingredient"].lower(): c["mg"] for c in med_b.composition}
    if not dict_b:
        return 0.0
    total = max(len(dict_a), len(dict_b))
    score = 0.0
    for ing, mg_b in dict_b.items():
        if ing not in dict_a:
            continue
        mg_a = dict_a[ing]
        if mg_b == 0:
            score += 1.0
        else:
            ratio = abs(mg_a - mg_b) / mg_b
            if ratio <= 0.01:
                score += 1.0
            elif ratio <= 0.20:
                score += 0.5
            else:
                score += 0.2
    return score / total if total > 0 else 0.0


def score_path_a(candidate: MedicineNode, baseline: MedicineNode,
                 max_price: float) -> float:
    """
    Path A relative score vs a baseline medicine.
    Score = (CompMatch×0.40) + (Eff×0.30) - (PricePenalty×0.20) + (Avail×0.10)
    """
    comp = _composition_match(candidate, baseline)
    eff = candidate.effectiveness_score
    pen = abs(candidate.price - baseline.price) / max_price if max_price > 0 else 0.0
    avail = 1.0 if candidate.availability else 0.0
    return max(0.0, comp * 0.40 + eff * 0.30 - pen * 0.20 + avail * 0.10)


def score_path_b(candidate: MedicineNode, max_price: float) -> float:
    """
    Path B absolute score (no baseline).
    Score = (Eff×0.50) + (Avail×0.30) - (NormPrice×0.20)
    """
    eff = candidate.effectiveness_score
    avail = 1.0 if candidate.availability else 0.0
    norm_price = candidate.price / max_price if max_price > 0 else 0.0
    return max(0.0, eff * 0.50 + avail * 0.30 - norm_price * 0.20)


# ══════════════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════

class RecommendationEngine:
    """
    Central engine tying together:
      BTree      — primary disease index
      AVLTree    — medicine store per disease (inside each BTree node)
      SplayTree  — MRU cache for recently queried diseases
      FibHeap    — priority queue for ranked recommendations
    """

    def __init__(self, t: int = 3):
        self.db = BTree(t=t)
        self.mru_cache = SplayTree()
        self._all_medicines: list = []
        self._disease_set: set = set()
        self._medicine_keys: set = set()   # (name, disease) dedup guard

    # ── Data loading ──────────────────────────────────────────────────

    def load_csv(self, filepath: str):
        """Load medicines CSV → B-Tree/AVL structures. O(n log n)."""
        import recommendation_engine as _self_module
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        with open(filepath, newline='', encoding='utf-8') as f:
            count = 0
            for row in csv.DictReader(f):
                try:
                    self._insert_medicine(self._parse_row(row))
                    count += 1
                except Exception as e:
                    print(f"[WARN] Skipping row: {e}")
        _self_module._ENGINE_DISEASE_SET = self._disease_set
        print(f"[INFO] Loaded {count} medicines across {len(self._disease_set)} diseases.")

    def _parse_row(self, row: dict) -> MedicineNode:
        import ast
        comp = ast.literal_eval(row["composition"])
        suit = ast.literal_eval(row["suitable_for"])
        avail = row["availability"].strip().lower() in ("true", "1", "yes")
        return MedicineNode(
            name=row["name"].strip(),
            disease_target=row["disease_target"].strip(),
            composition=comp,
            suitable_for=suit,
            price=float(row["price"]),
            effectiveness_score=float(row["effectiveness_score"]),
            availability=avail,
        )

    def _insert_medicine(self, med: MedicineNode):
        self.db.insert_disease(med.disease_target, med)
        self._disease_set.add(med.disease_target)
        key = (med.name, med.disease_target)
        if key not in self._medicine_keys:
            self._medicine_keys.add(key)
            self._all_medicines.append(med)

    # ── Path A: Direct alternative finder ────────────────────────────

    def find_alternatives(self, disease: str, baseline_name: str,
                          age_group: str, top_n: int = 5) -> list:
        """
        Find top-N alternatives to baseline_name for disease + age_group.
        Steps: BTree lookup → AVL age filter → score_path_a → FibHeap rank.
        """
        avl = self._get_avl(disease)
        if avl is None:
            return []
        # Find baseline (case-insensitive fallback)
        baseline = avl.search(baseline_name)
        if baseline is None:
            for m in self._all_medicines:
                if m.name.lower() == baseline_name.lower() and m.disease_target == disease:
                    baseline = m
                    break
        if baseline is None:
            return []

        candidates = [c for c in avl.filter_by_age_group(age_group)
                      if c.name != baseline_name]
        if not candidates:
            return []

        max_price = max(c.price for c in candidates) or 1.0
        heap = FibonacciHeap()
        for med in candidates:
            heap.insert_max(score_path_a(med, baseline, max_price), med)

        results = []
        for _ in range(min(top_n, heap.size())):
            r = heap.extract_max()
            if r:
                results.append({"medicine": r[0], "score": round(r[1], 4)})
        return results

    # ── Path B: Symptom-based predictor ──────────────────────────────

    def predict_from_symptoms(self, symptom_string: str,
                              age_group: str, top_n: int = 5) -> dict:
        """
        Map symptoms → disease → rank medicines with score_path_b.
        """
        disease = symptoms_to_disease(symptom_string)
        avl = self._get_avl(disease)
        if avl is None:
            disease = self._closest_disease(disease)
            avl = self._get_avl(disease)
        if avl is None:
            return {"inferred_disease": disease, "recommendations": []}

        candidates = avl.filter_by_age_group(age_group)
        if not candidates:
            return {"inferred_disease": disease, "recommendations": []}

        max_price = max(c.price for c in candidates) or 1.0
        heap = FibonacciHeap()
        for med in candidates:
            heap.insert_max(score_path_b(med, max_price), med)

        results = []
        for _ in range(min(top_n, heap.size())):
            r = heap.extract_max()
            if r:
                results.append({"medicine": r[0], "score": round(r[1], 4)})
        return {"inferred_disease": disease, "recommendations": results}

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_avl(self, disease: str):
        self.mru_cache.access(disease)
        return self.db.search(disease)

    def _closest_disease(self, target: str) -> str:
        tl = target.lower()
        for d in self._disease_set:
            if tl in d.lower() or d.lower() in tl:
                return d
        return next(iter(self._disease_set), "Seasonal Flu")

    def get_mru_diseases(self, k: int = 5):
        return self.mru_cache.get_top_k(k)

    def get_all_diseases(self):
        return sorted(self._disease_set)

    def get_medicines_for_disease(self, disease: str):
        avl = self.db.search(disease)
        if avl is None:
            return []
        return [m.name for m in avl.get_all_medicines()]
