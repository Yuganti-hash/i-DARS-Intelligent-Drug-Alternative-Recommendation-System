"""
generate_dataset.py — Mock Medicine Dataset Generator

Generates a CSV with 2000+ realistic medicine rows.
Each row adheres to the MedicineNode schema:
  name, disease_target, composition, suitable_for, price, effectiveness_score, availability

BUG FIX: mg values with small ranges (e.g. 1-2mg) were being rounded to 0
because the old logic did `round(x / 5) * 5`. Fixed: rounding is now
adaptive. A hard floor of 0.5mg ensures no ingredient ever shows as 0mg.

Run:
    python generate_dataset.py
Output:
    medicines_dataset.csv
"""

import csv
import random

random.seed(42)


def _smart_round_mg(value: float) -> float:
    """
    Round a mg value to a clinically sensible precision.

    Tiers (inclusive upper bound triggers next tier):
      < 0.01 mg  -> round to 4 decimal places  (e.g. Latanoprost 0.005 mg)
      < 1 mg     -> round to 3 decimal places  (e.g. Digoxin 0.125 mg)
      < 20 mg    -> round to nearest 0.5 mg    (e.g. Clonazepam 1.5 mg)
      < 100 mg   -> round to nearest 1 mg      (e.g. Omeprazole 40 mg)
      >= 100 mg  -> round to nearest 5 mg      (e.g. Paracetamol 500 mg)

    NO artificial floor: real drugs like Digoxin (0.0625 mg),
    Latanoprost (0.005 mg), Scopolamine (1 mg) must be preserved exactly.
    The only hard rule: result must be > 0.
    """
    if value <= 0:
        return 0.001  # shouldn't happen, but guard against degenerate input
    if value < 0.01:
        return round(value, 4)
    if value < 1.0:
        return round(value, 3)
    if value < 20.0:
        return round(value * 2) / 2   # nearest 0.5
    if value < 100.0:
        return round(value)           # nearest 1
    return round(value / 5) * 5      # nearest 5


DISEASE_MEDICINE_MAP = {
    "Seasonal Flu": [
        ("Paraflu",          [{"ingredient": "Paracetamol",        "mg_range": (400, 650)},
                               {"ingredient": "Chlorpheniramine",  "mg_range": (2,   4  )}]),
        ("Flucold",          [{"ingredient": "Paracetamol",        "mg_range": (500, 650)},
                               {"ingredient": "Phenylephrine",     "mg_range": (5,   10 )}]),
        ("Sinarest",         [{"ingredient": "Paracetamol",        "mg_range": (500, 500)},
                               {"ingredient": "Cetirizine",        "mg_range": (5,   10 )},
                               {"ingredient": "Phenylephrine",     "mg_range": (5,   10 )}]),
        ("D-Cold",           [{"ingredient": "Paracetamol",        "mg_range": (325, 500)}]),
        ("Febrex",           [{"ingredient": "Paracetamol",        "mg_range": (500, 650)},
                               {"ingredient": "Ibuprofen",         "mg_range": (200, 400)}]),
        ("Vicks Action 500", [{"ingredient": "Paracetamol",        "mg_range": (500, 500)},
                               {"ingredient": "Caffeine",          "mg_range": (30,  60 )}]),
        ("Grilinctus",       [{"ingredient": "Bromhexine",         "mg_range": (4,   8  )},
                               {"ingredient": "Guaifenesin",       "mg_range": (50,  100)}]),
        ("Nasivion",         [{"ingredient": "Oxymetazoline",      "mg_range": (0.5, 0.5)}]),
    ],
    "Common Cold": [
        ("Benadryl",   [{"ingredient": "Diphenhydramine",  "mg_range": (25,  50 )}]),
        ("Cetrizet",   [{"ingredient": "Cetirizine",       "mg_range": (5,   10 )}]),
        ("Zyrtec",     [{"ingredient": "Cetirizine",       "mg_range": (10,  10 )}]),
        ("Allegra",    [{"ingredient": "Fexofenadine",     "mg_range": (120, 180)}]),
        ("Avil",       [{"ingredient": "Pheniramine",      "mg_range": (22,  45 )}]),
        ("Piriton",    [{"ingredient": "Chlorpheniramine", "mg_range": (4,   8  )}]),
        ("Actifed",    [{"ingredient": "Triprolidine",     "mg_range": (2.5, 2.5)},
                        {"ingredient": "Pseudoephedrine",  "mg_range": (60,  60 )}]),
    ],
    "Bronchitis": [
        ("Amoxicillin",    [{"ingredient": "Amoxicillin Trihydrate",  "mg_range": (250, 875)}]),
        ("Azithromycin",   [{"ingredient": "Azithromycin Dihydrate",  "mg_range": (250, 500)}]),
        ("Clarithromycin", [{"ingredient": "Clarithromycin",          "mg_range": (250, 500)}]),
        ("Doxycycline",    [{"ingredient": "Doxycycline Hyclate",     "mg_range": (100, 200)}]),
        ("Salbutamol",     [{"ingredient": "Salbutamol Sulphate",     "mg_range": (2,   4  )}]),
        ("Guaifenesin",    [{"ingredient": "Guaifenesin",             "mg_range": (100, 400)}]),
        ("Levofloxacin",   [{"ingredient": "Levofloxacin",            "mg_range": (250, 750)}]),
    ],
    "Pneumonia": [
        ("Amoxiclav",   [{"ingredient": "Amoxicillin",      "mg_range": (500,1000)},
                         {"ingredient": "Clavulanic Acid",  "mg_range": (125, 125)}]),
        ("Ceftriaxone", [{"ingredient": "Ceftriaxone Sodium","mg_range":(250,2000)}]),
        ("Levofloxacin",[{"ingredient": "Levofloxacin",     "mg_range": (500, 750)}]),
        ("Azithromycin",[{"ingredient": "Azithromycin",     "mg_range": (500, 500)}]),
        ("Meropenem",   [{"ingredient": "Meropenem",        "mg_range": (500,1000)}]),
        ("Piperacillin",[{"ingredient": "Piperacillin",     "mg_range": (2000,4000)},
                         {"ingredient": "Tazobactam",       "mg_range": (250, 500)}]),
    ],
    "Tuberculosis": [
        ("Rifampicin",   [{"ingredient": "Rifampicin",   "mg_range": (150, 600)}]),
        ("Isoniazid",    [{"ingredient": "Isoniazid",    "mg_range": (100, 300)}]),
        ("Pyrazinamide", [{"ingredient": "Pyrazinamide", "mg_range": (500,2000)}]),
        ("Ethambutol",   [{"ingredient": "Ethambutol HCl","mg_range":(100,1200)}]),
        ("RHEZ Combo",   [{"ingredient": "Rifampicin",   "mg_range": (150, 150)},
                          {"ingredient": "Isoniazid",   "mg_range": (75,  75 )},
                          {"ingredient": "Ethambutol",  "mg_range": (275, 275)},
                          {"ingredient": "Pyrazinamide","mg_range": (400, 400)}]),
    ],
    "Malaria": [
        ("Chloroquine",  [{"ingredient": "Chloroquine Phosphate",  "mg_range": (150, 500)}]),
        ("Artemether",   [{"ingredient": "Artemether",             "mg_range": (20,  80 )},
                          {"ingredient": "Lumefantrine",           "mg_range": (120, 480)}]),
        ("Quinine",      [{"ingredient": "Quinine Sulphate",       "mg_range": (300, 600)}]),
        ("Primaquine",   [{"ingredient": "Primaquine Phosphate",   "mg_range": (7.5, 15 )}]),
        ("Doxycycline",  [{"ingredient": "Doxycycline Hyclate",    "mg_range": (100, 200)}]),
    ],
    "Dengue Fever": [
        ("Paracetamol",    [{"ingredient": "Paracetamol",           "mg_range": (500,1000)}]),
        ("ORS Dengue",     [{"ingredient": "Sodium Chloride",       "mg_range": (520, 520)},
                            {"ingredient": "Glucose",               "mg_range": (2700,2700)}]),
        ("Papaya Leaf Ext",[{"ingredient": "Carica Papaya Extract", "mg_range": (1100,1100)}]),
    ],
    "Hypertension": [
        ("Amlodipine",     [{"ingredient": "Amlodipine",            "mg_range": (2.5, 10  )}]),
        ("Telma",          [{"ingredient": "Telmisartan",           "mg_range": (20,  80  )}]),
        ("Losartan",       [{"ingredient": "Losartan",              "mg_range": (25,  100 )}]),
        ("Atenolol",       [{"ingredient": "Atenolol",              "mg_range": (25,  100 )}]),
        ("Metolar",        [{"ingredient": "Metoprolol",            "mg_range": (25,  100 )}]),
        ("Ramipril",       [{"ingredient": "Ramipril",              "mg_range": (1.25,10  )}]),
        ("Lisinopril",     [{"ingredient": "Lisinopril",            "mg_range": (5,   40  )}]),
        ("Candesartan",    [{"ingredient": "Candesartan",           "mg_range": (4,   32  )}]),
        ("Nifedipine",     [{"ingredient": "Nifedipine",            "mg_range": (10,  60  )}]),
        ("Inderal",        [{"ingredient": "Propranolol",           "mg_range": (10,  80  )}]),
        ("Chlorthalidone", [{"ingredient": "Chlorthalidone",        "mg_range": (12.5,25  )}]),
        ("Hydrochlorothiazide",[{"ingredient": "Hydrochlorothiazide","mg_range":(12.5,50  )}]),
    ],
    "Angina": [
        ("Nitroglycerine",  [{"ingredient": "Glyceryl Trinitrate",  "mg_range": (0.5, 0.6)}]),
        ("Isosorbide Mono", [{"ingredient": "Isosorbide Mononitrate","mg_range":(10,  60 )}]),
        ("Isosorbide Di",   [{"ingredient": "Isosorbide Dinitrate", "mg_range": (5,   40 )}]),
        ("Amlodipine",      [{"ingredient": "Amlodipine Besylate",  "mg_range": (5,   10 )}]),
        ("Ranolazine",      [{"ingredient": "Ranolazine",           "mg_range": (500,1000)}]),
        ("Ivabradine",      [{"ingredient": "Ivabradine HCl",       "mg_range": (5,   7.5)}]),
        ("Diltiazem",       [{"ingredient": "Diltiazem HCl",        "mg_range": (30,  360)}]),
    ],
    "Arrhythmia": [
        ("Amiodarone",  [{"ingredient": "Amiodarone HCl",     "mg_range": (100, 400)}]),
        ("Digoxin",     [{"ingredient": "Digoxin",            "mg_range": (0.0625,0.25)}]),
        ("Flecainide",  [{"ingredient": "Flecainide Acetate", "mg_range": (50,  200)}]),
        ("Sotalol",     [{"ingredient": "Sotalol HCl",        "mg_range": (40,  320)}]),
        ("Verapamil",   [{"ingredient": "Verapamil HCl",      "mg_range": (40,  240)}]),
        ("Propafenone", [{"ingredient": "Propafenone HCl",    "mg_range": (150, 300)}]),
    ],
    "Heart Failure": [
        ("Furosemide",   [{"ingredient": "Furosemide",         "mg_range": (20,  80  )}]),
        ("Spironolactone",[{"ingredient": "Spironolactone",    "mg_range": (12.5,50  )}]),
        ("Sacubitril",   [{"ingredient": "Sacubitril",         "mg_range": (24,  97  )},
                          {"ingredient": "Valsartan",          "mg_range": (26,  103 )}]),
        ("Bisoprolol",   [{"ingredient": "Bisoprolol Fumarate","mg_range": (1.25,10  )}]),
        ("Digoxin",      [{"ingredient": "Digoxin",            "mg_range": (0.0625,0.25)}]),
        ("Eplerenone",   [{"ingredient": "Eplerenone",         "mg_range": (25,  50  )}]),
    ],
    "Hyperlipidemia": [
        ("Atorvastatin", [{"ingredient": "Atorvastatin Calcium","mg_range":(10,  80  )}]),
        ("Rosuvastatin", [{"ingredient": "Rosuvastatin Calcium","mg_range":(5,   40  )}]),
        ("Simvastatin",  [{"ingredient": "Simvastatin",         "mg_range": (10,  80  )}]),
        ("Ezetimibe",    [{"ingredient": "Ezetimibe",           "mg_range": (10,  10  )}]),
        ("Fenofibrate",  [{"ingredient": "Fenofibrate",         "mg_range": (67,  200 )}]),
        ("Evolocumab",   [{"ingredient": "Evolocumab",          "mg_range": (140, 420 )}]),
        ("Niacin SR",    [{"ingredient": "Niacin",              "mg_range": (500,2000 )}]),
    ],
    "Type 2 Diabetes": [
        ("Metformin",    [{"ingredient": "Metformin HCl",       "mg_range": (500,1000)}]),
        ("Glipizide",    [{"ingredient": "Glipizide",           "mg_range": (5,   20 )}]),
        ("Januvia",      [{"ingredient": "Sitagliptin",         "mg_range": (25,  100)}]),
        ("Jardiance",    [{"ingredient": "Empagliflozin",       "mg_range": (10,  25 )}]),
        ("Glimepride",   [{"ingredient": "Glimepiride",         "mg_range": (1,   4  )}]),
        ("Vildagliptin", [{"ingredient": "Vildagliptin",        "mg_range": (50,  100)}]),
        ("Pioglitazone", [{"ingredient": "Pioglitazone",        "mg_range": (15,  45 )}]),
        ("Dapagliflozin",[{"ingredient": "Dapagliflozin",       "mg_range": (5,   10 )}]),
        ("Acarbose",     [{"ingredient": "Acarbose",            "mg_range": (25,  100)}]),
        ("Linagliptin",  [{"ingredient": "Linagliptin",         "mg_range": (5,   5  )}]),
    ],
    "Type 1 Diabetes": [
        ("Insulin Glargine",[{"ingredient": "Insulin Glargine", "mg_range": (100, 100)}]),
        ("Insulin Aspart",  [{"ingredient": "Insulin Aspart",   "mg_range": (100, 100)}]),
        ("Insulin Lispro",  [{"ingredient": "Insulin Lispro",   "mg_range": (100, 100)}]),
        ("Insulin NPH",     [{"ingredient": "Isophane Insulin", "mg_range": (100, 100)}]),
        ("Insulin Detemir", [{"ingredient": "Insulin Detemir",  "mg_range": (100, 100)}]),
    ],
    "Hypothyroidism": [
        ("Levothyroxine",[{"ingredient": "Levothyroxine Sodium","mg_range": (25,  200)}]),
        ("Thyronorm",    [{"ingredient": "Levothyroxine Sodium","mg_range": (25,  150)}]),
        ("Eltroxin",     [{"ingredient": "Levothyroxine Sodium","mg_range": (25,  100)}]),
        ("Liothyronine", [{"ingredient": "Liothyronine Sodium", "mg_range": (5,   25 )}]),
    ],
    "Hyperthyroidism": [
        ("Methimazole",     [{"ingredient": "Methimazole",       "mg_range": (5,   30 )}]),
        ("Propylthiouracil",[{"ingredient": "Propylthiouracil", "mg_range": (50,  300)}]),
        ("Carbimazole",     [{"ingredient": "Carbimazole",       "mg_range": (5,   20 )}]),
        ("Propranolol",     [{"ingredient": "Propranolol HCl",   "mg_range": (10,  80 )}]),
    ],
    "Obesity": [
        ("Orlistat",     [{"ingredient": "Orlistat",            "mg_range": (60,  120)}]),
        ("Phentermine",  [{"ingredient": "Phentermine HCl",     "mg_range": (15,  37.5)}]),
        ("Naltrexone SR",[{"ingredient": "Naltrexone",          "mg_range": (8,   32 )},
                          {"ingredient": "Bupropion",           "mg_range": (90,  360)}]),
        ("Semaglutide",  [{"ingredient": "Semaglutide",         "mg_range": (0.25,2.4)}]),
        ("Liraglutide",  [{"ingredient": "Liraglutide",         "mg_range": (0.6, 3  )}]),
    ],
    "Migraine": [
        ("Sumatriptan",   [{"ingredient": "Sumatriptan",              "mg_range": (25,  100)}]),
        ("Rizact",        [{"ingredient": "Rizatriptan",              "mg_range": (5,   10 )}]),
        ("Suminat",       [{"ingredient": "Sumatriptan Succinate",    "mg_range": (50,  100)}]),
        ("Cafergot",      [{"ingredient": "Ergotamine",               "mg_range": (1,   2  )},
                           {"ingredient": "Caffeine",                 "mg_range": (100, 100)}]),
        ("Topamax",       [{"ingredient": "Topiramate",               "mg_range": (25,  100)}]),
        ("Propanorm",     [{"ingredient": "Propranolol",              "mg_range": (40,  80 )}]),
        ("Amitriptyline", [{"ingredient": "Amitriptyline",            "mg_range": (10,  25 )}]),
        ("Migranil",      [{"ingredient": "Paracetamol",              "mg_range": (500, 650)},
                           {"ingredient": "Caffeine",                 "mg_range": (40,  65 )}]),
        ("Naratriptan",   [{"ingredient": "Naratriptan HCl",          "mg_range": (1,   2.5)}]),
        ("Eletriptan",    [{"ingredient": "Eletriptan Hydrobromide",  "mg_range": (20,  40 )}]),
    ],
    "Epilepsy": [
        ("Levetiracetam", [{"ingredient": "Levetiracetam",     "mg_range": (250,1500)}]),
        ("Valproate",     [{"ingredient": "Sodium Valproate",  "mg_range": (200, 500)}]),
        ("Lamotrigine",   [{"ingredient": "Lamotrigine",       "mg_range": (25,  400)}]),
        ("Carbamazepine", [{"ingredient": "Carbamazepine",     "mg_range": (100, 400)}]),
        ("Phenytoin",     [{"ingredient": "Phenytoin Sodium",  "mg_range": (50,  300)}]),
        ("Oxcarbazepine", [{"ingredient": "Oxcarbazepine",     "mg_range": (150, 600)}]),
        ("Topiramate",    [{"ingredient": "Topiramate",        "mg_range": (25,  200)}]),
        ("Clonazepam",    [{"ingredient": "Clonazepam",        "mg_range": (0.5, 2  )}]),
    ],
    "Parkinson's Disease": [
        ("Levodopa",    [{"ingredient": "Levodopa",          "mg_range": (100, 250)},
                         {"ingredient": "Carbidopa",         "mg_range": (25,  25 )}]),
        ("Pramipexole", [{"ingredient": "Pramipexole HCl",   "mg_range": (0.125,1.5)}]),
        ("Ropinirole",  [{"ingredient": "Ropinirole HCl",    "mg_range": (0.25,8  )}]),
        ("Rasagiline",  [{"ingredient": "Rasagiline Mesylate","mg_range":(0.5, 1  )}]),
        ("Entacapone",  [{"ingredient": "Entacapone",        "mg_range": (200, 200)}]),
        ("Amantadine",  [{"ingredient": "Amantadine HCl",    "mg_range": (100, 200)}]),
    ],
    "Alzheimer's Disease": [
        ("Donepezil",   [{"ingredient": "Donepezil HCl",          "mg_range": (5,   10 )}]),
        ("Rivastigmine",[{"ingredient": "Rivastigmine Tartrate",   "mg_range": (1.5, 6  )}]),
        ("Galantamine", [{"ingredient": "Galantamine HBr",         "mg_range": (4,   24 )}]),
        ("Memantine",   [{"ingredient": "Memantine HCl",           "mg_range": (5,   20 )}]),
    ],
    "Vertigo": [
        ("Betahistine",     [{"ingredient": "Betahistine Dihydrochloride","mg_range":(8, 24)}]),
        ("Meclizine",       [{"ingredient": "Meclizine HCl",       "mg_range": (12.5,50 )}]),
        ("Dimenhydrinate",  [{"ingredient": "Dimenhydrinate",       "mg_range": (25,  50 )}]),
        ("Prochlorperazine",[{"ingredient": "Prochlorperazine Maleate","mg_range":(5, 10 )}]),
        ("Scopolamine",     [{"ingredient": "Scopolamine",          "mg_range": (1,   1  )}]),
        ("Flunarizine",     [{"ingredient": "Flunarizine HCl",      "mg_range": (5,   10 )}]),
    ],
    "Multiple Sclerosis": [
        ("Interferon Beta",[{"ingredient": "Interferon Beta-1a",   "mg_range": (22,  44 )}]),
        ("Glatiramer",     [{"ingredient": "Glatiramer Acetate",   "mg_range": (20,  40 )}]),
        ("Fingolimod",     [{"ingredient": "Fingolimod HCl",       "mg_range": (0.5, 0.5)}]),
        ("Dimethyl Fumarate",[{"ingredient": "Dimethyl Fumarate",  "mg_range": (120, 240)}]),
        ("Natalizumab",    [{"ingredient": "Natalizumab",          "mg_range": (300, 300)}]),
        ("Teriflunomide",  [{"ingredient": "Teriflunomide",        "mg_range": (7,   14 )}]),
    ],
    "Anxiety Disorder": [
        ("Alprazolam",   [{"ingredient": "Alprazolam",           "mg_range": (0.25,1  )}]),
        ("Buspirone",    [{"ingredient": "Buspirone HCl",         "mg_range": (5,   30 )}]),
        ("Sertraline",   [{"ingredient": "Sertraline HCl",        "mg_range": (25,  200)}]),
        ("Escitalopram", [{"ingredient": "Escitalopram Oxalate",  "mg_range": (5,   20 )}]),
        ("Clonazepam",   [{"ingredient": "Clonazepam",            "mg_range": (0.25,2  )}]),
        ("Venlafaxine",  [{"ingredient": "Venlafaxine HCl",       "mg_range": (37.5,225)}]),
        ("Paroxetine",   [{"ingredient": "Paroxetine HCl",        "mg_range": (10,  60 )}]),
        ("Duloxetine",   [{"ingredient": "Duloxetine HCl",        "mg_range": (30,  120)}]),
    ],
    "Major Depressive Disorder": [
        ("Fluoxetine",   [{"ingredient": "Fluoxetine HCl",        "mg_range": (10,  60 )}]),
        ("Sertraline",   [{"ingredient": "Sertraline HCl",        "mg_range": (25,  200)}]),
        ("Bupropion",    [{"ingredient": "Bupropion HCl",         "mg_range": (75,  300)}]),
        ("Mirtazapine",  [{"ingredient": "Mirtazapine",           "mg_range": (7.5, 45 )}]),
        ("Duloxetine",   [{"ingredient": "Duloxetine HCl",        "mg_range": (20,  120)}]),
        ("Amitriptyline",[{"ingredient": "Amitriptyline HCl",     "mg_range": (10,  150)}]),
        ("Vortioxetine", [{"ingredient": "Vortioxetine",          "mg_range": (5,   20 )}]),
        ("Agomelatine",  [{"ingredient": "Agomelatine",           "mg_range": (25,  50 )}]),
    ],
    "Bipolar Disorder": [
        ("Lithium",      [{"ingredient": "Lithium Carbonate",     "mg_range": (150, 600)}]),
        ("Valproate",    [{"ingredient": "Sodium Valproate",      "mg_range": (200, 800)}]),
        ("Quetiapine",   [{"ingredient": "Quetiapine Fumarate",   "mg_range": (25,  800)}]),
        ("Lamotrigine",  [{"ingredient": "Lamotrigine",           "mg_range": (25,  400)}]),
        ("Olanzapine",   [{"ingredient": "Olanzapine",            "mg_range": (2.5, 20 )}]),
        ("Risperidone",  [{"ingredient": "Risperidone",           "mg_range": (0.5, 6  )}]),
        ("Aripiprazole", [{"ingredient": "Aripiprazole",          "mg_range": (2,   30 )}]),
    ],
    "Schizophrenia": [
        ("Risperidone",  [{"ingredient": "Risperidone",           "mg_range": (0.5, 6  )}]),
        ("Olanzapine",   [{"ingredient": "Olanzapine",            "mg_range": (5,   20 )}]),
        ("Clozapine",    [{"ingredient": "Clozapine",             "mg_range": (12.5,600)}]),
        ("Haloperidol",  [{"ingredient": "Haloperidol",           "mg_range": (0.5, 20 )}]),
        ("Quetiapine",   [{"ingredient": "Quetiapine Fumarate",   "mg_range": (50,  800)}]),
        ("Ziprasidone",  [{"ingredient": "Ziprasidone HCl",       "mg_range": (20,  80 )}]),
        ("Paliperidone", [{"ingredient": "Paliperidone",          "mg_range": (3,   12 )}]),
    ],
    "ADHD": [
        ("Methylphenidate",[{"ingredient": "Methylphenidate HCl", "mg_range": (5,   60 )}]),
        ("Amphetamine",    [{"ingredient": "Mixed Amphetamine Salts","mg_range":(5,  30 )}]),
        ("Atomoxetine",    [{"ingredient": "Atomoxetine HCl",     "mg_range": (10,  100)}]),
        ("Guanfacine",     [{"ingredient": "Guanfacine HCl",      "mg_range": (1,   4  )}]),
        ("Clonidine",      [{"ingredient": "Clonidine HCl",       "mg_range": (0.1, 0.4)}]),
        ("Lisdexamfetamine",[{"ingredient": "Lisdexamfetamine",   "mg_range": (20,  70 )}]),
    ],
    "PTSD": [
        ("Sertraline",  [{"ingredient": "Sertraline HCl",  "mg_range": (25,  200)}]),
        ("Paroxetine",  [{"ingredient": "Paroxetine HCl",  "mg_range": (20,  60 )}]),
        ("Prazosin",    [{"ingredient": "Prazosin HCl",    "mg_range": (1,   15 )}]),
        ("Venlafaxine", [{"ingredient": "Venlafaxine HCl", "mg_range": (75,  225)}]),
        ("Mirtazapine", [{"ingredient": "Mirtazapine",     "mg_range": (15,  45 )}]),
    ],
    "Insomnia": [
        ("Zolpidem",        [{"ingredient": "Zolpidem Tartrate",  "mg_range": (5,   10 )}]),
        ("Melatonin",       [{"ingredient": "Melatonin",          "mg_range": (1,   10 )}]),
        ("Eszopiclone",     [{"ingredient": "Eszopiclone",        "mg_range": (1,   3  )}]),
        ("Diphenhydramine", [{"ingredient": "Diphenhydramine HCl","mg_range": (25,  50 )}]),
        ("Doxepin",         [{"ingredient": "Doxepin HCl",        "mg_range": (3,   6  )}]),
        ("Ramelteon",       [{"ingredient": "Ramelteon",          "mg_range": (8,   8  )}]),
        ("Suvorexant",      [{"ingredient": "Suvorexant",         "mg_range": (5,   20 )}]),
    ],
    "Gastritis": [
        ("Omeprazole",  [{"ingredient": "Omeprazole",         "mg_range": (10,  40 )}]),
        ("Pantoprazole",[{"ingredient": "Pantoprazole Sodium","mg_range": (20,  40 )}]),
        ("Rabeprazole", [{"ingredient": "Rabeprazole",         "mg_range": (10,  20 )}]),
        ("Ranitidine",  [{"ingredient": "Ranitidine",          "mg_range": (75,  300)}]),
        ("Sucralfate",  [{"ingredient": "Sucralfate",          "mg_range": (500,1000)}]),
        ("Famotidine",  [{"ingredient": "Famotidine",          "mg_range": (20,  40 )}]),
        ("Misoprostol", [{"ingredient": "Misoprostol",         "mg_range": (0.1, 0.2)}]),
    ],
    "GERD": [
        ("Nexium",         [{"ingredient": "Esomeprazole",          "mg_range": (20,  40 )}]),
        ("Dexlansoprazole",[{"ingredient": "Dexlansoprazole",        "mg_range": (30,  60 )}]),
        ("Metoclopramide", [{"ingredient": "Metoclopramide",         "mg_range": (5,   10 )}]),
        ("Domperidone",    [{"ingredient": "Domperidone",            "mg_range": (10,  20 )}]),
        ("Lansoprazole",   [{"ingredient": "Lansoprazole",           "mg_range": (15,  30 )}]),
        ("Vonoprazan",     [{"ingredient": "Vonoprazan Fumarate",    "mg_range": (10,  20 )}]),
    ],
    "Gastroenteritis": [
        ("ORS",         [{"ingredient": "Sodium Chloride",        "mg_range": (520, 520)},
                         {"ingredient": "Potassium Chloride",     "mg_range": (300, 300)},
                         {"ingredient": "Glucose",                "mg_range": (2700,2700)}]),
        ("Ondansetron", [{"ingredient": "Ondansetron HCl",        "mg_range": (4,   8  )}]),
        ("Loperamide",  [{"ingredient": "Loperamide HCl",         "mg_range": (2,   4  )}]),
        ("Probiotics",  [{"ingredient": "Lactobacillus acidophilus","mg_range":(1,  10 )}]),
        ("Metronidazole",[{"ingredient": "Metronidazole",         "mg_range": (200, 400)}]),
        ("Racecadotril",[{"ingredient": "Racecadotril",           "mg_range": (10,  100)}]),
    ],
    "Irritable Bowel Syndrome": [
        ("Mebeverine",  [{"ingredient": "Mebeverine HCl",  "mg_range": (135, 200)}]),
        ("Dicyclomine", [{"ingredient": "Dicyclomine HCl", "mg_range": (10,  20 )}]),
        ("Rifaximin",   [{"ingredient": "Rifaximin",        "mg_range": (200, 550)}]),
        ("Linaclotide", [{"ingredient": "Linaclotide",      "mg_range": (72,  290)}]),
        ("Psyllium",    [{"ingredient": "Ispaghula Husk",   "mg_range": (3500,7000)}]),
        ("Lubiprostone",[{"ingredient": "Lubiprostone",     "mg_range": (8,   24 )}]),
    ],
    "Peptic Ulcer": [
        ("Omeprazole",  [{"ingredient": "Omeprazole",    "mg_range": (20,  40 )}]),
        ("Triple Therapy",[{"ingredient": "Amoxicillin", "mg_range": (500,1000)},
                           {"ingredient": "Clarithromycin","mg_range":(500, 500)},
                           {"ingredient": "Omeprazole",  "mg_range": (20,  40 )}]),
        ("Bismuth",     [{"ingredient": "Bismuth Subcitrate","mg_range":(120, 120)}]),
        ("Sucralfate",  [{"ingredient": "Sucralfate",    "mg_range": (500,1000)}]),
    ],
    "Crohn's Disease": [
        ("Mesalamine",  [{"ingredient": "Mesalamine",   "mg_range": (400,4800)}]),
        ("Prednisolone",[{"ingredient": "Prednisolone", "mg_range": (5,   60 )}]),
        ("Azathioprine",[{"ingredient": "Azathioprine", "mg_range": (25,  150)}]),
        ("Infliximab",  [{"ingredient": "Infliximab",   "mg_range": (100, 500)}]),
        ("Adalimumab",  [{"ingredient": "Adalimumab",   "mg_range": (40,  160)}]),
        ("Budesonide",  [{"ingredient": "Budesonide",   "mg_range": (3,   9  )}]),
    ],
    "Arthritis": [
        ("Diclofenac",     [{"ingredient": "Diclofenac Sodium",    "mg_range": (50,  100)}]),
        ("Celecoxib",      [{"ingredient": "Celecoxib",             "mg_range": (100, 200)}]),
        ("Methotrexate",   [{"ingredient": "Methotrexate",          "mg_range": (2.5, 25 )}]),
        ("Hydroxychloroquine",[{"ingredient":"Hydroxychloroquine",  "mg_range": (200, 400)}]),
        ("Prednisolone",   [{"ingredient": "Prednisolone",          "mg_range": (5,   40 )}]),
        ("Ibuprofen",      [{"ingredient": "Ibuprofen",             "mg_range": (200, 800)}]),
        ("Naproxen",       [{"ingredient": "Naproxen",              "mg_range": (250, 500)}]),
        ("Sulfasalazine",  [{"ingredient": "Sulfasalazine",         "mg_range": (500,1000)}]),
        ("Etanercept",     [{"ingredient": "Etanercept",            "mg_range": (25,  50 )}]),
    ],
    "Osteoporosis": [
        ("Alendronate", [{"ingredient": "Alendronate Sodium",    "mg_range": (5,   70 )}]),
        ("Risedronate", [{"ingredient": "Risedronate Sodium",    "mg_range": (5,   35 )}]),
        ("Calcium+VitD",[{"ingredient": "Calcium Carbonate",     "mg_range": (500,1250)},
                         {"ingredient": "Cholecalciferol",       "mg_range": (200, 400)}]),
        ("Denosumab",   [{"ingredient": "Denosumab",             "mg_range": (60,  60 )}]),
        ("Teriparatide",[{"ingredient": "Teriparatide",          "mg_range": (20,  20 )}]),
    ],
    "Fibromyalgia": [
        ("Pregabalin",   [{"ingredient": "Pregabalin",      "mg_range": (75,  300)}]),
        ("Duloxetine",   [{"ingredient": "Duloxetine HCl",  "mg_range": (30,  120)}]),
        ("Milnacipran",  [{"ingredient": "Milnacipran HCl", "mg_range": (25,  100)}]),
        ("Cyclobenzaprine",[{"ingredient":"Cyclobenzaprine HCl","mg_range":(5, 10 )}]),
        ("Gabapentin",   [{"ingredient": "Gabapentin",      "mg_range": (100, 900)}]),
        ("Tramadol",     [{"ingredient": "Tramadol HCl",    "mg_range": (50,  100)}]),
    ],
    "Lower Back Pain": [
        ("Muscle Relaxant",[{"ingredient": "Methocarbamol",          "mg_range": (500, 750)},
                            {"ingredient": "Ibuprofen",              "mg_range": (200, 400)}]),
        ("Cyclobenzaprine",[{"ingredient": "Cyclobenzaprine HCl",    "mg_range": (5,   10 )}]),
        ("Tizanidine",     [{"ingredient": "Tizanidine HCl",         "mg_range": (2,   8  )}]),
        ("Tramadol",       [{"ingredient": "Tramadol HCl",           "mg_range": (50,  100)}]),
        ("Naproxen",       [{"ingredient": "Naproxen Sodium",        "mg_range": (220, 550)}]),
        ("Diclofenac Gel", [{"ingredient": "Diclofenac Diethylamine","mg_range": (10,  10 )}]),
        ("Lidocaine Patch",[{"ingredient": "Lidocaine",              "mg_range": (50,  700)}]),
    ],
    "Gout": [
        ("Allopurinol",  [{"ingredient": "Allopurinol",   "mg_range": (100, 800)}]),
        ("Colchicine",   [{"ingredient": "Colchicine",    "mg_range": (0.5, 1.2)}]),
        ("Febuxostat",   [{"ingredient": "Febuxostat",    "mg_range": (40,  80 )}]),
        ("Probenecid",   [{"ingredient": "Probenecid",   "mg_range": (500,1000)}]),
        ("Indomethacin", [{"ingredient": "Indomethacin", "mg_range": (25,  75 )}]),
    ],
    "Asthma": [
        ("Ventolin",    [{"ingredient": "Salbutamol",         "mg_range": (100, 200)}]),
        ("Foracort",    [{"ingredient": "Formoterol",         "mg_range": (6,   12 )},
                         {"ingredient": "Budesonide",         "mg_range": (100, 400)}]),
        ("Seretide",    [{"ingredient": "Salmeterol",         "mg_range": (25,  50 )},
                         {"ingredient": "Fluticasone",        "mg_range": (100, 500)}]),
        ("Montelukast", [{"ingredient": "Montelukast Sodium", "mg_range": (4,   10 )}]),
        ("Theo-Asthalin",[{"ingredient":"Theophylline",       "mg_range": (100, 400)},
                          {"ingredient":"Etofylline",         "mg_range": (50,  100)}]),
        ("Ipratropium", [{"ingredient": "Ipratropium Bromide","mg_range": (17,  34 )}]),
        ("Beclate",     [{"ingredient": "Beclomethasone",     "mg_range": (50,  250)}]),
        ("Tiotropium",  [{"ingredient": "Tiotropium Bromide", "mg_range": (9,   18 )}]),
    ],
    "COPD": [
        ("Tiotropium",   [{"ingredient": "Tiotropium Bromide",   "mg_range": (9,   18 )}]),
        ("Indacaterol",  [{"ingredient": "Indacaterol",          "mg_range": (75,  150)}]),
        ("Umeclidinium", [{"ingredient": "Umeclidinium Bromide", "mg_range": (62.5,62.5)}]),
        ("Roflumilast",  [{"ingredient": "Roflumilast",          "mg_range": (250, 500)}]),
        ("Salmeterol",   [{"ingredient": "Salmeterol Xinafoate", "mg_range": (25,  50 )}]),
        ("N-Acetylcysteine",[{"ingredient":"N-Acetylcysteine",   "mg_range": (200, 600)}]),
    ],
    "Allergic Rhinitis": [
        ("Fexofenadine",   [{"ingredient": "Fexofenadine HCl",      "mg_range": (60,  180)}]),
        ("Loratadine",     [{"ingredient": "Loratadine",             "mg_range": (10,  10 )}]),
        ("Desloratadine",  [{"ingredient": "Desloratadine",          "mg_range": (5,   5  )}]),
        ("Levocetirizine", [{"ingredient": "Levocetirizine HCl",     "mg_range": (2.5, 5  )}]),
        ("Azelastine",     [{"ingredient": "Azelastine HCl",         "mg_range": (137, 274)}]),
        ("Fluticasone Nasal",[{"ingredient":"Fluticasone Propionate","mg_range": (50,  200)}]),
        ("Budesonide Nasal",[{"ingredient":"Budesonide",             "mg_range": (32,  256)}]),
        ("Montelukast",    [{"ingredient": "Montelukast Sodium",     "mg_range": (4,   10 )}]),
    ],
    "Acne Vulgaris": [
        ("Benzoyl Peroxide",[{"ingredient": "Benzoyl Peroxide",     "mg_range": (25,  100)}]),
        ("Adapalene",       [{"ingredient": "Adapalene",            "mg_range": (1,   3  )}]),
        ("Tretinoin",       [{"ingredient": "Tretinoin",            "mg_range": (0.025,0.1)}]),
        ("Clindamycin",     [{"ingredient": "Clindamycin Phosphate","mg_range": (10,  10 )}]),
        ("Doxycycline",     [{"ingredient": "Doxycycline Hyclate",  "mg_range": (50,  100)}]),
        ("Isotretinoin",    [{"ingredient": "Isotretinoin",         "mg_range": (5,   40 )}]),
        ("Azithromycin",    [{"ingredient": "Azithromycin",         "mg_range": (250, 500)}]),
    ],
    "Eczema": [
        ("Hydrocortisone",[{"ingredient": "Hydrocortisone",         "mg_range": (5,   25 )}]),
        ("Tacrolimus",    [{"ingredient": "Tacrolimus",             "mg_range": (0.03,0.1)}]),
        ("Pimecrolimus",  [{"ingredient": "Pimecrolimus",           "mg_range": (10,  10 )}]),
        ("Betamethasone", [{"ingredient": "Betamethasone Valerate", "mg_range": (1,   5  )}]),
        ("Clobetasol",    [{"ingredient": "Clobetasol Propionate",  "mg_range": (0.5, 0.5)}]),
        ("Dupilumab",     [{"ingredient": "Dupilumab",              "mg_range": (200, 300)}]),
    ],
    "Psoriasis": [
        ("Methotrexate", [{"ingredient": "Methotrexate",   "mg_range": (2.5, 25 )}]),
        ("Cyclosporine", [{"ingredient": "Cyclosporine",   "mg_range": (25,  100)}]),
        ("Acitretin",    [{"ingredient": "Acitretin",      "mg_range": (10,  50 )}]),
        ("Secukinumab",  [{"ingredient": "Secukinumab",    "mg_range": (150, 300)}]),
        ("Adalimumab",   [{"ingredient": "Adalimumab",     "mg_range": (40,  80 )}]),
        ("Coal Tar",     [{"ingredient": "Coal Tar Solution","mg_range":(1,  10 )}]),
    ],
    "Fungal Infection": [
        ("Fluconazole",  [{"ingredient": "Fluconazole",   "mg_range": (50,  400)}]),
        ("Itraconazole", [{"ingredient": "Itraconazole",  "mg_range": (100, 200)}]),
        ("Terbinafine",  [{"ingredient": "Terbinafine HCl","mg_range":(125, 250)}]),
        ("Ketoconazole", [{"ingredient": "Ketoconazole",  "mg_range": (200, 200)}]),
        ("Clotrimazole", [{"ingredient": "Clotrimazole",  "mg_range": (10,  10 )}]),
        ("Voriconazole", [{"ingredient": "Voriconazole",  "mg_range": (50,  200)}]),
    ],
    "Urinary Tract Infection": [
        ("Nitrofurantoin",[{"ingredient": "Nitrofurantoin","mg_range": (50,  100)}]),
        ("Trimethoprim",  [{"ingredient": "Trimethoprim",  "mg_range": (100, 200)}]),
        ("Ciprofloxacin", [{"ingredient": "Ciprofloxacin HCl","mg_range":(250,500)}]),
        ("Fosfomycin",    [{"ingredient": "Fosfomycin Trometamol","mg_range":(3000,3000)}]),
        ("Cephalexin",    [{"ingredient": "Cephalexin",    "mg_range": (250, 500)}]),
        ("Augmentin",     [{"ingredient": "Amoxicillin",   "mg_range": (250, 875)},
                           {"ingredient": "Clavulanate",   "mg_range": (125, 125)}]),
    ],
    "Chronic Kidney Disease": [
        ("Erythropoietin",[{"ingredient": "Epoetin Alfa",        "mg_range": (2000,10000)}]),
        ("Calcium Acetate",[{"ingredient":"Calcium Acetate",     "mg_range": (667,1334)}]),
        ("Sevelamer",     [{"ingredient": "Sevelamer Carbonate", "mg_range": (800,1600)}]),
        ("Darbepoetin",   [{"ingredient": "Darbepoetin Alfa",    "mg_range": (25, 200 )}]),
        ("Bicarbonate",   [{"ingredient": "Sodium Bicarbonate",  "mg_range": (325, 650)}]),
    ],
    "Benign Prostatic Hyperplasia": [
        ("Tamsulosin",  [{"ingredient": "Tamsulosin HCl",    "mg_range": (0.4, 0.8)}]),
        ("Finasteride", [{"ingredient": "Finasteride",       "mg_range": (1,   5  )}]),
        ("Dutasteride", [{"ingredient": "Dutasteride",       "mg_range": (0.5, 0.5)}]),
        ("Doxazosin",   [{"ingredient": "Doxazosin Mesylate","mg_range": (1,   8  )}]),
        ("Silodosin",   [{"ingredient": "Silodosin",         "mg_range": (4,   8  )}]),
    ],
    "Cancer Pain Management": [
        ("Morphine",       [{"ingredient": "Morphine Sulphate","mg_range": (5,   200)}]),
        ("Oxycodone",      [{"ingredient": "Oxycodone HCl",    "mg_range": (5,   80 )}]),
        ("Fentanyl Patch", [{"ingredient": "Fentanyl",         "mg_range": (12.5,100)}]),
        ("Gabapentin",     [{"ingredient": "Gabapentin",       "mg_range": (100, 900)}]),
        ("Pregabalin",     [{"ingredient": "Pregabalin",       "mg_range": (75,  300)}]),
        ("Dexamethasone",  [{"ingredient": "Dexamethasone",    "mg_range": (0.5, 16 )}]),
    ],
    "Chemotherapy Nausea": [
        ("Ondansetron",  [{"ingredient": "Ondansetron HCl",   "mg_range": (4,   32 )}]),
        ("Granisetron",  [{"ingredient": "Granisetron HCl",   "mg_range": (1,   2  )}]),
        ("Aprepitant",   [{"ingredient": "Aprepitant",        "mg_range": (80,  125)}]),
        ("Dexamethasone",[{"ingredient": "Dexamethasone",     "mg_range": (8,   20 )}]),
        ("Metoclopramide",[{"ingredient":"Metoclopramide HCl","mg_range": (10,  40 )}]),
        ("Palonosetron", [{"ingredient": "Palonosetron HCl",  "mg_range": (0.25,0.5)}]),
    ],
    "Conjunctivitis": [
        ("Ciprofloxacin Eye",[{"ingredient": "Ciprofloxacin HCl","mg_range":(3, 3  )}]),
        ("Ofloxacin Eye",    [{"ingredient": "Ofloxacin",          "mg_range": (3, 3  )}]),
        ("Tobramycin Eye",   [{"ingredient": "Tobramycin",          "mg_range": (3, 3  )}]),
        ("Chloramphenicol",  [{"ingredient": "Chloramphenicol",     "mg_range": (5, 10 )}]),
        ("Cromolyn Eye",     [{"ingredient": "Cromolyn Sodium",     "mg_range": (40,40 )}]),
        ("Ketotifen Eye",    [{"ingredient": "Ketotifen Fumarate",  "mg_range": (0.25,0.5)}]),
    ],
    "Glaucoma": [
        ("Timolol Eye",  [{"ingredient": "Timolol Maleate",   "mg_range": (2.5, 5  )}]),
        ("Latanoprost",  [{"ingredient": "Latanoprost",       "mg_range": (0.005,0.005)}]),
        ("Brimonidine",  [{"ingredient": "Brimonidine Tartrate","mg_range":(1,  2  )}]),
        ("Dorzolamide",  [{"ingredient": "Dorzolamide HCl",   "mg_range": (20,  20 )}]),
        ("Bimatoprost",  [{"ingredient": "Bimatoprost",       "mg_range": (0.03,0.03)}]),
        ("Acetazolamide",[{"ingredient": "Acetazolamide",     "mg_range": (125, 500)}]),
    ],
    "Otitis Media": [
        ("Amoxicillin", [{"ingredient": "Amoxicillin Trihydrate","mg_range":(250, 500)}]),
        ("Augmentin",   [{"ingredient": "Amoxicillin",       "mg_range": (400, 875)},
                         {"ingredient": "Clavulanic Acid",   "mg_range": (57,  125)}]),
        ("Cefuroxime",  [{"ingredient": "Cefuroxime Axetil", "mg_range": (125, 500)}]),
        ("Azithromycin",[{"ingredient": "Azithromycin Dihydrate","mg_range":(250,500)}]),
        ("Ear Drops",   [{"ingredient": "Ciprofloxacin",     "mg_range": (3,   3  )},
                         {"ingredient": "Dexamethasone",     "mg_range": (1,   1  )}]),
    ],
    "Polycystic Ovary Syndrome": [
        ("Metformin",   [{"ingredient": "Metformin HCl",        "mg_range": (500,1500)}]),
        ("Clomiphene",  [{"ingredient": "Clomiphene Citrate",   "mg_range": (25,  100)}]),
        ("Spironolactone",[{"ingredient":"Spironolactone",      "mg_range": (25,  200)}]),
        ("OCP",         [{"ingredient": "Ethinyl Estradiol",    "mg_range": (0.02,0.035)},
                         {"ingredient": "Drospirenone",         "mg_range": (3,   3  )}]),
        ("Letrozole",   [{"ingredient": "Letrozole",            "mg_range": (2.5, 5  )}]),
    ],
    "Endometriosis": [
        ("Dienogest",    [{"ingredient": "Dienogest",           "mg_range": (2,   2  )}]),
        ("GnRH Agonist", [{"ingredient": "Leuprolide Acetate",  "mg_range": (1,   7.5)}]),
        ("Norethisterone",[{"ingredient":"Norethisterone",      "mg_range": (5,   15 )}]),
        ("Danazol",      [{"ingredient": "Danazol",             "mg_range": (100, 800)}]),
    ],
    "Chronic Pain": [
        ("Pregabalin",  [{"ingredient": "Pregabalin",   "mg_range": (75,  600)}]),
        ("Gabapentin",  [{"ingredient": "Gabapentin",   "mg_range": (100,3600)}]),
        ("Tramadol",    [{"ingredient": "Tramadol HCl", "mg_range": (50,  400)}]),
        ("Duloxetine",  [{"ingredient": "Duloxetine HCl","mg_range":(30,  120)}]),
        ("Tapentadol",  [{"ingredient": "Tapentadol HCl","mg_range":(50,  250)}]),
        ("Celecoxib",   [{"ingredient": "Celecoxib",    "mg_range": (100, 400)}]),
    ],
    "Neuropathic Pain": [
        ("Pregabalin",      [{"ingredient": "Pregabalin",          "mg_range": (75,  300)}]),
        ("Gabapentin",      [{"ingredient": "Gabapentin",          "mg_range": (100,3600)}]),
        ("Duloxetine",      [{"ingredient": "Duloxetine HCl",      "mg_range": (30,  120)}]),
        ("Amitriptyline",   [{"ingredient": "Amitriptyline HCl",   "mg_range": (10,  75 )}]),
        ("Topical Lidocaine",[{"ingredient":"Lidocaine",           "mg_range": (50,  700)}]),
        ("Capsaicin Patch", [{"ingredient": "Capsaicin",           "mg_range": (8,   8  )}]),
    ],
    "Post-operative Pain": [
        ("Ketorolac",    [{"ingredient": "Ketorolac Tromethamine","mg_range":(10, 30 )}]),
        ("Paracetamol",  [{"ingredient": "Paracetamol",           "mg_range": (500,1000)}]),
        ("Morphine",     [{"ingredient": "Morphine Sulphate",     "mg_range": (2,   15 )}]),
        ("Diclofenac",   [{"ingredient": "Diclofenac Sodium",     "mg_range": (50,  75 )}]),
        ("Celecoxib",    [{"ingredient": "Celecoxib",             "mg_range": (200, 400)}]),
    ],
}

# Update symptom map too (exported so recommendation_engine.py can import if needed)
EXTRA_SYMPTOM_HINTS = {
    "chest tightness": "Asthma", "coughing blood": "Tuberculosis",
    "night sweats": "Tuberculosis", "productive cough": "Bronchitis",
    "high fever": "Pneumonia", "chills": "Malaria",
    "tremor": "Parkinson's Disease", "memory loss": "Alzheimer's Disease",
    "cholesterol": "Hyperlipidemia", "swollen ankles": "Heart Failure",
    "shortness of breath exertion": "Heart Failure",
    "psoriasis": "Psoriasis", "scaling skin": "Psoriasis",
    "fungal": "Fungal Infection", "athlete's foot": "Fungal Infection",
    "blurred vision": "Glaucoma", "eye pressure": "Glaucoma",
    "frequent urge urinate": "Benign Prostatic Hyperplasia",
    "enlarged prostate": "Benign Prostatic Hyperplasia",
    "chemotherapy": "Chemotherapy Nausea", "cancer pain": "Cancer Pain Management",
    "crohn": "Crohn's Disease", "inflammatory bowel": "Crohn's Disease",
    "pcos": "Polycystic Ovary Syndrome", "irregular periods": "Polycystic Ovary Syndrome",
    "endometriosis": "Endometriosis", "pelvic pain": "Endometriosis",
    "bone loss": "Osteoporosis", "fracture risk": "Osteoporosis",
    "kidney disease": "Chronic Kidney Disease", "renal failure": "Chronic Kidney Disease",
    "post surgery pain": "Post-operative Pain", "operation pain": "Post-operative Pain",
    "hyperactivity": "ADHD", "attention deficit": "ADHD",
    "manic episode": "Bipolar Disorder", "mood swings": "Bipolar Disorder",
    "hallucination": "Schizophrenia", "psychosis": "Schizophrenia",
    "flashback": "PTSD", "trauma": "PTSD",
    "obesity": "Obesity", "overweight": "Obesity",
    "uric acid": "Gout", "big toe pain": "Gout",
    "ms": "Multiple Sclerosis", "numb limbs": "Multiple Sclerosis",
    "dengue": "Dengue Fever", "platelet drop": "Dengue Fever",
    "tuberculosis": "Tuberculosis", "tb": "Tuberculosis",
    "malaria": "Malaria",
}


def generate_suitable_for():
    options = [
        ["Adult"], ["Adult", "Senior"], ["Child", "Adult"],
        ["Child", "Adult", "Senior"], ["Child", "Adult", "Senior"],
        ["Child", "Adult", "Senior"], ["Adult", "Senior"],
        ["Senior"], ["Child"],
    ]
    return random.choice(options)


def generate_composition(template_ingredients):
    """
    Build a realistic composition list.

    FIX (Bug 1 + Bug 3):
      - mg values are rounded with _smart_round_mg (no artificial floor).
      - percentage is DERIVED from the actual rounded mg values, not randomly
        assigned. This guarantees Amoxicillin 870mg + Clarithromycin 500mg +
        Omeprazole 40mg always shows 61.7% / 35.5% / 2.8%, not random numbers.
      - For single-ingredient medicines the percentage is always 100.0%.
      - Rounding is done on each ingredient then the last ingredient absorbs
        any floating-point residual so percentages always sum to exactly 100.0.
    """
    # Step 1: generate all mg values
    mg_values = []
    for ing in template_ingredients:
        mg_min, mg_max = ing["mg_range"]
        raw = random.uniform(mg_min, mg_max)
        mg_values.append(_smart_round_mg(raw))

    # Step 2: compute percentages from actual mg values
    total_mg = sum(mg_values)

    composition = []
    running_pct = 0.0
    n = len(template_ingredients)

    for idx, (ing, mg) in enumerate(zip(template_ingredients, mg_values)):
        if idx < n - 1:
            pct = round(mg / total_mg * 100, 1)
            running_pct += pct
        else:
            # Last ingredient absorbs residual so sum is exactly 100.0
            pct = round(100.0 - running_pct, 1)

        composition.append({
            "ingredient": ing["ingredient"],
            "mg": mg,
            "percentage": pct,
        })

    return composition


def generate_price(disease):
    base_ranges = {
        "Seasonal Flu": (40,180), "Common Cold": (30,150), "Bronchitis": (60,400),
        "Pneumonia": (150,2000), "Tuberculosis": (80,600), "Malaria": (60,500),
        "Dengue Fever": (30,300), "Hypertension": (60,800), "Angina": (80,1000),
        "Arrhythmia": (100,1500), "Heart Failure": (120,3000), "Hyperlipidemia": (80,2500),
        "Type 2 Diabetes": (100,1200), "Type 1 Diabetes": (200,3000),
        "Hypothyroidism": (30,200), "Hyperthyroidism": (60,400), "Obesity": (100,1500),
        "Migraine": (80,600), "Epilepsy": (80,1200), "Parkinson's Disease": (200,4000),
        "Alzheimer's Disease": (500,8000), "Vertigo": (40,350),
        "Multiple Sclerosis": (500,15000), "Anxiety Disorder": (60,800),
        "Major Depressive Disorder": (80,900), "Bipolar Disorder": (80,1500),
        "Schizophrenia": (100,2000), "ADHD": (150,2000), "PTSD": (80,800),
        "Insomnia": (40,400), "Gastritis": (30,300), "GERD": (40,500),
        "Gastroenteritis": (20,200), "Irritable Bowel Syndrome": (40,600),
        "Peptic Ulcer": (50,400), "Crohn's Disease": (300,10000),
        "Arthritis": (50,2000), "Osteoporosis": (100,5000), "Fibromyalgia": (100,900),
        "Lower Back Pain": (50,800), "Gout": (50,500), "Asthma": (80,1500),
        "COPD": (200,2500), "Allergic Rhinitis": (40,500),
        "Acne Vulgaris": (50,600), "Eczema": (60,3000), "Psoriasis": (100,15000),
        "Fungal Infection": (40,500), "Urinary Tract Infection": (50,400),
        "Chronic Kidney Disease": (200,5000), "Benign Prostatic Hyperplasia": (80,800),
        "Cancer Pain Management": (200,5000), "Chemotherapy Nausea": (150,3000),
        "Conjunctivitis": (30,250), "Glaucoma": (80,1000), "Otitis Media": (50,350),
        "Polycystic Ovary Syndrome": (60,800), "Endometriosis": (100,2000),
        "Chronic Pain": (100,1500), "Neuropathic Pain": (100,1200),
        "Post-operative Pain": (80,800),
    }
    lo, hi = base_ranges.get(disease, (50, 500))
    return round(random.uniform(lo, hi), 2)


def generate_dataset(output_path="medicines_dataset.csv", target_rows=2200):
    rows = []
    diseases = list(DISEASE_MEDICINE_MAP.keys())
    suffixes = ["", " XR", " SR", " Forte", " Plus", " 500mg",
                " 250mg", " 1000mg", " Generic", " OD", " CR"]

    while len(rows) < target_rows:
        for disease in diseases:
            for brand_name, ingredients in DISEASE_MEDICINE_MAP[disease]:
                if len(rows) >= target_rows:
                    break
                for v in range(random.randint(1, 3)):
                    if len(rows) >= target_rows:
                        break
                    name = brand_name + (random.choice(suffixes[1:]) if v > 0 else "")
                    rows.append({
                        "name": name,
                        "disease_target": disease,
                        "composition": str(generate_composition(ingredients)),
                        "suitable_for": str(generate_suitable_for()),
                        "price": generate_price(disease),
                        "effectiveness_score": round(random.uniform(0.50, 0.97), 4),
                        "availability": random.random() > 0.15,
                    })

    random.shuffle(rows)
    rows = rows[:target_rows]
    fieldnames = ["name","disease_target","composition","suitable_for",
                  "price","effectiveness_score","availability"]
    with open(output_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Generated {len(rows)} rows, {len(diseases)} diseases -> {output_path}")
    return output_path


if __name__ == "__main__":
    generate_dataset()
