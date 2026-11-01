"""
Maps the words labs actually use ("Urine", "ICU", "Amoxicillin", "Klebsiella") onto the
vocabulary the model was trained on. Keep all such mappings in this one file.
"""
import re

SPECIES = {
    "escherichia coli": "E. coli", "e. coli": "E. coli", "e.coli": "E. coli", "e coli": "E. coli", "ecoli": "E. coli",
    "klebsiella pneumoniae": "K. pneumoniae", "k. pneumoniae": "K. pneumoniae", "k.pneumoniae": "K. pneumoniae",
    "klebsiella": "K. pneumoniae",
    "pseudomonas aeruginosa": "P. aeruginosa", "p. aeruginosa": "P. aeruginosa", "p.aeruginosa": "P. aeruginosa",
    "pseudomonas": "P. aeruginosa",
    "acinetobacter baumannii": "A. baumannii", "a. baumannii": "A. baumannii", "a.baumannii": "A. baumannii",
    "acinetobacter": "A. baumannii",
}
MONITORED_SPECIES = ["E. coli", "K. pneumoniae", "P. aeruginosa", "A. baumannii"]


def species(name: str | None) -> str | None:
    if not name:
        return None
    return SPECIES.get(re.sub(r"\s+", " ", name.strip().lower()))


SOURCE_KEYWORDS = [
    ("GU", ["urine", "urin", "kidney", "bladder", "ureth", "prostat"]),
    ("CVS", ["blood", "heart", "vessel"]),
    ("Respiratory", ["respir", "sputum", "bronch", "trache", "endotrach", "lung", "bal", "sinus"]),
    ("INT", ["wound", "skin", "burn", "ulcer", "cellulitis", "decubitus"]),
    ("Bodily Fluids", ["pus", "fluid", "peritoneal", "pleural", "csf", "bile", "ascit", "tissue", "aspirate"]),
    ("GI", ["stool", "feces", "faeces", "rectal", "rectum", "gall", "liver", "abdominal", "appendix"]),
    ("Instruments", ["catheter", "drain", "tube", "line tip"]),
    ("HEENT", ["ear", "eye", "throat", "nose", "mouth"]),
    ("Reproductive", ["vagin", "uter", "cervix", "genital"]),
    ("Skeletal", ["bone"]),
    ("CNS", ["brain", "spinal"]),
]


def source_group(sample: str | None) -> str:
    if not sample:
        return "None Given"
    low = sample.lower()
    for grp, kws in SOURCE_KEYWORDS:
        if low == grp.lower() or any(k in low for k in kws):
            return grp
    return "None Given"


def speciality(ward: str | None) -> str:
    if not ward:
        return "None Given"
    low = ward.lower()
    icu = any(k in low for k in ("icu", "intensive", "critical"))
    if any(k in low for k in ("pediat", "paed", "nicu", "picu", "child")):
        return "Pediatric ICU" if icu else "Pediatric General"
    if "surg" in low:
        return "Surgery ICU" if icu else "Surgery General"
    if icu:
        return "Medicine ICU" if "medic" in low else "General Unspecified ICU"
    if "emerg" in low or "casualty" in low or low.strip() in ("er", "ed"):
        return "Emergency Room"
    if any(k in low for k in ("opd", "clinic", "office", "outpatient")):
        return "Clinic / Office"
    if "nursing" in low or "rehab" in low:
        return "Nursing Home / Rehab"
    if "medic" in low or "ward" in low or "general" in low:
        return "Medicine General"
    return "Other"


def patient_type(value: str | None) -> str:
    low = (value or "").strip().lower()
    if low.startswith("in") or low == "ipd":
        return "Inpatient"
    if low.startswith("out") or low == "opd":
        return "Outpatient"
    return "None Given"


AGE_GROUPS = ["0 to 2 Years", "3 to 12 Years", "13 to 18 Years", "19 to 64 Years", "65 to 84 Years", "85 and Over"]


def age_group(age: float | None = None, group: str | None = None) -> str:
    if group in AGE_GROUPS:
        return group
    if age is None:
        return "Unknown"
    for hi, g in [(2, AGE_GROUPS[0]), (12, AGE_GROUPS[1]), (18, AGE_GROUPS[2]), (64, AGE_GROUPS[3]), (84, AGE_GROUPS[4])]:
        if age <= hi:
            return g
    return AGE_GROUPS[5]


def gender(value: str | None) -> str:
    low = (value or "").strip().lower()
    return "Male" if low in ("m", "male") else "Female" if low in ("f", "female") else "Unknown"


def susceptibility(value: str | None) -> str | None:
    low = (value or "").strip().lower()
    if low in ("s", "sus", "susceptible", "sensitive"):
        return "S"
    if low in ("i", "int", "intermediate", "sdd"):
        return "I"
    if low in ("r", "res", "resistant", "resist"):
        return "R"
    return None
