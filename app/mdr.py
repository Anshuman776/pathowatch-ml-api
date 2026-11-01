"""
Multidrug resistance (MDR) definition - Magiorakos et al. 2012, as used in the PathoWatch paper (Table XIII).

An isolate is MDR when it is non-susceptible (I or R) to at least one agent in >= 3 antimicrobial
categories that apply to its species. Only the four monitored Gram-negative species are assessed;
other organisms (e.g. S. aureus, where MRSA has its own definition) are returned as "not assessed".
"""
from .vocab import species as norm_species
from .vocab import susceptibility

EC, KP, PA, AB = "E. coli", "K. pneumoniae", "P. aeruginosa", "A. baumannii"
ALL4 = (EC, KP, PA, AB)
ENTERO = (EC, KP)

# category -> {species: [agents]}   (agent names lower-case; synonyms handled below)
CATEGORIES = {
    "Aminoglycosides": {s: ["amikacin", "gentamicin", "tobramycin"] for s in ALL4},
    "Anti-MRSA cephalosporins": {s: ["ceftaroline"] for s in ENTERO},
    "Antipseudomonal penicillins + BLI": {s: ["piperacillin-tazobactam"] for s in ALL4},
    "Carbapenems": {EC: ["ertapenem", "imipenem", "meropenem", "doripenem"],
                    KP: ["ertapenem", "imipenem", "meropenem", "doripenem"],
                    PA: ["imipenem", "meropenem", "doripenem"], AB: ["imipenem", "meropenem", "doripenem"]},
    "Extended-spectrum cephalosporins": {EC: ["ceftriaxone", "cefotaxime", "ceftazidime", "cefepime"],
                                         KP: ["ceftriaxone", "cefotaxime", "ceftazidime", "cefepime"],
                                         PA: ["ceftazidime", "cefepime"],
                                         AB: ["ceftriaxone", "cefotaxime", "ceftazidime", "cefepime"]},
    "Cephamycins": {s: ["cefoxitin"] for s in ENTERO},
    "Fluoroquinolones": {s: ["ciprofloxacin", "levofloxacin", "ofloxacin"] for s in ALL4},
    "Folate pathway inhibitors": {s: ["trimethoprim-sulfamethoxazole"] for s in (EC, KP, AB)},
    "Glycylcyclines": {s: ["tigecycline"] for s in ENTERO},
    "Monobactams": {s: ["aztreonam"] for s in (EC, KP, PA)},
    "Penicillins": {EC: ["ampicillin"]},  # K. pneumoniae is intrinsically resistant -> not counted
    "Penicillins + BLI": {EC: ["amoxicillin-clavulanate"], KP: ["amoxicillin-clavulanate"], AB: ["ampicillin-sulbactam"]},
    "Polymyxins": {s: ["colistin", "polymyxin b"] for s in ALL4},
    "Tetracyclines": {s: ["tetracycline", "minocycline", "doxycycline"] for s in (EC, KP, AB)},
}

SYNONYMS = {
    "amoxicillin": "ampicillin", "amoxycillin": "ampicillin",
    "amoxicillin clavulanate": "amoxicillin-clavulanate", "amoxycillin clavulanate": "amoxicillin-clavulanate",
    "co-amoxiclav": "amoxicillin-clavulanate", "augmentin": "amoxicillin-clavulanate",
    "piperacillin tazobactam": "piperacillin-tazobactam", "pip-tazo": "piperacillin-tazobactam",
    "piperacillin/tazobactam": "piperacillin-tazobactam",
    "ampicillin sulbactam": "ampicillin-sulbactam",
    "trimethoprim sulfa": "trimethoprim-sulfamethoxazole", "trimethoprim sulfamethoxazole": "trimethoprim-sulfamethoxazole",
    "cotrimoxazole": "trimethoprim-sulfamethoxazole", "co-trimoxazole": "trimethoprim-sulfamethoxazole",
    "tmp-smx": "trimethoprim-sulfamethoxazole",
    "polymyxin": "polymyxin b",
}


def normalise_agent(name: str) -> str:
    n = " ".join(name.strip().lower().replace("_", " ").split())
    return SYNONYMS.get(n, n)


def classify(pathogen: str, results: dict[str, str]) -> dict:
    sp = norm_species(pathogen)
    if sp is None:
        return {"assessed": False, "mdr": None, "species": None, "tested_categories": [],
                "nonsusceptible_categories": [], "ignored_agents": sorted(results),
                "reason": "Organism is outside the MDR definition used by PathoWatch "
                          "(E. coli, K. pneumoniae, P. aeruginosa, A. baumannii)."}
    res = {normalise_agent(a): susceptibility(v) for a, v in results.items()}
    tested, ns, used = [], [], set()
    for cat, per_species in CATEGORIES.items():
        agents = per_species.get(sp, [])
        vals = [res[a] for a in agents if a in res and res[a] is not None]
        used.update(a for a in agents if a in res)
        if vals:
            tested.append(cat)
            if any(v in ("I", "R") for v in vals):
                ns.append(cat)
    ignored = sorted(a for a in res if a not in used)
    if len(tested) < 3:
        return {"assessed": False, "mdr": None, "species": sp, "tested_categories": tested,
                "nonsusceptible_categories": ns, "ignored_agents": ignored,
                "reason": f"Only {len(tested)} relevant antibiotic categories tested; at least 3 are needed."}
    return {"assessed": True, "mdr": len(ns) >= 3, "species": sp, "tested_categories": tested,
            "nonsusceptible_categories": ns, "ignored_agents": ignored, "reason": None}
