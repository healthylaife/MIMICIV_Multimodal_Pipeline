
import pandas as pd
import re
import pandas as pd
from tqdm import tqdm

def _add_discharge_headers():
    SECTION_PATTERNS = [
        ("Chief Complaint",              r"(?:chief\s*complaint|^cc\b)"),
        ("Major Procedures",             r"major\s+surgical\s+or\s+invasive\s+procedure"),
        ("History of Present Illness",   r"(?:history\s+of\s+present\s+illness|present\s+illness|^hpi\b)"),
        ("Review of Systems",            r"(?:review\s+of\s+systems|^ros\b)"),
        ("Past Medical History",         r"past\s+medical\s+history"),
        ("Past Surgical History",        r"past\s+surgical\s+history"),
        ("Social History",               r"social\s+history"),
        ("Family History",               r"family\s+history"),
        ("Physical Exam",                r"(?:admission\s+physical\s+exam|physical\s+exam\b)"),
        ("Discharge Physical Exam",      r"(?:discharge\s*pe|discharge\s+physical\s+exam)"),
        ("Pertinent Results",            r"pertinent\s+results"),
        ("Labs on Admission",            r"labs?\s+on\s+admission"),
        ("Labs on Discharge",            r"labs?\s+on\s+discharge"),
        ("Microbiology",                 r"(?:microbiology|^micro\b)"),
        ("Diagnostic Paracentesis",      r"diagnos?i?stic\s+para(?:centesis)?"),
        ("Imaging",                      r"(?:imaging|radiology)\b"),
        ("Hospital Course",              r"(?:brief\s+hospital\s+course|hospital\s+course(?:\s*by\s*problem)?)"),
        ("Transitional Issues",          r"transitional\s+issues?"),
        ("Medications on Admission",     r"(?:medications?\s+(?:on|at|prior\s+to)\s+admission|home\s+medications?)"),
        ("Discharge Medications",        r"discharge\s+medications?"),
        ("Discharge Disposition",        r"discharge\s+disposition"),
        ("Discharge Diagnosis",          r"discharge\s+diagnos(?:is|es)"),
        ("Discharge Condition",          r"discharge\s+condition"),
        ("Discharge Instructions",       r"discharge\s+instructions?"),
        ("Follow-up",                    r"follow[\s-]*up(?:\s*instructions?)?"),
    ]
    COMPILED = [(name, re.compile(r"(?i)\b(" + pat + r")\s*:")) for name, pat in SECTION_PATTERNS]
    
    # strip admin banner (top-of-note demographics) so the first true header is clinical
    ADMIN = re.compile(
        r"(?im)^\s*(Name|Unit\s*No|MRN|DOB|Date\s*of\s*Birth|Sex|Service|Allergies|Attending|Admission\s*Date|Discharge\s*Date)\s*:.*?$"
    )
    return COMPILED, ADMIN

def _add_radiology_headers():
    RAD_SECTIONS = [
        ("Exam",         r"(?:exam(?:ination)?|study|procedure)"),
        ("Indication",   r"(?:indication|history|clinical\s+history|reason\s+for\s+exam)"),
        ("Comparison",   r"(?:comparison|compared\s+to|prior(?:\s+study)?)"),
        ("Technique",    r"(?:technique|protocol|acquisition|contrast|dose)"),
        ("Findings",     r"(?:findings|results)"),
        ("Impression",   r"(?:impression|conclusion|assessment|opinion)"),
        ("Addendum",     r"(?:addendum|correction|amendment)"),
    ]
    COMPILED = [(name, re.compile(rf"(?i)\b({pat})\s*:", re.IGNORECASE)) for name, pat in RAD_SECTIONS]
    ADMIN = re.compile(r"(?im)^\s*(Name|MRN|DOB|Accession|Exam\s*Date|Referring|Location)\s*:.*?$")
    return COMPILED, ADMIN
    

def _preclean_discharge(text: str, ADMIN) -> str:
    return "" if not isinstance(text, str) else ADMIN.sub("", text).strip()

def _preclean_radiology(text: str, ADMIN) -> str:
    if not isinstance(text, str): return ""
    s = ADMIN.sub("", text).strip()
    #collapse excessive whitespace to make inline headers easier to catch
    return re.sub(r"[ \t]+", " ", s)

def _split_discharge_to_sections(text: str):
    COMPILED,ADMIN = _add_discharge_headers()
    """Return list[(section_name, section_text)] using section headers only."""
    s = _preclean_discharge(text, ADMIN)
    if not s:
        return [("NOTE", "")]
    hits = [] #list of tuples when find the start and end position of title it is a hit
    for name, cre in COMPILED:
        for m in cre.finditer(s):
            hits.append((m.start(), m.end(), name))
    if not hits:
        return [("NOTE", s)]
    # sort by position and dedupe identical starts
    hits.sort(key=lambda t: (t[0], -t[1]))
    dedup = [] #store the unique section headers after removing duplicates
    last_start = -1
    for st, en, nm in hits:
        if st != last_start:
            dedup.append((st, en, nm))
            last_start = st
    # slice bodies between consecutive headers
    out = []
    for i, (st, en, nm) in enumerate(dedup):
        body_start = en
        body_end = dedup[i+1][0] if i+1 < len(dedup) else len(s)
        out.append((nm, s[body_start:body_end].strip()))
    return out

def _split_radiology_sections(text: str):
    COMPILED, ADMIN = _add_radiology_headers()
    #return the list of sections
    s = _preclean_radiology(text, ADMIN)
    if not s:
        return [("NOTE", "")]
    hits = []
    for name, cre in COMPILED:
        for m in cre.finditer(s):
            hits.append((m.start(), m.end(), name))
    if not hits:
        #no recognized headers -> keep whole report
        return [("NOTE", s.strip())]

    #sort and dedupe by start position
    hits.sort(key=lambda t: (t[0], -t[1]))
    uniq = []
    last = -1
    for st, en, nm in hits:
        if st != last:
            uniq.append((st, en, nm))
            last = st

    #slice section bodies between consecutive headers
    out = []
    for i, (st, en, nm) in enumerate(uniq):
        body_start = en
        body_end   = uniq[i+1][0] if i+1 < len(uniq) else len(s)
        body = s[body_start:body_end].strip()
        #data cleaning
        body = re.sub(r"^[\-–•:\s]+", "", body)
        out.append((nm, body))
    return out


def extract_data_sectionizer(df, type_df):
    match type_df:
        case 'discharge':
            meta_cols = [c for c in ["subject_id","hadm_id","note_seq","charttime","storetime"] if c in df.columns]
            rows = []
            for i, r in tqdm(df.iterrows(), total=len(df)):
                note_id = str(r["note_id"]) if "note_id" in r else f"note{i}"
                for k, (sec_name, sec_text) in enumerate(_split_discharge_to_sections(r.get("text", ""))):
                    base = {c: r[c] for c in meta_cols}
                    base.update({
                        "note_id": note_id,
                        "section_id": f"{note_id}-{k}",
                        "section_name": sec_name,
                        "section_text": sec_text
                    })
                    rows.append(base)
            
            sections_df = pd.DataFrame(rows)
            sections_df = sections_df[meta_cols + ["note_id","section_id","section_name","section_text"]]
            
            return sections_df
        case 'radiology':
            meta_cols = [c for c in ["subject_id","hadm_id","note_seq","charttime","storetime"] if c in df.columns]
            rows = []
            for i, r in df.iterrows():
                note_id = str(r["note_id"]) if "note_id" in r else f"note{i}"
                text    = r.get("text", "")
                for k, (sec_name, sec_text) in enumerate(_split_radiology_sections(text)):
                    base = {c: r[c] for c in meta_cols}
                    base.update({
                        "note_id": note_id,
                        "section_id": f"{note_id}-{k}",
                        "section_name": sec_name,
                        "section_text": sec_text
                    })
                    rows.append(base)
            
            rad_sections_df = pd.DataFrame(rows)
            rad_sections_df = rad_sections_df[meta_cols + ["note_id","section_id","section_name","section_text"]]
            return rad_sections_df
                        
        
