"""Structure check before anything is built.

The check reads the PDB file, compares each protein chain with its reference
sequence in UniProt, and looks for tags, extra chains, metals, gaps and other
things that need attention. Each finding is a "problem" (the run stops until
the user deals with it) or a "note" (recorded only). The run continues only
when there are no problems, or when the user explicitly accepts them.

Reference sequence: PDB files list the UniProt entry and residue range of each
chain (DBREF records), and every known difference with its reason (SEQADV
records, for example EXPRESSION TAG or ENGINEERED MUTATION). The UniProt
sequence is downloaded once to structures/uniprot/<accession>.fasta.
"""

from __future__ import annotations

import math
import urllib.error
import urllib.request
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common modified residues, read as their parent amino acid.
    "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y", "HYP": "P", "CSO": "C", "KCX": "K", "MLY": "K",
}
NUCLEIC = {"DA", "DC", "DG", "DT", "DU", "A", "C", "G", "U", "I"}
WATER = {"HOH", "WAT", "DOD"}
STRUCTURAL_METALS = {"ZN", "FE", "FE2", "CU", "CU1", "MN", "CO", "NI", "CD", "HG", "MO", "W"}
CHECK_METALS = {"CA", "MG"}
TAG_WORDS = ("EXPRESSION TAG", "CLONING ARTIFACT", "LINKER", "INITIATING METHIONINE", "LEADER SEQUENCE",
             "ACETYLATION", "AMIDATION")
LONG_GAP = 5
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/stream?query=accession:{acc}&includeIsoform=true&format=fasta"

ResKey = tuple[str, int, str]   # (chain, residue number, insertion code)


@dataclass
class Finding:
    level: str   # "problem" or "note"
    text: str
    fix: str = ""


@dataclass
class CheckResult:
    path: str
    findings: list[Finding] = field(default_factory=list)
    remove_residues: list[ResKey] = field(default_factory=list)   # Tag residues to delete before building.
    chains: list[str] = field(default_factory=list)                # Protein chains that are kept.

    @property
    def problems(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "problem"]

    @property
    def clean(self) -> bool:
        return not self.problems

    def problem(self, text, fix=""):
        self.findings.append(Finding("problem", text, fix))

    def note(self, text):
        self.findings.append(Finding("note", text))

    def report(self) -> str:
        head = (f"Structure check of {self.path}: "
                + ("clean." if self.clean else f"{len(self.problems)} problem(s) to deal with before the run."))
        lines = [head]
        for f in self.problems:
            lines.append(f"  PROBLEM: {f.text}" + (f"\n           What to do: {f.fix}" if f.fix else ""))
        for f in self.findings:
            if f.level == "note":
                lines.append(f"  note: {f.text}")
        return "\n".join(lines)


# ---------------------------------------------------------------- reading the file

@dataclass
class _Pdb:
    seqres: dict = field(default_factory=lambda: OrderedDict())     # chain -> [resname]
    dbref: list = field(default_factory=list)                        # dicts
    seqadv: dict = field(default_factory=dict)                       # (chain, num, icode) -> (resname, dbres, comment)
    missing: dict = field(default_factory=dict)                      # chain -> {(num, icode): resname}
    observed: dict = field(default_factory=lambda: OrderedDict())   # chain -> OrderedDict((num, icode) -> resname)
    ca_c_n: dict = field(default_factory=dict)                       # (chain, num, icode) -> {atom: xyz}
    het: Counter = field(default_factory=Counter)
    water: int = 0
    models: int = 0
    altloc_residues: set = field(default_factory=set)
    missing_atoms: int = 0


def _int(s: str) -> int | None:
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        return None


def read_pdb(path: Path) -> _Pdb:
    p = _Pdb()
    dbref1 = {}
    in_first_model = True
    in_465 = in_470 = False
    for line in Path(path).read_text(errors="replace").splitlines():
        rec = line[:6]
        if rec == "MODEL ":
            p.models += 1
            in_first_model = p.models == 1
        elif rec == "SEQRES":
            p.seqres.setdefault(line[11], []).extend(line[19:].split())
        elif rec == "DBREF ":
            p.dbref.append(dict(chain=line[12], begin=_int(line[14:18]), end=_int(line[20:24]),
                                db=line[26:32].strip(), acc=line[33:41].strip(),
                                db_begin=_int(line[55:60]), db_end=_int(line[62:67])))
        elif rec == "DBREF1":
            dbref1[line[12]] = dict(chain=line[12], begin=_int(line[14:18]), end=_int(line[20:24]),
                                    db=line[26:32].strip())
        elif rec == "DBREF2" and line[12] in dbref1:
            d = dbref1.pop(line[12])
            d.update(acc=line[18:40].strip(), db_begin=_int(line[45:55]), db_end=_int(line[57:67]))
            p.dbref.append(d)
        elif rec == "SEQADV":
            num = _int(line[18:22])
            if num is not None:
                p.seqadv[(line[16], num, line[22].strip())] = (line[12:15].strip(), line[39:42].strip(),
                                                              line[49:70].strip())
        elif rec == "REMARK":
            code = line[7:10]
            if code == "465":
                if "SSSEQ" in line:
                    in_465 = True
                elif in_465 and len(line) > 26 and line[15:18].strip():
                    num = _int(line[21:26])
                    if num is not None:
                        p.missing.setdefault(line[19], {})[(num, line[26:27].strip())] = line[15:18].strip()
            elif code == "470":
                if "SSEQI" in line or "SSSEQI" in line:
                    in_470 = True
                elif in_470 and line[11:].strip():
                    p.missing_atoms += 1
        elif rec in ("ATOM  ", "HETATM") and in_first_model:
            resname, chain, num, icode = line[17:20].strip(), line[21], _int(line[22:26]), line[26].strip()
            if num is None:
                continue
            key = (chain, num, icode)
            if line[16] not in (" ", "A"):
                p.altloc_residues.add(key)
            if resname in WATER:
                p.water += 1 if line[12:16].strip() == "O" else 0
            elif rec == "ATOM  " or resname in THREE_TO_ONE:
                p.observed.setdefault(chain, OrderedDict())[(num, icode)] = resname
                name = line[12:16].strip()
                if name in ("C", "N", "CA") and line[16] in (" ", "A"):
                    try:
                        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                        p.ca_c_n.setdefault(key, {})[name] = xyz
                    except ValueError:
                        pass
            else:
                p.het[(resname, chain, num)] += 1
    return p


def uniprot_sequences(acc: str, root: Path = Path("."), opener=None) -> list[tuple[str, str]]:
    """(name, sequence) of the UniProt entry and its isoforms, canonical first.

    Cached in structures/uniprot/<acc>.fasta.
    """
    cache = root / "structures" / "uniprot" / f"{acc}.fasta"
    if not cache.exists():
        opener = opener or urllib.request.urlopen
        try:
            with opener(UNIPROT_URL.format(acc=acc), timeout=60) as r:
                text = r.read().decode()
        except (urllib.error.URLError, OSError) as e:
            raise ValueError(f"could not download UniProt entry {acc}: {e}") from e
        if not text.startswith(">"):
            raise ValueError(f"UniProt returned no sequence for {acc}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
    out, name, seq = [], None, []
    for line in cache.read_text().splitlines():
        if line.startswith(">"):
            if name:
                out.append((name, "".join(seq)))
            name, seq = line[1:].split("|")[1] if "|" in line else line[1:].split()[0], []
        else:
            seq.append(line.strip())
    if name:
        out.append((name, "".join(seq)))
    canonical = [x for x in out if x[0] == acc]
    return canonical + [x for x in out if x[0] != acc]


# ---------------------------------------------------------------- the check

def _full_numbering(p: _Pdb, chain: str) -> list[tuple[int, str]]:
    keys = set(p.observed.get(chain, {})) | set(p.missing.get(chain, {}))
    return sorted(keys, key=lambda k: (k[0], k[1]))


def _dist(a, b) -> float:
    return math.dist(a, b)


def check_structure(path: str | Path, chains: list[str] | None = None, keep_tags: bool = False,
                    uniprot: str | None = None, root: Path = Path("."), fetch=None) -> CheckResult:
    fetch = fetch or uniprot_sequences
    p = read_pdb(Path(path))
    res = CheckResult(str(path))

    protein = [c for c, r in p.observed.items() if any(n in THREE_TO_ONE for n in r.values())]
    if not protein:
        res.problem("The file has no protein chain.")
        return res
    if chains:
        unknown = [c for c in chains if c not in protein]
        if unknown:
            res.problem(f"Chain(s) {', '.join(unknown)} not found. Protein chains in the file: {', '.join(protein)}.",
                        "Choose chains that exist.")
        kept = [c for c in protein if c in chains]
        dropped = [c for c in protein if c not in chains]
        if dropped:
            res.note(f"Only chain(s) {', '.join(kept)} are kept; chain(s) {', '.join(dropped)} are removed.")
    else:
        kept = protein
        seqs = {c: "".join(THREE_TO_ONE.get(n, "X") for n in (p.seqres.get(c) or p.observed[c].values()))
                for c in protein}
        if len(protein) > 1:
            groups: dict[str, list[str]] = {}
            for c in protein:
                groups.setdefault(seqs[c], []).append(c)
            copies = [g for g in groups.values() if len(g) > 1]
            if copies:
                res.problem(f"The file has {len(protein)} protein chains ({', '.join(protein)}), and some are copies "
                            f"of the same protein ({'; '.join(', '.join(g) for g in copies)}). Crystal structures "
                            "often contain several copies that do not belong together.",
                            f"Keep one copy, for example --chains {copies[0][0]}; or keep them all with "
                            f"--chains {','.join(protein)}.")
            else:
                res.note(f"The file has {len(protein)} different protein chains ({', '.join(protein)}); all are kept.")
    res.chains = kept

    if p.models > 1:
        res.note(f"The file has {p.models} models (for example an NMR ensemble); only the first is used.")
    alt = [k for k in p.altloc_residues if k[0] in kept]
    if alt:
        res.note(f"{len(alt)} residue(s) have alternate positions; the first (A) is used.")

    # Other molecules.
    if p.water:
        res.note(f"{p.water} crystal water molecule(s) are removed (a new water box is added).")
    het_names = Counter(name for (name, chain, num) in p.het if chain in kept or not chains)
    metals = sorted(n for n in het_names if n in STRUCTURAL_METALS)
    if metals:
        res.problem(f"Metal ion(s) {', '.join(metals)} found. They may be needed for the protein structure, "
                    "and MDAgent would remove them.",
                    "Prepare the system with the metal site yourself (for example with tleap and MCPB) and use "
                    "--prmtop/--inpcrd, or accept that the metal is removed with --accept-problems.")
    check_ions = sorted(n for n in het_names if n in CHECK_METALS)
    if check_ions:
        res.note(f"Ion(s) {', '.join(check_ions)} are removed. Check that they are not needed for the structure.")
    others = sorted(n for n in het_names if n not in STRUCTURAL_METALS and n not in CHECK_METALS
                    and n not in NUCLEIC)
    if others:
        res.note(f"Other molecule(s) are removed: {', '.join(others)}.")

    for c in kept:
        unknown = [f"{n}{num}" for (num, ic), n in p.observed[c].items() if n == "UNK"]
        if unknown:
            res.problem(f"Chain {c} has unknown residue(s) ({', '.join(unknown[:5])}).",
                        "Use a structure where all residues are known.")
        modified = sorted({n for n in p.observed[c].values() if n in THREE_TO_ONE and len(THREE_TO_ONE[n]) == 1
                           and n not in ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
                                         "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL")})
        if modified:
            res.note(f"Chain {c}: modified residue(s) {', '.join(modified)} are replaced by the standard amino acid.")

    for c in kept:
        _check_chain(p, c, res, keep_tags, uniprot, root, fetch)

    if p.missing_atoms:
        res.note(f"{p.missing_atoms} residue(s) have missing side-chain atoms; they are added.")
    return res


def _check_chain(p: _Pdb, c: str, res: CheckResult, keep_tags: bool, uniprot: str | None, root: Path, fetch):
    observed = p.observed[c]
    numbering = _full_numbering(p, c)
    seqres = p.seqres.get(c, [])
    names = {k: observed.get(k) or p.missing.get(c, {}).get(k) for k in numbering}

    # Which residues belong to the reference sequence.
    refs = [d for d in p.dbref if d["chain"] == c and d["db"] in ("UNP", "UNIPROT")]
    in_ref: set = set()
    if uniprot and not refs:
        refs = [dict(chain=c, begin=numbering[0][0], end=numbering[-1][0], acc=uniprot, db_begin=None,
                     db_end=None, _user=True)]
    if not refs:
        res.problem(f"Chain {c} has no reference sequence in the file (no UniProt DBREF record), so tags and "
                    "sequence changes cannot be checked.",
                    "Give the UniProt accession with --uniprot <ACC>, or accept with --accept-problems.")
    for d in refs:
        try:
            candidates = fetch(d["acc"], root)
        except ValueError as e:
            res.problem(f"Chain {c}: {e}.", "Check the internet connection, or accept with --accept-problems.")
            continue
        best = None
        for name, ref_seq in candidates:
            for how in ("file", "aligned"):
                if how == "file" and d.get("db_begin") is None:
                    continue
                cmp = _compare(p, c, names, numbering, d, ref_seq, aligned=(how == "aligned"))
                if best is None or len(cmp["unexplained"]) < len(best["unexplained"]):
                    best = dict(cmp, name=name, how=how)
                if how == "file" and len(cmp["unexplained"]) <= 0.2 * max(1, cmp["compared"]):
                    break  # The file's numbering works; no need to align.
            if best and not best["unexplained"]:
                break
        in_ref |= best["in_ref"]
        label = f"UniProt {best['name']}" + (" (an isoform)" if best["name"] != d["acc"] else "")
        if best["how"] == "aligned" and d.get("db_begin") is not None and d.get("_user") is None:
            res.note(f"Chain {c}: the residue numbering in the file's reference record does not fit {label}; "
                     "the chain was aligned to the reference sequence instead.")
        if best["mism"]:
            res.note(f"Chain {c} differs from {label} at {len(best['mism'])} position(s), each explained in the "
                     f"file: {', '.join(best['mism'][:8])}{' ...' if len(best['mism']) > 8 else ''}.")
        if best["unexplained"]:
            u = best["unexplained"]
            res.problem(f"Chain {c} differs from {label} at {len(u)} position(s) without an explanation in the "
                        f"file: {', '.join(u[:8])}{' ...' if len(u) > 8 else ''}.",
                        "Check that this is the protein you want, or accept with --accept-problems.")
        if not best["mism"] and not best["unexplained"]:
            res.note(f"Chain {c} matches {label} (residues {best['first']}-{best['last']}).")

    # Tags: residues outside the reference range, or marked as tags.
    if refs:
        tags = [k for k in numbering if k not in in_ref
                or any(w in (p.seqadv.get((c, k[0], k[1]), ("", "", ""))[2]) for w in TAG_WORDS)]
        tag_set = set(tags)
        seen_tags = [k for k in tags if k in observed]
        if tags:
            res.note(f"Chain {c} has {len(tags)} tag or extra residue(s) that are not part of the protein "
                     f"({_ranges(tags)}); {len(seen_tags)} of them have coordinates.")
        obs_keys = list(observed)
        # Tag residues at the chain ends (with coordinates) can be removed cleanly.
        head = []
        for k in obs_keys:
            if k in tag_set:
                head.append(k)
            else:
                break
        tail = []
        for k in reversed(obs_keys):
            if k in tag_set:
                tail.append(k)
            else:
                break
        ends = set(head) | set(tail)
        inner = [k for k in seen_tags if k not in ends]
        if inner:
            res.problem(f"Chain {c} has tag or extra residue(s) inside the chain ({_ranges(inner)}), for example a "
                        "linker. They cannot be removed without breaking the chain.",
                        "Check the structure, or accept with --accept-problems.")
        if ends:
            if keep_tags:
                res.note(f"Chain {c}: the tag residue(s) at the chain ends are kept (--keep-tags).")
            else:
                res.remove_residues += [(c, n, ic) for (n, ic) in sorted(ends)]
                res.note(f"Chain {c}: tag residue(s) {_ranges(sorted(ends))} at the chain ends are removed "
                         "(use --keep-tags to keep them).")
        protein_keys = [k for k in numbering if k not in tag_set]
    else:
        protein_keys = numbering
        his_run = "".join(THREE_TO_ONE.get(n, "X") for n in seqres)
        if "HHHHHH" in his_run:
            res.note(f"Chain {c} contains six or more histidines in a row, which looks like a His-tag.")

    # Gaps inside the protein part.
    gaps, current = [], []
    if protein_keys:
        first = next((i for i, k in enumerate(protein_keys) if k in observed), None)
        last = next((i for i in range(len(protein_keys) - 1, -1, -1) if protein_keys[i] in observed), None)
        if first is None:
            res.problem(f"Chain {c} has no residues with coordinates.")
            return
        for k in protein_keys[first:last + 1]:
            if k not in observed:
                current.append(k)
            elif current:
                gaps.append(current)
                current = []
        n_term, c_term = first, len(protein_keys) - 1 - last
        if n_term or c_term:
            res.note(f"Chain {c}: {n_term} residue(s) missing at the start and {c_term} at the end; chain ends are "
                     "not modeled.")
    long = [g for g in gaps if len(g) > LONG_GAP]
    short = [g for g in gaps if len(g) <= LONG_GAP]
    if long:
        res.problem(f"Chain {c} has {len(long)} long gap(s) inside the chain ({'; '.join(_ranges(g) for g in long)}). "
                    "Modeling a long missing loop is not reliable.",
                    "Use a more complete structure (for example an AlphaFold model), or accept with --accept-problems.")
    if short:
        res.note(f"Chain {c}: {len(short)} short gap(s) inside the chain ({'; '.join(_ranges(g) for g in short)}) "
                 "are modeled.")

    # Chain breaks that the file does not explain.
    keys = [k for k in observed if k not in set((n, i) for (_, n, i) in res.remove_residues if _ == c)]
    for a, b in zip(keys, keys[1:]):
        if a not in protein_keys or b not in protein_keys:
            continue
        if any(k not in observed for k in protein_keys[protein_keys.index(a) + 1:protein_keys.index(b)]):
            continue  # A known gap.
        ca = p.ca_c_n.get((c, *a), {}).get("C")
        nb = p.ca_c_n.get((c, *b), {}).get("N")
        if ca and nb and _dist(ca, nb) > 2.0:
            res.problem(f"Chain {c} is broken between residues {a[0]} and {b[0]} (C-N distance {_dist(ca, nb):.1f} Å), "
                        "and the file does not say that residues are missing there.",
                        "Use a more complete structure, or accept with --accept-problems.")


def _compare(p: _Pdb, c: str, names: dict, numbering: list, d: dict, ref_seq: str, aligned: bool) -> dict:
    """Compare the chain with a reference sequence.

    Without `aligned`, residue number n of the chain matches reference position
    db_begin + (n - begin), as the file's DBREF record says. With `aligned`, the
    residues in the DBREF range are placed where they fit the reference best.
    """
    keys = [k for k in numbering if d["begin"] is None or d["begin"] <= k[0] <= d["end"]]
    if not keys:
        return dict(mism=[], unexplained=[], in_ref=set(), compared=0, first=0, last=0)
    if aligned:
        seq = "".join(THREE_TO_ONE.get(names[k] or "", "X") for k in keys)
        off = _best_offset(seq, ref_seq)
        pos = {k: off + i + 1 for i, k in enumerate(keys)}
    else:
        pos = {k: d["db_begin"] + (k[0] - d["begin"]) for k in keys}
    mism, unexplained = [], []
    for k in keys:
        upos = pos[k]
        if not (1 <= upos <= len(ref_seq)):
            unexplained.append(f"{k[0]} (outside the reference)")
            continue
        mine = THREE_TO_ONE.get(names[k] or "", "X")
        ref = ref_seq[upos - 1]
        if mine != ref:
            why = p.seqadv.get((c, k[0], k[1]))
            text = f"{ref}{k[0]}{mine}"
            (mism if why else unexplained).append(text + (f" ({why[2].lower()})" if why else ""))
    return dict(mism=mism, unexplained=unexplained, in_ref=set(keys), compared=len(keys),
                first=min(pos.values()), last=max(pos.values()))


def _best_offset(chain_seq: str, ref: str) -> int:
    """Start index in ref where chain_seq fits best (most identical residues)."""
    best, best_score = 0, -1
    for off in range(-len(chain_seq) + 1, len(ref)):
        score = sum(1 for i, ch in enumerate(chain_seq) if 0 <= off + i < len(ref) and ref[off + i] == ch)
        if score > best_score:
            best, best_score = off, score
    return best


def _ranges(keys) -> str:
    nums = [k[0] if isinstance(k, tuple) and len(k) == 2 else k[1] for k in keys]
    if not nums:
        return ""
    out, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = n
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ", ".join(out)
