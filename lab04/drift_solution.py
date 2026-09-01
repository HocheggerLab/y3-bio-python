"""Bee barcode drift — how fast does a gene change, and how fast does its protein?

Every file in `sequences/` is the same 621-letter stretch of the COI "barcode"
gene, read from a different species of bee. This program translates them all and
compares each one to the buff-tailed bumblebee, Bombus terrestris.

Run it with:

    uv run drift.py

REFERENCE SOLUTION — the students' copy is drift.py, with the file handling removed.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# GIVEN — the biology. You do not need to change anything in this section.
# ---------------------------------------------------------------------------

# The INVERTEBRATE MITOCHONDRIAL code. It is not the table you used in Lab 2:
# in a bee's mitochondria TGA means tryptophan, not stop.
codon_table = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "M", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "W", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "S", "AGG": "S",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

REFERENCE = "Bombus_terrestris"


def sequence_from_fasta(text):
    """Pull the DNA out of FASTA text: skip the > line, join the rest."""
    lines = text.strip().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">"))


def translate(dna):
    """Read codons from the start and return the protein."""
    protein = ""
    for i in range(0, len(dna) - 2, 3):
        protein = protein + codon_table[dna[i:i + 3]]
    return protein


def count_differences(a, b):
    """How many positions differ between two equal-length sequences?"""
    return sum(1 for x, y in zip(a, b) if x != y)


# ---------------------------------------------------------------------------
# YOUR JOB — find the data, read it, and write the answer out.
# ---------------------------------------------------------------------------

# TODO 1 — point this at the sequences folder you downloaded and unzipped.
DATA_DIR = Path("/Users/hh65/code/y3-bio-python/lab04/sequences")

# TODO 2 — say something useful if that folder isn't there.
if not DATA_DIR.exists():
    raise SystemExit(f"I can't find {DATA_DIR} — check the path in TODO 1.")

# TODO 3 — read every .fasta file into a dictionary: species name -> DNA.
sequences = {}
for path in sorted(DATA_DIR.glob("*.fasta")):
    with open(path) as f:
        text = f.read()
    sequences[path.stem] = sequence_from_fasta(text)

print(f"Read {len(sequences)} sequences from {DATA_DIR.name}/")

# --- the comparison (given) ---
reference_dna = sequences[REFERENCE]
reference_protein = translate(reference_dna)

rows = []
for species, dna in sequences.items():
    if species == REFERENCE:
        continue
    dna_diffs = count_differences(reference_dna, dna)
    protein_diffs = count_differences(reference_protein, translate(dna))
    rows.append((species, dna_diffs, protein_diffs, dna_diffs - protein_diffs))

rows.sort(key=lambda row: row[1])

# TODO 4 — make a results folder next to this script.
results_dir = Path(__file__).parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# TODO 5 — write the rows out as a CSV.
out_file = results_dir / "drift.csv"
with open(out_file, "w") as f:
    f.write("species,dna_differences,protein_differences,silent\n")
    for species, dna_diffs, protein_diffs, silent in rows:
        f.write(f"{species},{dna_diffs},{protein_diffs},{silent}\n")

print(f"Wrote {len(rows)} rows to {out_file}")
print()
print("Closest and most distant relatives of the buff-tailed bumblebee:")
print(f"{'species':<28}{'DNA':>6}{'protein':>9}{'silent':>8}")
for species, dna_diffs, protein_diffs, silent in rows[:3] + rows[-3:]:
    print(f"{species:<28}{dna_diffs:>6}{protein_diffs:>9}{silent:>8}")
