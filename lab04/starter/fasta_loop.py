"""Read EVERY FASTA file in the sequences folder.

Run it from the terminal, inside your lab4 folder:

    uv run fasta_loop.py

The data: the same 621 bases of COI -- the mitochondrial gene for the last
step of respiration -- from 99 bees. The protein barely changes, but silent
DNA changes pile up. Your homework measures both.

In the lab you only run this file. Your homework is at the bottom, and the
self-checks at the very end tell you when each level is right.
"""

from pathlib import Path

# The folder THIS SCRIPT is saved in -- the same answer wherever you run it from.
HERE = Path(__file__).parent
DATA_DIR = HERE / "sequences"

# Stop early, with a clear message, if the folder isn't there.
if not DATA_DIR.exists():
    print(f"Can't find {DATA_DIR}")
    print("Is the sequences folder next to this script?")
    raise SystemExit(1)


# Bees' MITOCHONDRIA use a slightly different genetic code from the one in
# Lab 2: TGA means tryptophan (W), not stop. You need it for Level 3 only —
# scroll past it for now.
CODON_TABLE = {
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


def read_fasta(path):
    """Read one FASTA file and return its DNA as a single string."""
    with open(path) as f:
        lines = f.read().splitlines()
    return "".join(lines[1:])    # skip the ">" header, join the rest


# The loop: one pass per .fasta file, sorted A-Z so the order is the same for everyone.
count = 0
for fasta_file in sorted(DATA_DIR.glob("*.fasta")):
    species = fasta_file.stem    # the file name without .fasta
    dna = read_fasta(fasta_file)
    print(f"{species:<28} {len(dna)} bp")
    count = count + 1

print(f"\nRead {count} sequences from {DATA_DIR.name}/")


# ===========================================================================
# HOMEWORK -- turn this into a program that produces a result.
# Work through the levels in order. Run the script after EVERY step.
# ===========================================================================
#
# LEVEL 1 -- GC content
# ---------------------
#   Step 1. Above the loop (next to read_fasta), write a function:
#               def gc_content(dna):
#           Count the G's and the C's, divide by the length, times 100.
#           Hint: dna.count("G") gives you the number of G's.
#   Step 2. Add a docstring that says what goes in and what comes out.
#   Step 3. return the percentage, rounded to 1 decimal place: round(gc, 1)
#   Step 4. In the loop, call it:  gc = gc_content(dna)
#           and add it to the print:  print(f"{species:<28} {len(dna)} bp  {gc}% GC")
#   Check:  Bombus_terrestris should show 24.0% GC. Bees are AT-rich!
#
# LEVEL 2 -- write the results to a file
# --------------------------------------
#   Step 1. Before the loop, make an empty list:  rows = []
#   Step 2. Inside the loop, add one tuple per species:
#               rows.append((species, len(dna), gc))
#   Step 3. After the loop, make a results folder next to this script:
#               results_dir = HERE / "results"
#               results_dir.mkdir(exist_ok=True)
#           exist_ok=True means running it a second time is not an error.
#   Step 4. Open a file for WRITING -- note the "w":
#               with open(results_dir / "bee_summary.csv", "w") as f:
#   Step 5. Inside that with block, write a header line first:
#               f.write("species,length,gc\n")
#           then loop over rows and write one line each:
#               for species, length, gc in rows:
#                   f.write(f"{species},{length},{gc}\n")
#           Don't forget the \n -- write() does not start a new line for you.
#   Check:  results/bee_summary.csv appears in VS Code's Explorer.
#           Click it: 1 header line + 99 rows.
#
# LEVEL 3 (stretch) -- how fast does a gene change, and its protein?
# ------------------------------------------------------------------
#   Every file is the same 621-base stretch of the COI gene. Compare every bee
#   to the buff-tailed bumblebee, Bombus_terrestris.
#   Step 1. Change the loop to also store each sequence in a dictionary:
#               sequences = {}              (before the loop)
#               sequences[species] = dna    (inside the loop)
#   Step 2. Write translate(dna) -- the same idea as Lab 2: step through the
#           DNA three letters at a time and look each codon up in CODON_TABLE
#           (given near the top of this file).
#               for i in range(0, len(dna) - 2, 3):
#                   codon = dna[i:i + 3]
#   Step 3. Write count_differences(a, b): how many positions differ between
#           two strings of the same length? zip(a, b) pairs them up letter by letter.
#   Step 4. Translate the reference once:
#               ref_dna = sequences["Bombus_terrestris"]
#               ref_protein = translate(ref_dna)
#   Step 5. After the first loop, loop over sequences.items() and, for every
#           species, count the DNA differences and the protein differences
#           against the reference. Write them to a second file,
#           results/drift.csv, with the header  species,dna_diffs,protein_diffs
#           (Bombus_terrestris itself comes out as 0, 0 -- a free sanity check.)
#   Check:  Bombus_cryptarum -> 36 DNA differences, but only 1 in the protein.
#           Why would the gene change so much more than the protein it codes for?
#
# ===========================================================================


# ===========================================================================
# SELF-CHECKS (do not edit) -- each level's check switches on by itself once
# you have written that level. Run the script and read the last few lines.
# ===========================================================================
print()

# --- Level 1 ---
if "gc_content" in globals():
    test = gc_content("GGCCAATT")
    assert test is not None, "❌ Level 1: gc_content() gave back None -- did you forget to return?"
    assert round(test, 1) == 50.0, f"❌ Level 1: gc_content('GGCCAATT') should be 50.0, got {test}"
    bee = gc_content(read_fasta(DATA_DIR / "Bombus_terrestris.fasta"))
    assert round(bee, 1) == 24.0, f"❌ Level 1: Bombus_terrestris should be 24.0% GC, got {bee}"
    assert gc_content.__doc__, "❌ Level 1: works -- now add a docstring to gc_content()"
    print("✅ Level 1: gc_content() works")
else:
    print("⏳ Level 1: write gc_content(dna) to switch this check on")

# --- Level 2 ---
summary = HERE / "results" / "bee_summary.csv"
if summary.exists():
    with open(summary) as f:
        lines = f.read().splitlines()
    assert lines, "❌ Level 2: bee_summary.csv is empty -- did you write inside the with block?"
    assert len(lines) > 2, "❌ Level 2: only one line -- did you forget the \\n at the end of each write()?"
    assert lines[0] == "species,length,gc", f"❌ Level 2: first line should be the header species,length,gc -- got {lines[0]!r}"
    assert len(lines) == 100, f"❌ Level 2: expected 1 header + 99 rows = 100 lines, got {len(lines)}"
    assert "Bombus_terrestris,621,24.0" in lines, "❌ Level 2: no line reading Bombus_terrestris,621,24.0 -- check your f-string"
    print("✅ Level 2: results/bee_summary.csv is right")
else:
    print("⏳ Level 2: write results/bee_summary.csv to switch this check on")

# --- Level 3 ---
if "translate" in globals() and "count_differences" in globals():
    assert translate("ATGTGATAA") == "MW*", f"❌ Level 3: translate('ATGTGATAA') should be 'MW*', got {translate('ATGTGATAA')!r}"
    assert count_differences("ATGC", "ATGC") == 0, "❌ Level 3: identical strings should have 0 differences"
    assert count_differences("ATGC", "TTGA") == 2, "❌ Level 3: count_differences('ATGC', 'TTGA') should be 2"
    drift = HERE / "results" / "drift.csv"
    assert drift.exists(), "❌ Level 3: functions work -- now write results/drift.csv"
    with open(drift) as f:
        lines = f.read().splitlines()
    assert lines[0] == "species,dna_diffs,protein_diffs", f"❌ Level 3: header should be species,dna_diffs,protein_diffs -- got {lines[0]!r}"
    assert "Bombus_terrestris,0,0" in lines, "❌ Level 3: Bombus_terrestris compared with itself should be 0,0"
    assert "Bombus_cryptarum,36,1" in lines, "❌ Level 3: Bombus_cryptarum should be 36 DNA and 1 protein difference"
    print("✅ Level 3: results/drift.csv is right -- 36 changes in the gene, 1 in the protein. Why?")
else:
    print("⏳ Level 3 (stretch): write translate() and count_differences() to switch this check on")
