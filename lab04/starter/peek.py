"""Peek inside ONE FASTA file.

Run it from the terminal, inside your lab4 folder:

    uv run peek.py

Then change the file name on the marked line to another bee and run it again.
Every file in sequences/ is named Genus_species.fasta.
"""

from pathlib import Path

# A path RELATIVE to the folder your terminal is in.
# Change the species name here (and only here):
fasta_file = Path("sequences") / "Bombus_terrestris.fasta"   # <-- try another bee

# 1. Ask the path some questions -- no file is opened yet
print("Looking for:", fasta_file)
print("Full path:  ", fasta_file.resolve())
print("Exists?     ", fasta_file.exists())
print("Name:       ", fasta_file.name)
print("Stem:       ", fasta_file.stem)      # the species name, for free
print("Suffix:     ", fasta_file.suffix)
print()

# 2. Open the file and read all of its text.
#    `with` closes the file again as soon as the indented block ends.
with open(fasta_file) as f:
    text = f.read()

# 3. A FASTA file is one header line starting with ">", then the DNA,
#    split over several lines. Split the text into lines and put the DNA back together.
lines = text.splitlines()
header = lines[0]
dna = "".join(lines[1:])

print("Lines in file:", len(lines))
print("Header:       ", header)
print("First 60 bp:  ", dna[:60])
print("Length:       ", len(dna), "bases")
