"""Translate a bumblebee gene from DNA into protein.

A complete, defensive ORF translator — the program Claudia built across
Lectures 1-4, now running on YOUR machine. Run it with:

    uv run translate_bee.py

It will ask you for a DNA sequence. Press Enter to use the built-in example:
a short coding fragment of the COI ("barcode") gene of the buff-tailed
bumblebee, Bombus terrestris — the very gene ecologists use to tell one
species from another.
"""

# The genetic code: every three-letter codon -> its amino acid (one letter).
codon_table = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
stop_codons = {"TAA", "TAG", "TGA"}

# A short coding fragment of the Bombus terrestris COI barcode gene.
EXAMPLE_BEE_GENE = "ATGTTTGTTCTTACTCATGGTAAACCTTGGGAATTAGCTCGTATTGGTAATTCTGTTGATTAA"


def clean_dna(seq):
    """Tidy a sequence and refuse anything that isn't DNA."""
    seq = seq.upper().strip()
    if not all(base in "ATGC" for base in seq):
        raise ValueError(f"that doesn't look like DNA: {seq}")
    return seq


def translate(seq):
    """Read codons from the start, stopping at the first stop codon."""
    protein = ""
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i + 3]
        if codon in stop_codons:
            break
        protein = protein + codon_table[codon]
    return protein


def main():
    print("🐝  Bumblebee gene translator")
    print("Paste a DNA sequence, or just press Enter for the example bee gene.")
    raw = input("> ")

    if raw.strip() == "":
        raw = EXAMPLE_BEE_GENE
        print(f"(using the example Bombus terrestris COI fragment)")

    try:
        seq = clean_dna(raw)
        protein = translate(seq)
        print(f"\nDNA     : {seq}")
        print(f"Protein : {protein}")
        print(f"Length  : {len(protein)} amino acids")
    except ValueError as error:
        print(f"\n⚠️  {error}")
        print("Tip: a DNA sequence uses only the letters A, T, G and C.")


if __name__ == "__main__":
    main()
