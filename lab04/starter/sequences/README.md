# Bee COI barcode sequences

99 species of bee, one FASTA file each, for the Lab 4 "Paths & Files" session of
*Python for Biologists* (University of Sussex, Y3).

## What is in here

Each file holds the **same 621-base stretch** of the mitochondrial *cytochrome c
oxidase subunit I* (COI) gene — the "barcode" region ecologists use to tell one
species from another — read from a different species of bee.

- 99 species across 42 genera (51 *Bombus*, 48 other bees)
- 621 bases = 207 codons, identical length in every file, all in reading frame 1
- Filenames are `Genus_species.fasta`; the header line also carries the GenBank accession

Because every file covers the same window in the same frame, sequences can be
compared position by position with `zip()` — no alignment step is needed.

## Important: the genetic code

COI is **mitochondrial**. Insect mitochondria use NCBI translation table 5
(invertebrate mitochondrial), not the standard code:

| codon | standard | invertebrate mitochondrial |
|-------|----------|----------------------------|
| TGA   | stop     | tryptophan (W)             |
| ATA   | isoleucine | methionine (M)           |
| AGA / AGG | arginine | serine (S)             |

Translating these sequences with the standard table truncates almost every one
of them immediately.

## Provenance

Records were downloaded from NCBI Nucleotide, one per species, selected for
length and trimmed to a common in-frame window anchored on the conserved `IRMEL`
motif. Every sequence translates without an internal stop codon under table 5.
`Bombus cullumanus` (GU672806.1) was excluded as a divergent outlier.

Source records remain in the public domain via GenBank; accession numbers are in
each file's header line.
