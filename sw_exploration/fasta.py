"""FASTA/FASTQ parsing helpers built on the kseqpp cffi binding (kseq_reader).

load_fasta_pairs loads two parallel FASTA files and pairs them by index.
normalize_sequence strips whitespace and uppercases a sequence string.
"""

from __future__ import annotations

from pathlib import Path


def normalize_sequence(sequence: str, label: str) -> str:
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError(f"{label} sequence is empty")
    return normalized


def load_fasta_pairs(
    query_file: str,
    reference_file: str,
) -> list[tuple[str, str, str, str]]:
    """Load paired queries/references from two parallel FASTA files.

    Record i in query_file is aligned against record i in reference_file.
    Both files must contain the same number of records. Sequences may have
    any length; mixed lengths within a file are allowed.
    """
    from kseq_reader import KseqReader

    with KseqReader(str(Path(query_file).expanduser())) as reader:
        queries_raw = list(reader)
    with KseqReader(str(Path(reference_file).expanduser())) as reader:
        references_raw = list(reader)

    if not queries_raw or not references_raw:
        raise ValueError(
            f"files contain no sequences: {query_file}, {reference_file}"
        )
    if len(queries_raw) != len(references_raw):
        raise ValueError(
            f"paired files have different record counts: "
            f"{query_file} has {len(queries_raw)}, "
            f"{reference_file} has {len(references_raw)}"
        )

    return [
        (
            q_name,
            normalize_sequence(q_seq, "query"),
            r_name,
            normalize_sequence(r_seq, "reference"),
        )
        for (q_name, q_seq), (r_name, r_seq) in zip(queries_raw, references_raw)
    ]
