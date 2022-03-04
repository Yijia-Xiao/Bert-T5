python extract.py esm1b_t33_650M_UR50S data/fluorescence.fasta data/embed/esm1b/ \
    --include mean per_tok bos

# python extract.py esm1b_t33_650M_UR50S examples/some_proteins.fasta examples/some_proteins_emb_esm1b/ \
#     --include mean per_tok
# python extract.py esm1b_t33_650M_UR50S examples/some_proteins.fasta examples/some_proteins_emb_esm1b/ \
#     --repr_layers 0 32 33 --include mean per_tok

