"""Empirically-derived BPFO/BPFI spectral bands.

The bearing geometry needed for the theoretical BPFO/BPFI/BSF formulas
isn't present in the raw files, so these bins come from comparing the
averaged power spectrum of BPFO-labeled and BPFI-labeled
`aceleracao_x_mancal_a` windows against operacao_normal, at native
N=64/fs=25600Hz resolution (bearing-defect resonances sit at
~1.3-2.2 kHz, well above the frequency range that buries the rotational
fundamental at this resolution -- no decimation needed here).

Both bands come out roughly two orders of magnitude above the
normal/unbalance/misalign level for either bearing-fault class. They
aren't individually specific to BPFO vs. BPFI, but that isn't required:
both fault types collapse into the single "desgaste_rolamento" label,
so either band firing strongly is the signal that matters.

BSF (ball spin frequency) has no empirical grounding in this dataset --
there's no ball-defect-labeled source to derive it from -- and is
omitted rather than guessed from an unverified formula.

Not currently wired into the feature vector: the low-frequency
vibration spectrum used elsewhere already separates
`desgaste_rolamento` from the other classes without needing a
dedicated bearing-fault band.
"""

BPFO_BINS = (5, 6)  # ~2000-2400 Hz at fs=25600, N=64
BPFI_BINS = (3, 4)  # ~1200-1600 Hz at fs=25600, N=64
