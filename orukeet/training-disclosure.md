# Orukeet r3: training and benchmark exposure

This disclosure applies to [Orukeet](https://huggingface.co/oruk/orukeet), native NeMo artifact `orukeet-v0.1.0.nemo` at revision `555136b50265a132d4cea0d35560c26fc4f657ab`. Its SHA-256 is `031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`.

**LibriSpeech test-other was used directly for training and model selection.** The released r3 checkpoint trained on all 2,939 recordings for three passes, totaling 8,817 training presentations. The same split informed candidate selection. Its reported WER is therefore a result on exposed data, not a held-out generalization estimate. Exact source-ID and decoded-waveform matching confirmed all 2,939 records against the current evaluation partition.

**The preceding FT-4035 continuation used 223,452 records across 24 complete selected partitions, totaling 371.47 hours for one pass.** The previously reported 6,118 examples were a sampled follow-up fit diagnostic; they were not the full training population. Those partitions comprised EuroSpeech Bulgarian, Greek and Italian; 17 English GigaSpeechBench domains; Golos crowd Russian; Lesbos Greek; and NST Danish and Swedish. These names describe that continuation stage, not a complete accounting of upstream Parakeet or earlier Orukeet training.

**Voice Arena Monsoon English has prior evaluation and selection exposure.** All 2,102 records from the same source revision used in the current benchmark participated in the earlier evaluation that informed selection of FT-4035 training partitions. Monsoon itself was not among the 24 partitions directly trained on in that continuation. LibriSpeech test-clean was also evaluated earlier; evaluation alone is not presented as proof of direct training or of an additional selection decision.

An audit of the retained FT-4035 source and training waveform fingerprints found no exact whole-waveform matches to the eight current public partitions. This limited comparison does not rule out inherited exposure, partial overlap, different segmentation or transformations of the same recording. GigaSpeechBench domain training and GigaSpeech-Cleaned evaluation are distinct source definitions; their similar names do not establish exact overlap or independence.

The local eight-dataset mean includes test-other. A separately reported seven-dataset sensitivity omits that split, but still includes Monsoon selection exposure and does not certify the remaining audio as unseen. We have not selected a new subset retrospectively.

We ask the leaderboard maintainers how this known exposure should be represented and whether they require a different checkpoint for inclusion. We do not assume that disclosure guarantees acceptance or that the published guidelines automatically exclude this model.
