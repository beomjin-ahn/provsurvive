# provsurvive methodology (v1.0)

This document defines how provsurvive measures whether image provenance survives edits; every results file it writes (`benchmark-results.json`) links to it.

## 1. Scope

- **Assets:** still images.
- **Edits:** benign edits made with tools that are not C2PA-aware. Adversarial removal, forgery and collision are out of scope.
- **Perspective:** a verifier holding only the edited file and a resolver.

## 2. Masters

For each source image *s* with soft-binding algorithm *A*:

1. Embed *A* with a payload derived from a seed: `sha256(global_seed | s | "payload")`.
2. Sign a C2PA manifest over the watermarked pixels. The manifest contains:
   - a `c2pa.actions` assertion with `c2pa.created` and the source's IPTC `digitalSourceType`;
   - `c2pa.watermarked.bound` (watermarks only);
   - a `c2pa.soft-binding` assertion with *A*'s registry `alg` and value.

   The signer is a test chain (root → intermediate → ES256 leaf) with no TSA.
3. Store the master as PNG, and register its manifest store with the resolver.

## 3. Edits

An edit maps file bytes to file bytes with a per-cell seed `sha256(global_seed | s | seed_key)`. Paired variants share a `seed_key`, so they share masks; seeds include the source id, so a derived corpus (e.g. the 1024-px photographs) draws its own. Every edit's parameters and implementation version are recorded per cell, and a parametrized edit also records what it did (mask, editor and revision, prompt, steps, cache key; schema 1.4). The suites are listed in `src/provsurvive/suites/`.

## 4. Observations per cell

| Layer | Observation |
|---|---|
| Embedded | manifest present; SDK validation state; raw status codes; claim signature OK; hard binding match/mismatch/missing; active manifest is the original |
| Soft binding | detected; ECC-corrected payload; payload match; raw-bit accuracy (if exposed); ECC schema and whether it is the deployed one; fingerprint distance (fingerprints) |
| Recovery | resolver top-k over **all masters of the run** using only the decoded value; top-1 correct / wrong / none, or ambiguous when another manifest has the same top score; recovered manifest's signature |
| Quality | PSNR, SSIM (full resolution), LPIPS-Alex (long side 512) vs the master after alignment; inside/outside mask for inpainting |

## 5. States

- `embedded` is one of:
  - `intact`: present ∧ signature ok ∧ hard binding match ∧ original;
  - `superseded`: present ∧ signature ok ∧ hard binding match ∧ ¬original (a C2PA-aware editor re-signed the asset);
  - `stale`: present ∧ signature ok ∧ ¬match;
  - `invalid`: present ∧ (¬signature ok ∨ read error);
  - `absent`: no manifest.
- `recovery` is `correct` / `wrong` / `none` (top-1), or `ambiguous` when several manifests share the top score: a resolver's choice among them is arbitrary, so it counts as neither recovered nor misattributed.
- `provenance_available` = intact ∨ superseded ∨ correct.
- `misattributed` = wrong.
- `disclosure_consistent`: the digitalSourceType of the available manifest agrees with whether the edited image contains AI-generated content. That is the case when the master is generated or the edit is generative. A manifest's digitalSourceType is read from all of its actions, and a declaration of AI involvement on any action (e.g. `c2pa.edited` with compositeWithTrainedAlgorithmicMedia) wins.

## 6. Controls

Every edit, including the generative ones (their outputs come from the same precomputed cache), is applied to unmarked, unsigned copies of every source. Report:
- false-detection cells;
- false-detection images;
- false detections in the deployed ECC schema;
- false-match cells.

Cells from one source are not independent, so quote per-image counts. A run without controls reports them as not run, and controls that ran but all failed report null counts next to their failures; neither is zero. For fingerprint algorithms every image yields a code and an unmarked copy rightly finds its master's manifest, so no unmarked controls are run; report misattribution instead: across distinct masters, against registries padded with unrelated images (a query whose true master and another entry are equally near counts as tied), and for unregistered images looked up against the registry.

## 7. Reporting

- Rates are over the evaluated cells, with the expected and the failed cells counted next to them; a rate with no evaluated cell is null.
- Pooled rates (over several edits of the same images) carry image-clustered percentile-bootstrap 95% intervals; a single edit on 200 masters carries a Wilson interval of at most ±7 points.
- Report per edit and per family, split by generated / photographic masters.
- Report the mark's cost (source → master) separately from edit distortion (master → output).
- A run is identified by `run.json`: tool commit (read when the run starts), config hash, package and codec versions, generative models and revisions, seeds and hardware. Generative outputs come from a separate stage, cached under a key that covers the input pixels and size, the edit, and every generation setting (models, revisions, steps, guidance, working size); each precompute writes its own record.

Every run also writes `benchmark-results.json` (the measured results above; schema v2) and `benchmarking-information.json`, the registry's proposed benchmarking record, which points to the results file by URL and SHA-256 digest.

## 8. Files processed outside provsurvive

`prepare` makes the masters of a *case* from a folder of images as in §2 (PNG, or JPEG at quality 95 whose decoded pixels are then the master); its resolver holds the case's masters. Each master is probed once as it is: its baseline passes if it is intact and correctly recovered. `evaluate` probes the files a pipeline made from the masters, as displayed (EXIF orientation applied), as in §4, except:

- **Matching.** A file belongs to the master named by its file name without extension (in any case), and its folder is its setting, or to the master and setting an explicit map names. A folder with neither images nor subfolders is a setting whose files are all missing. Two files for one master and setting, files that match no master, and files that carry another master's valid manifest, decode to another master's watermark payload or carry another case's manifest are counted, not evaluated.
- **Truth.** Whether the processing used AI is unknown, so `disclosure_consistent` is null.
- **Quality.** Computed when the output has the master's size; with `--align rescale` also when it has the master's aspect ratio, after rescaling to the master's size (assumed uncropped). Otherwise it is not evaluated, with the reason recorded.
- **Populations.** Manifest states and recovery outcomes count every evaluated file. The recovery rate counts the evaluated files whose master passed its baseline, and quality the files with a measurement.
- **Controls.** None, so false detection and false match are not measured.
