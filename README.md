# ProvSurvive

Does the C2PA provenance of your images survive your image pipeline? `provsurvive` makes signed, watermarked test copies of your images. You put them through your resize server, CDN or editor, and it reports for every file whether the manifest survived, whether the watermark ([TrustMark](https://github.com/adobe/trustmark)) or fingerprint (ISCC) still leads back to it, and what happened to image quality. Comparing two pipeline versions shows what a change broke or fixed.

## Install

```bash
git clone https://github.com/beomjin-ahn/provsurvive && cd provsurvive
python -m venv .venv && source .venv/bin/activate
pip install -e .              # includes TrustMark and LPIPS (torch; CPU is enough)
pip install -e ".[iscc]"      # optional: the ISCC fingerprint
```

## Test your pipeline

```bash
provsurvive prepare ./my-images --out case
# run your pipeline on case/masters/*, one output folder per setting, keeping the file names:
#   processed/cdn-q80/beach.jpg, processed/resize-800/beach.webp, ...
provsurvive evaluate case processed --out report-v1
# after a change, run the pipeline again into processed2:
provsurvive evaluate case processed2 --out report-v2
provsurvive compare report-v1 report-v2
```

- `prepare` watermarks each image with TrustMark (`--binding iscc` for the fingerprint) and signs it as a photograph (`--kind generated` for AI images).
- `evaluate` matches outputs to masters by file name, or by `--map map.csv` (`output,master[,setting]`) for other layouts, and writes `summary.md` and `files.csv`.
- `compare` lists file by file what regressed, improved or changed, and how quality moved (`--out DIR` also writes `diff.csv`).

Per setting, the report shows:

| Column | Meaning |
|---|---|
| `intact` · `superseded` · `stale` · `invalid` · `absent` | the embedded manifest: valid; replaced by another valid one (a C2PA-aware editor's); no longer matching the file; broken; gone |
| `correct` · `wrong` · `ambiguous` · `none` | what a watermark lookup returns: the original manifest, another one, a tie, nothing |
| `recovered` | correct lookups, out of the masters that pass unprocessed |
| PSNR · SSIM · LPIPS | change from the master, for outputs of its size (`--align rescale` for resized ones) |

Missing, unreadable and unmatched files are counted and listed by name. Lookups search only the case's masters, and false detections and whether the provenance stays accurate about AI use are not measured.

### Example

`sh examples/demo.sh` runs all of this on four procedural images and a toy pipeline (CPU, about a minute). The pipeline copies the masters, re-encodes them as JPEG and strips their manifests; its second version also strips one copy's manifest and loses another:

```
setting  evaluated  intact  superseded  stale  invalid  absent  correct  wrong  ambiguous   none  recovered       PSNR
copy             4       4           0      0        0       0        4      0          0      0        4/4  identical
jpeg             4       0           0      0        0       4        4      0          0      0        4/4       29.7
strip            4       0           0      0        0       4        4      0          0      0        4/4  identical

copy     regressed 2 · same 2
           embedded intact→absent: 1 (syn-000)
           status ok→missing: 1 (syn-001)
jpeg     no change (4 same)
strip    no change (4 same)
```

## Benchmark suites

Code for our paper **ProvSurvive: Benchmarking C2PA Provenance under Everyday and Generative Image Edits**. `provsurvive run` applies a suite of everyday and generative edits to signed, marked masters itself (and, for the watermarks, to unmarked controls) and reports what a verifier can still get:

| Layer | Question | Outcome |
|---|---|---|
| Embedded manifest | Is it still in the file? Does the signature validate? Does the hard binding still match? | `intact` · `superseded` · `stale` · `invalid` · `absent` |
| Soft binding | Can the watermark or fingerprint still be decoded? | bit accuracy, exact payload match |
| Recovery | Does a soft-binding lookup find the **right** manifest again? | `correct` · `wrong` · `ambiguous` · `none` |
| Truthfulness | Does the provenance you get still describe the edited image (AI involvement)? | `disclosure_consistent` |
| Cost | What did the mark cost, and how much did the edit change the image? | PSNR · SSIM · LPIPS, manifest bytes |

Every run records the code commit, library and codec versions, model revisions, seeds and hardware, and every cell keeps the raw C2PA status codes. Definitions are in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

```bash
provsurvive run --workers 4       # the `quick` suite on 8 procedural images (CPU, about a minute);
                                   # writes runs/<id>/summary.md
```

### Full run

```bash
# corpus (images + index.json with per-file license records)
provsurvive corpus fetch-photos --out data/corpus/photos           # 100 CC0 photos from Wikimedia Commons
pip install -e ".[gen]"                                             # a 24 GB GPU for the next two steps
provsurvive corpus generate --out data/corpus/generated            # 100 T2I images from the packaged prompts

# stage 1 (GPU): cache the generative edits of the marked masters and of the unmarked controls
export PROVSURVIVE_CACHE=data/cache
S=data/corpus/generated,data/corpus/photos
provsurvive precompute --suite full --source $S
provsurvive precompute --suite full --source $S --unmarked
# stage 2 (CPU): everything else, in parallel; --controls-gpu also runs the generative edits on the controls
PROVSURVIVE_CACHE_ONLY=1 provsurvive run --suite full --source $S --controls-gpu --workers 12
```

With both caches in `PROVSURVIVE_CACHE` (marked masters and generative outputs), stage 2 needs no GPU and reproduces every edited image and cell state. Keep the masters cache with the generative one: watermark encoders differ slightly across CPU types, so re-marked masters would miss it.

The paper's experiments are the other bundled suites in `src/provsurvive/suites/` (`core22`, `crop2x2`, `rotation`, `sweep`, `replicate`, `editors_*`, `native1024`); run them the same way and aggregate `results.jsonl` as needed.

| Variable | Default | Used for |
|---|---|---|
| `PROVSURVIVE_HOME` | `~/.provsurvive` | test CA |
| `PROVSURVIVE_CACHE` | `~/.provsurvive/cache` | marked masters and generative outputs |
| `PROVSURVIVE_CACHE_ONLY` | unset | a generative cache miss is an error, never a model load |
| `PROVSURVIVE_VIDEOSEAL` | `$PROVSURVIVE_DATA/vendor/videoseal` (`PROVSURVIVE_DATA` defaults to `data`) | videoseal checkout for PixelSeal |
| `PROVSURVIVE_CONTACT` | this repository's URL | your contact in the Wikimedia User-Agent, which their policy requires |
| `PROVSURVIVE_RESULTS_URL_BASE` | unset | published location of `benchmark-results.json`, written into the registry record |

### What gets measured

**Edits** (suite `full`: 35; each seeded and versioned in `src/provsurvive/transforms/`). Every image-relevant pre-defined C2PA 2.4 action has at least one edit.

| Family | Edits |
|---|---|
| codec | JPEG q90/75/50, WebP q80, AVIF q60, PNG re-encode |
| metadata | strip C2PA data only; edit pixels while carrying the old manifest along |
| geometric | resize 0.5×/0.25×, crop 5%/20%, reframe to 4:5, rotate 90°, horizontal flip |
| composite | screenshot |
| color | exposure +0.5 EV, saturation ×1.3, 3D LUT, Reinhard color transfer |
| chain | edit-and-export, downscale-and-share |
| enhance, filter | sharpen, denoise, blur |
| draw, overlay | brush strokes, caption text, sticker |
| removal | classical (Telea) fill of the inpainting masks, 10% / 30% |
| generative (GPU) | Z-Image-Turbo inpainting 10% / 30% (composited and raw), img2img 0.2 / 0.4 |

**Soft bindings**, each by its [C2PA soft binding registry](https://github.com/c2pa-org/softbinding-algorithm-list) id. A plugin implements `embed`, `decode` and `match` (`src/provsurvive/bindings/base.py`).
- `com.adobe.trustmark.Q`, `.C`, `.P` (`--variant`), optionally with rotation-search decoding. As the library ships it, P marks only the central square of a non-square image.
- PixelSeal (`--binding pixelseal`, extra `[pixelseal]` plus a videoseal checkout at the commit in `bindings/pixelseal.py`). The registry lists a third-party deployment with an unpublished payload format, so the plugin uses the unregistered id `io.github.provsurvive.pixelseal-meta.v1`.
- `io.iscc.v0`, the ISCC image code (`--binding iscc`, extra `[iscc]`), normalized as iscc-sdk does but without its default trimming of uniform borders.

**Controls.** For the watermarks, the edits also run on unmarked images (the generative ones with `--controls-gpu`, as in the full run above), so every recovery rate comes with a false-detection and a false-match rate.

### Output

```
runs/<run_id>/
  run.json                        # configuration, environment, versions, seeds, timings
  assets.jsonl                    # one line per signed master (source, license, mark cost)
  results.jsonl                   # one line per image × edit (schema: provsurvive.schema.CellResult)
  controls.jsonl                  # unmarked-image cells (empty without controls)
  manifests.sqlite                # the signed manifest store the resolver looks up
  summary.csv  summary.md         # aggregates by edit, with intervals
  benchmark-results.json          # measured results for a soft binding registry entry
  benchmarking-information.json   # the registry's proposed benchmarking record, pointing to the results file
```

## Scope

- Everyday edits, not attacks: adversarial removal and forgery are out of scope.
- In `run`, photographs are signed as camera captures, so that whether recovered provenance still describes an edited photograph can be measured.

## Citation

```bibtex
@inproceedings{ahn2027provsurvive,
  title     = {{ProvSurvive}: Benchmarking {C2PA} Provenance under Everyday and Generative Image Edits},
  author    = {Ahn, Beomjin},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
  year      = {2027}
}
```

## License

MIT.
