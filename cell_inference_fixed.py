import gc

all_records = {}
_loaded_pipes = {}

def get_pipe(model_id):
    if model_id not in _loaded_pipes:
        _loaded_pipes[model_id] = hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    return _loaded_pipes[model_id]


def run_inference(run_cfg, dataset):
    pipe = get_pipe(run_cfg["model_id"])
    records = []
    t0 = time.time()

    for i, item in enumerate(dataset):
        audio   = item["audio"]
        ref_raw = item.get(TEXT_KEY, "").strip()

        t_inf = time.time()
        out = pipe(
            {"array": audio["array"], "sampling_rate": audio["sampling_rate"]},
            generate_kwargs=run_cfg["generate_kwargs"],
        )
        inf_time = time.time() - t_inf

        ref_norm = normalize_for_wer(ref_raw)
        hyp_raw  = out["text"].strip()
        hyp_norm = normalize_for_wer(hyp_raw)

        w = wer(ref_norm, hyp_norm) if ref_norm else 0.0
        c = cer(ref_norm, hyp_norm) if ref_norm else 0.0

        records.append({
            "id"            : item.get("id", f"sample_{i:04d}"),
            "reference_raw" : ref_raw,
            "reference"     : ref_norm,
            "hypothesis_raw": hyp_raw,
            "hypothesis"    : hyp_norm,
            "wer"           : round(w, 4),
            "cer"           : round(c, 4),
            "duration_s"    : round(len(audio["array"]) / audio["sampling_rate"], 2),
            "inference_s"   : round(inf_time, 3),
        })

        if (i + 1) % 50 == 0:
            avg_wer = np.mean([r["wer"] for r in records])
            elapsed = time.time() - t0
            print(f"[{run_cfg['run_id']}] {i+1}/{len(dataset)}  WER={avg_wer:.2%}  {elapsed:.0f}s")

    return records


for run in RUNS:
    records = run_inference(run, dataset)
    all_records[run["run_id"]] = records

    out_path = OUTPUT_DIR / f"predictions_{run['run_id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
