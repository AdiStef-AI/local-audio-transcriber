"""Local speech-to-text with optional speaker labels. No API key, no upload.

Usage:
    venv\\Scripts\\python transcribe.py recording.mp3 [transcript.txt]
        [--model small|medium|large-v3] [--lang en]
        [--diarize] [--num-speakers N] [--speakers "Alice,Bob"]

Speaker labels are on if any of --diarize / --num-speakers / --speakers is given.
Names in --speakers are assigned in order of first speech.
Output: "[mm:ss] Speaker 1: text" (or "[mm:ss] text" without diarization).

Why the wav step: on Python 3.14 the PyAV decoder bundled with faster-whisper
failed to open files, so we decode with ffmpeg ourselves and pass raw samples.
"""
import argparse, subprocess, tempfile, wave, os, sys, time
import numpy as np
import imageio_ffmpeg
from faster_whisper import WhisperModel

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
SEG_MODEL = os.path.join(MODELS, "diarization", "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx")
EMBEDDINGS = {
    "titanet-small": "nemo_en_titanet_small.onnx",
    "titanet-large": "nemo_en_titanet_large.onnx",
    "wespeaker": "wespeaker_en_voxceleb_resnet34.onnx",
    "eres2net": "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
}
DEFAULT_EMBEDDING = "eres2net"


def fmt(sec):
    sec = int(max(sec, 0))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def progress(label, frac, started):
    frac = min(max(frac, 0.0), 1.0)
    el = time.time() - started
    eta = (el / frac - el) if frac > 0.02 else None
    bar = "#" * int(frac * 30) + "-" * (30 - int(frac * 30))
    eta_s = f"ETA {fmt(eta)}" if eta is not None else "ETA --:--"
    sys.stdout.write(f"\r{label} [{bar}] {frac * 100:5.1f}%  elapsed {fmt(el)}  {eta_s}   ")
    sys.stdout.flush()


def done_line():
    sys.stdout.write("\n")
    sys.stdout.flush()


def hms(sec):
    sec = int(round(max(sec, 0)))
    h, rem = divmod(sec, 3600)
    m, r = divmod(rem, 60)
    return f"{h}h {m:02d}m {r:02d}s" if h else f"{m}m {r:02d}s"


def setup_cuda_dlls():
    """Make the CUDA libraries installed by pip (nvidia-cublas-cu12, nvidia-cudnn-cu12) findable."""
    for base in sys.path:
        root = os.path.join(base, "nvidia")
        if not os.path.isdir(root):
            continue
        for sub in os.listdir(root):
            b = os.path.join(root, sub, "bin")
            if os.path.isdir(b):
                os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(b)
                except (OSError, AttributeError):
                    pass


def make_whisper(name, device, lang):
    """Return (model, device_used). 'auto' tries the GPU and quietly falls back to CPU."""
    if device in ("auto", "cuda"):
        try:
            setup_cuda_dlls()
            m = WhisperModel(name, device="cuda", compute_type="float16", download_root=MODELS)
            # Loading can succeed while a missing DLL only fails on first use, so warm up now.
            list(m.transcribe(np.zeros(16000, np.float32), language=lang)[0])
            return m, "cuda"
        except Exception as e:  # no GPU, no driver, missing cuDNN/cuBLAS...
            if device == "cuda":
                sys.exit(f"GPU requested but not usable: {e}")
            print(f"No usable GPU ({str(e).splitlines()[0][:90]}); using CPU.")
    return WhisperModel(name, device="cpu", compute_type="int8", download_root=MODELS), "cpu"


def to_wav(src, dst):
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
         "-i", src, "-ar", "16000", "-ac", "1", dst],
        check=True,
    )


def diarize(samples, num_speakers, embedding=DEFAULT_EMBEDDING):
    import sherpa_onnx
    emb_model = os.path.join(MODELS, "diarization", EMBEDDINGS[embedding])
    for p in (SEG_MODEL, emb_model):
        if not os.path.exists(p):
            sys.exit(f"Missing diarization model: {p}")
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEG_MODEL),
            num_threads=4),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb_model, num_threads=4),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers if num_speakers > 0 else -1, threshold=0.5),
        min_duration_on=0.3, min_duration_off=0.5)
    if not cfg.validate():
        sys.exit("Invalid diarization config")
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    started = time.time()

    def cb(done, total):
        progress("Speakers   ", done / max(total, 1), started)
        return 0

    result = sd.process(samples, callback=cb).sort_by_start_time()
    progress("Speakers   ", 1.0, started)
    done_line()
    return [(r.start, r.end, r.speaker) for r in result]


def speaker_for(start, end, turns):
    best, best_ov = None, 0.0
    for ts, te, sp in turns:
        ov = min(end, te) - max(start, ts)
        if ov > best_ov:
            best, best_ov = sp, ov
    if best is None and turns:  # no overlap: take the nearest turn
        best = min(turns, key=lambda t: min(abs(t[0] - start), abs(t[1] - end)))[2]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--model", default="small")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--diarize", action="store_true")
    ap.add_argument("--num-speakers", type=int, default=0)
    ap.add_argument("--speakers", default="")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--embedding", default=DEFAULT_EMBEDDING, choices=list(EMBEDDINGS))
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.audio)[0] + ".transcript.txt"
    names = [n.strip() for n in a.speakers.split(",") if n.strip()]
    do_dia = a.diarize or a.num_speakers > 0 or bool(names)
    # Names imply the speaker count unless --num-speakers says otherwise.
    n_expected = a.num_speakers if a.num_speakers > 0 else len(names)
    if names and a.num_speakers > 0 and a.num_speakers != len(names):
        print(f"Note: {len(names)} names given but --num-speakers is {a.num_speakers}; using {a.num_speakers}.")

    t_start = time.time()
    print("Decoding audio...")
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "a.wav")
        to_wav(a.audio, wav)
        with wave.open(wav) as w:
            samples = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768

    t_decoded = time.time()
    turns = diarize(samples, n_expected, a.embedding) if do_dia else []
    t_diarized = time.time()

    model, device_used = make_whisper(a.model, a.device, a.lang)
    segs, info = model.transcribe(samples, language=a.lang, vad_filter=True, beam_size=5,
                                 word_timestamps=bool(turns))
    total = info.duration
    print(f"Audio length: {fmt(total)}")

    order, lines = {}, []
    stats = {}  # label -> [segments, words, seconds spoken]
    started = time.time()
    def who_for(sp):
        order.setdefault(sp, len(order))
        i = order[sp]
        return names[i] if i < len(names) else f"Speaker {i + 1}"

    def add(start, end, who, text):
        label = f"{who}: " if who else ""
        st = stats.setdefault(who or "All", [0, 0, 0.0])
        st[0] += 1
        st[1] += len(text.split())
        st[2] += max(end - start, 0.0)
        lines.append(f"[{fmt(start)}] {label}{text.strip()}")

    for s in segs:
        words = list(s.words or []) if turns else []
        if turns and words:
            # Give every word its own speaker, then start a new line at each change of speaker.
            labs = [speaker_for(w.start, w.end, turns) for w in words]
            for k in range(1, len(labs) - 1):  # a lone word between two identical labels is noise
                if labs[k - 1] == labs[k + 1] != labs[k]:
                    labs[k] = labs[k - 1]
            run = [0]
            for k in range(1, len(words) + 1):
                if k == len(words) or labs[k] != labs[run[0]]:
                    ws = words[run[0]:k]
                    add(ws[0].start, ws[-1].end, who_for(labs[run[0]]) if labs[run[0]] is not None else "Speaker (unlabelled)",
                        "".join(w.word for w in ws))
                    run = [k]
        elif turns:
            sp = speaker_for(s.start, s.end, turns)
            add(s.start, s.end, who_for(sp) if sp is not None else "Speaker (unlabelled)", s.text)
        else:
            add(s.start, s.end, "", s.text)
        progress("Transcribing", s.end / total, started)
    progress("Transcribing", 1.0, started)
    done_line()

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    t_end = time.time()

    n_words = sum(v[1] for v in stats.values())
    spoken = sum(v[2] for v in stats.values())
    total_time = t_end - t_start
    bar = "=" * 60
    print()
    print(bar)
    print(" SUMMARY")
    print(bar)
    print(f" Input file       : {os.path.basename(a.audio)} ({os.path.getsize(a.audio) / 1048576:.1f} MB)")
    print(f" Audio length     : {hms(total)}")
    print(f" Speech detected  : {hms(spoken)} ({(spoken / total * 100) if total else 0:.0f}% of the audio)")
    print(f" Whisper model    : {a.model}   language: {a.lang}   device: {device_used.upper()}")
    print(f" Transcript       : {len(lines)} lines, {n_words} words")
    print(f" Output file      : {out}")
    if turns:
        print(f" Speakers found   : {len(order)}" + (f" (expected {n_expected})" if n_expected else " (auto-detected, may be unreliable: use -NumSpeakers)"))
        for label, (n_seg, n_w, secs) in sorted(stats.items(), key=lambda kv: -kv[1][2]):
            pct = (secs / spoken * 100) if spoken else 0
            print(f"   - {label:<14} {pct:4.0f}% of speech, {hms(secs)}, {n_seg} lines, {n_w} words")
        print(f" Voice model      : {a.embedding}")
    else:
        print(" Speakers         : not labelled (use -NumSpeakers or -Diarize)")
    print(" Time taken:")
    print(f"   Decoding audio : {hms(t_decoded - t_start)}")
    if turns:
        print(f"   Speaker detect : {hms(t_diarized - t_decoded)}")
    print(f"   Transcription  : {hms(t_end - t_diarized)}")
    print(f"   TOTAL          : {hms(total_time)}")
    if total_time > 0:
        print(f"   Speed          : {total / total_time:.2f}x real time (audio length / processing time)")
    print(bar)
    sys.stdout.flush()
    # Exit hard so lingering native threads (onnxruntime/ctranslate2) can't hold the console.
    os._exit(0)


if __name__ == "__main__":
    main()
