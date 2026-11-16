import os, re, json, time, tempfile, subprocess, shutil, asyncio, hashlib
from pathlib import Path

from PIL import Image

import streamlit as st
import requests
import edge_tts

APP_TITLE = "🎬 Myanmar Free Video Dubber"
MAX_MB = 500
GROQ_STT_MODEL = "whisper-large-v3-turbo"
GEMINI_MODEL = "gemini-3.8-flash"
OPENROUTER_MODEL = "openrouter/free"
XAI_STT_MODEL = "grok-voice-transcribe-2.0"
XAI_TTS_VOICE = "eve"
XAI_TTS_LANGUAGE = "my"
LOCAL_WHISPER_MODEL = "tiny"
CHUNK_SECONDS = 480

st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="centered")


def secret(name, default=""):
    try:
        return str(st.secrets.get(name, default) or default).strip()
    except Exception:
        return os.getenv(name, default).strip()


def run_cmd(args, timeout=900):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout)[-2500:])
    return p.stdout.strip()


def ffmpeg_path():
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("FFmpeg မတွေ့ပါ။ packages.txt ထဲတွင် ffmpeg ထည့်ထားကြောင်း စစ်ပါ။")
    return path


def probe_duration(path):
    value = run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return max(0.1, float(value))


def probe_video_info(path):
    raw = run_cmd(["ffprobe", "-v", "error", "-select_streams", "v:0",
                   "-show_entries", "stream=width,height", "-of", "json", str(path)])
    streams = json.loads(raw).get("streams") or []
    if not streams:
        raise RuntimeError("Video stream မတွေ့ပါ။")
    return int(streams[0]["width"]), int(streams[0]["height"])


def has_audio_stream(path):
    raw = run_cmd(["ffprobe", "-v", "error", "-select_streams", "a:0",
                   "-show_entries", "stream=index", "-of", "json", str(path)])
    return bool(json.loads(raw).get("streams") or [])


def extract_preview_frame(video_path, seconds, output_path, max_width=720):
    run_cmd([ffmpeg_path(), "-y", "-ss", f"{max(0, seconds):.3f}", "-i", str(video_path),
             "-frames:v", "1", "-vf", f"scale='min({max_width},iw)':-2",
             "-q:v", "3", str(output_path)], timeout=180)


def extract_audio(video_path, audio_path):
    # Original audio is extracted only for transcription. It is never copied to the final MP4.
    if not has_audio_stream(video_path):
        raise RuntimeError("ဒီ Video ထဲမှာ မူရင်းအသံ Track မရှိပါ။ စကားပြောသံပါဝင်သော Video ကို တင်ပေးပါ။")
    run_cmd([ffmpeg_path(), "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "libmp3lame", "-b:a", "48k", str(audio_path)], timeout=1800)


def split_audio(audio_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd([ffmpeg_path(), "-y", "-i", str(audio_path), "-f", "segment",
             "-segment_time", str(CHUNK_SECONDS), "-reset_timestamps", "1",
             "-c:a", "libmp3lame", "-b:a", "48k", str(out_dir / "chunk_%03d.mp3")], timeout=1800)
    return sorted(out_dir.glob("chunk_*.mp3"))


def groq_transcribe(audio_path, api_key):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    chunks = split_audio(audio_path, audio_path.parent / "audio_chunks")
    result = []
    for index, chunk in enumerate(chunks):
        with open(chunk, "rb") as file_obj:
            response = requests.post(
                url, headers=headers,
                data={"model": GROQ_STT_MODEL, "response_format": "verbose_json", "timestamp_granularities[]": "segment"},
                files={"file": (chunk.name, file_obj, "audio/mpeg")}, timeout=600)
        if response.status_code >= 400:
            raise RuntimeError(f"Groq STT error {response.status_code}: {response.text[:800]}")
        data = response.json()
        offset = index * CHUNK_SECONDS
        segments = data.get("segments") or []
        if segments:
            for item in segments:
                text = (item.get("text") or "").strip()
                if text:
                    result.append({"start": float(item.get("start", 0)) + offset,
                                   "end": float(item.get("end", item.get("start", 0) + 2)) + offset,
                                   "text": text})
        else:
            # Fallback for providers that return text but no segment timestamps.
            chunk_duration = probe_duration(chunk)
            text = (data.get("text") or "").strip()
            if text:
                result.extend(approximate_segments(text, offset, chunk_duration))
    return normalize_segments(result, probe_duration(audio_path))


def grok_transcribe(audio_path, api_key):
    """Transcribe audio with xAI's batch Grok Voice API."""
    url = "https://api.x.ai/v1/stt"
    chunks = split_audio(audio_path, audio_path.parent / "grok_audio_chunks")
    result = []
    for index, chunk in enumerate(chunks):
        with open(chunk, "rb") as file_obj:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                data={"model": XAI_STT_MODEL},
                files={"file": (chunk.name, file_obj, "audio/mpeg")},
                timeout=900,
            )
        if response.status_code >= 400:
            raise RuntimeError(f"Grok STT error {response.status_code}: {response.text[:800]}")
        data = response.json()
        offset = index * CHUNK_SECONDS
        segments = data.get("segments") or []
        if segments:
            for item in segments:
                text = (item.get("text") or "").strip()
                if text:
                    result.append({"start": float(item.get("start", 0)) + offset,
                                   "end": float(item.get("end", item.get("start", 0) + 2)) + offset,
                                   "text": text})
        else:
            text = (data.get("text") or data.get("transcript") or "").strip()
            if text:
                result.extend(approximate_segments(text, offset, probe_duration(chunk)))
    return normalize_segments(result, probe_duration(audio_path))


def approximate_segments(text, offset, duration):
    parts = [p.strip() for p in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if p.strip()]
    if not parts:
        parts = [text.strip()] if text.strip() else []
    if not parts:
        return []
    weights = [max(1, len(p)) for p in parts]
    total = sum(weights)
    cursor = 0.0
    result = []
    for part, weight in zip(parts, weights):
        span = max(1.0, duration * weight / total)
        result.append({"start": offset + cursor, "end": offset + min(duration, cursor + span), "text": part})
        cursor += span
    return result


def local_whisper_transcribe(audio_path):
    from faster_whisper import WhisperModel
    model = WhisperModel(secret("LOCAL_WHISPER_MODEL", LOCAL_WHISPER_MODEL),
                         device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), beam_size=3, vad_filter=True)
    return normalize_segments([{"start": s.start, "end": s.end, "text": s.text.strip()}
                               for s in segments if s.text.strip()], probe_duration(audio_path))


def normalize_segments(segments, duration):
    clean = []
    for item in sorted(segments, key=lambda x: float(x.get("start", 0))):
        text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
        if not text:
            continue
        start = max(0.0, min(float(duration), float(item.get("start", 0))))
        end = max(start + 0.25, min(float(duration), float(item.get("end", start + 2))))
        if start >= duration:
            continue
        clean.append({"start": start, "end": end, "text": text})
    return clean


def parse_json_array(raw):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    try:
        data = json.loads(raw)
    except Exception:
        match = re.search(r"\[.*\]", raw, flags=re.S)
        if not match:
            raise RuntimeError("AI ဘာသာပြန်ရလဒ်ကို JSON အဖြစ် ဖတ်မရပါ။")
        data = json.loads(match.group(0))
    if isinstance(data, dict):
        data = data.get("translations") or data.get("items") or []
    if not isinstance(data, list):
        raise RuntimeError("AI ဘာသာပြန်ရလဒ်ပုံစံ မမှန်ပါ။")
    return data


def call_grok_api(prompt, api_key):
    response = requests.post(
        "https://api.x.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "grok-4.7", "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.2},
        timeout=300,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Grok translation error {response.status_code}: {response.text[:700]}")
    data = response.json()
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")


def call_translation_api(prompt, provider, api_key):
    if provider == "Grok":
        return call_grok_api(prompt, api_key)
    if provider == "Gemini":
        from google import genai
        client = genai.Client(api_key=api_key)
        return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://streamlit.io", "X-Title": "Myanmar Auto Video Dubbing"},
        json={"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.2}, timeout=240)
    if response.status_code >= 400:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text[:700]}")
    return response.json()["choices"][0]["message"]["content"]


def free_translate_text(text):
    """Use the free Google Translate web endpoint through deep-translator."""
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="auto", target="my").translate(text).strip()


def free_translate_segments(segments, progress_callback=None):
    translated = []
    for index, seg in enumerate(segments):
        try:
            burmese = free_translate_text(seg["text"])
        except Exception:
            burmese = seg["text"]
        translated.append({**seg, "burmese": burmese or seg["text"]})
        if progress_callback:
            progress_callback((index + 1) / len(segments))
    return translated, "Free Google Translate"


def translate_segments(segments, grok_key, gemini_key, openrouter_key, progress_callback=None):
    translated = []
    batch_size = 18
    provider_order = []
    for start in range(0, len(segments), batch_size):
        batch = segments[start:start + batch_size]
        source_lines = [{"id": i, "text": seg["text"]} for i, seg in enumerate(batch)]
        prompt = (
            "Translate each line into natural, concise spoken Burmese for video dubbing. "
            "Preserve meaning, names, emotion and order. Do not add explanations or events. "
            "Keep each translation short enough to speak within its original time slot. "
            "Return ONLY a JSON array with objects containing id and myanmar. Keep every id.\n" +
            json.dumps(source_lines, ensure_ascii=False))
        raw = None
        errors = []
        if grok_key:
            try:
                raw = call_translation_api(prompt, "Grok", grok_key)
                provider_order.append("Grok")
            except Exception as exc:
                errors.append(f"Grok: {str(exc)[:160]}")
        if raw is None and gemini_key:
            try:
                raw = call_translation_api(prompt, "Gemini", gemini_key)
                provider_order.append("Gemini")
            except Exception as exc:
                errors.append(f"Gemini: {str(exc)[:160]}")
        if raw is None and openrouter_key:
            try:
                raw = call_translation_api(prompt, "OpenRouter", openrouter_key)
                provider_order.append("OpenRouter")
            except Exception as exc:
                errors.append(f"OpenRouter: {str(exc)[:160]}")
        if raw is None:
            if not (gemini_key or openrouter_key):
                return free_translate_segments(segments, progress_callback)
            raise RuntimeError("မြန်မာဘာသာပြန် API မရပါ။ " + " | ".join(errors))
        rows = parse_json_array(raw)
        mapping = {}
        for row in rows:
            try:
                mapping[int(row.get("id"))] = str(row.get("myanmar", row.get("translation", ""))).strip()
            except Exception:
                continue
        for i, seg in enumerate(batch):
            burmese = mapping.get(i, "")
            if not burmese:
                # Preserve the timeline rather than silently dropping a segment.
                burmese = seg["text"]
            translated.append({**seg, "burmese": burmese})
        if progress_callback:
            progress_callback(min(1.0, (start + len(batch)) / len(segments)))
    provider = " / ".join(dict.fromkeys(provider_order)) or "AI"
    return translated, provider


async def tts_save(text, voice, path):
    await edge_tts.Communicate(text, voice).save(str(path))


def grok_tts_save(text, api_key, path, voice_id=XAI_TTS_VOICE):
    response = requests.post(
        "https://api.x.ai/v1/tts",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"text": text, "voice_id": voice_id, "language": XAI_TTS_LANGUAGE},
        timeout=300,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Grok TTS error {response.status_code}: {response.text[:700]}")
    path.write_bytes(response.content)


def stamp(seconds):
    milliseconds = max(0, int(seconds * 1000))
    hours, rem = divmod(milliseconds, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def write_srt(segments, duration, path):
    blocks = []
    for index, seg in enumerate(segments, 1):
        start = min(duration, max(0, seg["start"]))
        end = min(duration, max(start + 0.25, seg["end"]))
        blocks.append(f"{index}\n{stamp(start)} --> {stamp(end)}\n{seg['burmese']}\n")
    path.write_text("\n".join(blocks), encoding="utf-8-sig")


def build_voice_track(segments, voice, workdir, duration, progress_callback=None,
                      grok_key=""):
    tts_dir = workdir / "tts_segments"
    tts_dir.mkdir(exist_ok=True)
    prepared = []
    cursor = 0.0
    for index, seg in enumerate(segments):
        out = tts_dir / f"voice_{index:04d}.mp3"
        if grok_key:
            try:
                grok_tts_save(seg["burmese"], grok_key, out)
            except Exception:
                asyncio.run(tts_save(seg["burmese"], voice, out))
        else:
            asyncio.run(tts_save(seg["burmese"], voice, out))
        tts_duration = probe_duration(out)
        requested_start = max(0.0, min(duration, float(seg["start"])))
        start = max(requested_start, cursor)
        if index + 1 < len(segments):
            next_start = max(start + 0.25, min(duration, float(segments[index + 1]["start"])))
            available = max(0.25, next_start - start)
        else:
            available = max(0.25, duration - start)
        speed = min(2.0, max(1.0, tts_duration / available))
        adjusted = out
        adjusted_duration = tts_duration / speed
        if speed > 1.02 or adjusted_duration > available:
            adjusted = tts_dir / f"voice_{index:04d}_fit.mp3"
            run_cmd([ffmpeg_path(), "-y", "-i", str(out), "-af",
                     f"atempo={speed:.3f},atrim=duration={available:.3f},asetpts=PTS-STARTPTS",
                     "-c:a", "libmp3lame", "-b:a", "96k", str(adjusted)], timeout=180)
        prepared.append((adjusted, int(start * 1000)))
        cursor = min(duration, start + min(available, adjusted_duration))
        if progress_callback:
            progress_callback((index + 1) / len(segments))
    return prepared


def render_dubbed_video(source, prepared_audio, duration, srt_path, output_path,
                        burn_subtitles, blur_regions=None, subtitle_size=30,
                        subtitle_color="White", subtitle_position="အောက်"):
    blur_regions = blur_regions or []
    args = [ffmpeg_path(), "-y", "-i", str(source), "-f", "lavfi", "-t", str(duration),
            "-i", "anullsrc=r=24000:cl=mono"]
    for audio_path, _ in prepared_audio:
        args += ["-i", str(audio_path)]
    filters = [f"[1:a]atrim=duration={duration:.3f},asetpts=PTS-STARTPTS[base]"]
    mix_labels = ["[base]"]
    for i, (_, delay_ms) in enumerate(prepared_audio, start=2):
        label = f"voice{i}"
        filters.append(f"[{i}:a]aresample=24000,adelay=delays={delay_ms}:all=1[{label}]")
        mix_labels.append(f"[{label}]")
    filters.append("".join(mix_labels) +
                   f"amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0:normalize=0,"
                   f"atrim=duration={duration:.3f},asetpts=PTS-STARTPTS[aout]")

    current_video = "0:v"
    frame_w, frame_h = probe_video_info(source)
    for i, region in enumerate(blur_regions):
        x, y, w, h = [int(v) for v in (region["x"], region["y"], region["w"], region["h"])]
        x, y = max(0, x), max(0, y)
        x, y = x - x % 2, y - y % 2
        w = max(2, min(frame_w - x, w))
        h = max(2, min(frame_h - y, h))
        w, h = w - w % 2, h - h % 2
        start_t, end_t = float(region["start"]), float(region["end"])
        if end_t <= start_t:
            raise RuntimeError(f"Blur နေရာ {i+1} ၏ အချိန်အပိုင်းအခြား မမှန်ပါ။")
        strength = max(2, min(40, int(region["strength"])))
        base_label, source_label, blur_label, out_label = (
            f"vbase{i}", f"vsrc{i}", f"vblur{i}", f"vout{i}")
        filters.append(f"[{current_video}]split=2[{base_label}][{source_label}]")
        filters.append(f"[{source_label}]crop={w}:{h}:{x}:{y},boxblur={strength}:2[{blur_label}]")
        filters.append(f"[{base_label}][{blur_label}]overlay={x}:{y}:"
                       f"enable='between(t,{start_t:.3f},{end_t:.3f})'[{out_label}]")
        current_video = out_label

    video_label = current_video
    video_filtered = bool(blur_regions) or bool(burn_subtitles)
    if burn_subtitles:
        srt_arg = str(srt_path).replace("\\\\", "/").replace(":", "\\:").replace("'", "\\'")
        color_map = {"White": "&H00FFFFFF", "Yellow": "&H0000FFFF", "Cyan": "&H00FFFF00"}
        alignment = {"အောက်": 2, "အလယ်": 5, "အပေါ်": 8}.get(subtitle_position, 2)
        force_style = (f"FontSize={int(subtitle_size)},PrimaryColour={color_map.get(subtitle_color, '&H00FFFFFF')},"
                       f"Outline=2,Shadow=1,Alignment={alignment},MarginV=36")
        filters.append(f"[{video_label}]subtitles='{srt_arg}':force_style='{force_style}'[vfinal]")
        video_label = "vfinal"
    args += ["-filter_complex", ";".join(filters)]
    args += ["-map", f"[{video_label}]" if video_filtered else "0:v:0", "-map", "[aout]",
             "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
             "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", str(output_path)]
    run_cmd(args, timeout=7200)


def main():
    st.title(APP_TITLE)
    st.write("Video တစ်ခုတည်းတင်ပါ။ မူရင်းအသံအားလုံးကို ဖယ်ရှားပြီး မြန်မာအသံနဲ့ အစားထိုးပေးပါမယ်။")
    st.info("မူရင်းဘာသာစကားအသံကို Final MP4 ထဲ မထည့်ဘဲ မြန်မာအသံတစ်ခုတည်း ထည့်ပေးပါမယ်။ API key မရှိရင် Free mode နဲ့ local processing လုပ်နိုင်ပါတယ်။")

    grok_key = secret("XAI_API_KEY")
    gemini_key = secret("GEMINI_API_KEY")
    groq_key = secret("GROQ_API_KEY")
    openrouter_key = secret("OPENROUTER_API_KEY")
    password = secret("APP_PASSWORD")
    if password:
        entered = st.text_input("ဝင်ရောက်ရန် Password", type="password")
        if entered != password:
            st.info("Password ထည့်ပြီးမှ ဆက်လုပ်နိုင်ပါမယ်။")
            st.stop()

    uploaded = st.file_uploader("Video တင်ပါ (MP4 / MOV / MKV / WEBM)", type=["mp4", "mov", "mkv", "webm"])
    active_providers = []
    if grok_key:
        active_providers.append("Grok")
    if groq_key:
        active_providers.append("Groq fallback")
    if not active_providers:
        active_providers.append("Free local Whisper → Free translation → Edge TTS")
    st.caption("အသုံးပြုမည့် processing mode: " + " → ".join(active_providers))
    voice_label = st.selectbox("မြန်မာအသံရွေးချယ်ပါ", ["နီလာ (မိန်းကလေး)", "သီဟ (ယောက်ျားလေး)"])
    burn_subtitles = st.checkbox("မြန်မာစာတန်းထိုးကို Video ပေါ်တွင်ပါ ထည့်မယ်", value=True)
    subtitle_size = st.slider("စာတန်းထိုး စာလုံးအရွယ်", 18, 48, 30)
    subtitle_color = st.selectbox("စာတန်းထိုး အရောင်", ["White", "Yellow", "Cyan"])
    subtitle_position = st.selectbox("စာတန်းထိုး နေရာ", ["အောက်", "အလယ်", "အပေါ်"])
    st.caption("Video ကြာချိန်ကို မူရင်းအတိုင်း ထိန်းထားပြီး မူရင်းအသံ Track ကို Final MP4 မှ ဖယ်ရှားပါမယ်။")

    blur_regions = []
    if uploaded:
        st.subheader("🎞️ Video Preview နှင့် ကိုယ်တိုင် Blur ရွေးချယ်ခြင်း")
        st.video(uploaded)
        preview_key = hashlib.md5(uploaded.getvalue()).hexdigest()[:12]
        if st.session_state.get("blur_video_hash") != preview_key:
            st.session_state["blur_video_hash"] = preview_key
            st.session_state["saved_blur_regions"] = []
        preview_dir = Path(tempfile.gettempdir()) / "myanmar_dub_preview"
        preview_dir.mkdir(exist_ok=True)
        preview_video = preview_dir / f"{preview_key}.mp4"
        if not preview_video.exists():
            preview_video.write_bytes(uploaded.getbuffer())
        try:
            preview_duration = probe_duration(preview_video)
            video_w, video_h = probe_video_info(preview_video)
            preview_time = st.slider("Blur နေရာရွေးရန် Video အချိန် (စက္ကန့်)", 0.0,
                                     float(max(0.1, preview_duration)), 0.0, 0.5)
            frame_path = preview_dir / f"{preview_key}_{int(preview_time*10)}.jpg"
            if not frame_path.exists():
                extract_preview_frame(preview_video, preview_time, frame_path)
            frame = Image.open(frame_path).convert("RGB")
            canvas_w, canvas_h = frame.size
            st.caption("Preview ပေါ်တွင် Blur လုပ်လိုသောနေရာကို လေးထောင့်ပုံ ဆွဲရွေးပါ။ တစ်ကြိမ်သိမ်းပြီး နောက်နေရာများကို ထပ်ရွေးနိုင်ပါတယ်။")
            try:
                from streamlit_drawable_canvas import st_canvas
                canvas = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.18)", stroke_width=3,
                    stroke_color="#ff3333", background_image=frame,
                    update_streamlit=True, height=canvas_h, width=canvas_w,
                    drawing_mode="rect",
                    key=f"blur_canvas_{preview_key}_{len(st.session_state.get('saved_blur_regions', []))}_{int(preview_time*10)}",
                )
                blur_start = st.number_input("ဒီ Blur စတင်မည့်အချိန် (စက္ကန့်)", 0.0,
                                              float(preview_duration), float(preview_time), 0.1)
                blur_end = st.number_input("ဒီ Blur ပြီးဆုံးမည့်အချိန် (စက္ကန့်)", 0.1,
                                            float(preview_duration), float(min(preview_duration, preview_time + 5.0)), 0.1)
                blur_strength = st.slider("Blur အား", 2, 40, 18)
                if st.button("➕ ရွေးထားသော Blur နေရာကို သိမ်းမယ်"):
                    objects = (canvas.json_data or {}).get("objects", []) if canvas.json_data else []
                    rects = [o for o in objects if o.get("type") == "rect" and o.get("width", 0) > 2 and o.get("height", 0) > 2]
                    if not rects:
                        st.warning("အရင်ဆုံး Preview ပေါ်မှာ လေးထောင့်ပုံ ဆွဲရွေးပါ။")
                    elif blur_end <= blur_start:
                        st.warning("Blur ပြီးဆုံးချိန်သည် စတင်ချိန်ထက် နောက်ကျရပါမယ်။")
                    else:
                        saved = st.session_state.setdefault("saved_blur_regions", [])
                        sx, sy = video_w / canvas_w, video_h / canvas_h
                        for rect in rects:
                            left = max(0, int(rect.get("left", 0) * sx))
                            top = max(0, int(rect.get("top", 0) * sy))
                            width = int(rect.get("width", 0) * float(rect.get("scaleX", 1)) * sx)
                            height = int(rect.get("height", 0) * float(rect.get("scaleY", 1)) * sy)
                            left -= left % 2
                            top -= top % 2
                            width = max(2, min(video_w - left, width - width % 2))
                            height = max(2, min(video_h - top, height - height % 2))
                            saved.append({"x": left, "y": top, "w": width, "h": height,
                                          "start": float(blur_start), "end": float(blur_end),
                                          "strength": int(blur_strength)})
                        st.success(f"Blur နေရာ {len(rects)} ခု သိမ်းပြီးပါပြီ။")
                        st.rerun()
                saved = st.session_state.get("saved_blur_regions", [])
                if saved:
                    st.write("သိမ်းထားသော Blur နေရာများ — တစ်ခုချင်း ပြင်နိုင် / ဖျက်နိုင်သည်")
                    for i, region in enumerate(saved):
                        with st.expander(f"Blur နေရာ {i+1} | {region['start']:.1f}s – {region['end']:.1f}s"):
                            c1, c2, c3 = st.columns(3)
                            region["start"] = c1.number_input("စတင်ချိန်", 0.0, float(preview_duration),
                                                               float(region["start"]), 0.1, key=f"brs_{preview_key}_{i}")
                            region["end"] = c2.number_input("ပြီးဆုံးချိန်", 0.1, float(preview_duration),
                                                             float(region["end"]), 0.1, key=f"bre_{preview_key}_{i}")
                            region["strength"] = c3.slider("Blur အား", 2, 40, int(region["strength"]),
                                                           key=f"brf_{preview_key}_{i}")
                            c4, c5, c6, c7 = st.columns(4)
                            region["x"] = c4.number_input("X", 0, video_w-2, int(region["x"]), key=f"brx_{preview_key}_{i}")
                            region["y"] = c5.number_input("Y", 0, video_h-2, int(region["y"]), key=f"bry_{preview_key}_{i}")
                            region["w"] = c6.number_input("အကျယ်", 2, video_w, int(region["w"]), key=f"brw_{preview_key}_{i}")
                            region["h"] = c7.number_input("အမြင့်", 2, video_h, int(region["h"]), key=f"brh_{preview_key}_{i}")
                            if st.button(f"ဒီ Blur နေရာ {i+1} ကို ဖျက်မယ်", key=f"brdel_{preview_key}_{i}"):
                                saved.pop(i)
                                st.session_state["saved_blur_regions"] = saved
                                st.rerun()
                    st.session_state["saved_blur_regions"] = saved
                    if st.button("သိမ်းထားသော Blur အားလုံး ဖျက်မယ်", key=f"clear_blur_{preview_key}"):
                        st.session_state["saved_blur_regions"] = []
                        st.rerun()
                    blur_regions = saved
            except ImportError:
                st.warning("Blur Canvas library မရှိသေးပါ။ requirements.txt ကို update လုပ်ပြီး redeploy လုပ်ပါ။")
        except Exception as preview_error:
            st.warning(f"Preview ပြင်ဆင်ရာတွင် အမှားရှိသည်: {preview_error}")

    if uploaded and uploaded.size > MAX_MB * 1024 * 1024:
        st.error(f"Video သည် {MAX_MB} MB ထက်ကျော်နေပါသည်။")
        st.stop()

    if st.button("🎙️ မြန်မာအသံနဲ့ Dubbing လုပ်မယ်", type="primary", disabled=uploaded is None):
        if not (grok_key or groq_key):
            st.info("API key မတွေ့ပါ။ Free local Whisper mode ကို အသုံးပြုနေပါတယ်။ ပထမဆုံး run မှာ Whisper model download လုပ်ရနိုင်ပါတယ်။")

        with tempfile.TemporaryDirectory() as temp_dir:
            work = Path(temp_dir)
            video = work / Path(uploaded.name).name
            video.write_bytes(uploaded.getbuffer())
            original_audio = work / "original_for_transcription.mp3"
            output_mp3 = work / "myanmar_voice.mp3"
            output_srt = work / "myanmar_subtitles.srt"
            output_mp4 = work / "myanmar_dubbed_video.mp4"
            duration = probe_duration(video)
            progress = st.progress(0, text="Video ပြင်ဆင်နေသည်... (0%)")
            status = st.empty()

            def set_progress(value, label):
                percent = int(max(0, min(100, value)))
                progress.progress(percent, text=f"{label} ({percent}%)")

            try:
                status.info("မူရင်းအသံကို စာသားဖော်ထုတ်ရန် ယာယီထုတ်ယူနေသည်...")
                extract_audio(video, original_audio)
                set_progress(8, "အသံဖိုင် ပြင်ဆင်ပြီးပါပြီ")

                status.info("Grok Voice STT သို့မဟုတ် Free local Whisper ဖြင့် မူရင်းစကားပြောသံနှင့် အချိန်မှတ်တမ်း ထုတ်ယူနေသည်...")
                try:
                    if not grok_key:
                        raise RuntimeError("XAI_API_KEY မရှိပါ")
                    segments = grok_transcribe(original_audio, grok_key)
                except Exception as grok_error:
                    if not groq_key:
                        status.warning(f"Grok မအောင်မြင်ပါ။ Local Whisper ကို စမ်းသုံးနေသည်။ {str(grok_error)[:180]}")
                        segments = local_whisper_transcribe(original_audio)
                    else:
                        status.warning(f"Grok မအောင်မြင်ပါ။ Groq Whisper fallback ကို စမ်းသုံးနေသည်။ {str(grok_error)[:180]}")
                        try:
                            segments = groq_transcribe(original_audio, groq_key)
                        except Exception as groq_error:
                            status.warning(f"Groq မအောင်မြင်ပါ။ Local Whisper ကို စမ်းသုံးနေသည်။ {str(groq_error)[:180]}")
                            segments = local_whisper_transcribe(original_audio)
                if not segments:
                    raise RuntimeError("မူရင်းစကားပြောသံမှ စာသားမရရှိပါ။ Video ထဲတွင် အသံရှိ/မရှိ စစ်ဆေးပါ။")
                set_progress(35, f"စကားပြောအပိုင်း {len(segments)} ခု ဖော်ထုတ်ပြီးပါပြီ")

                status.info("စာသားအပိုင်းတိုင်းကို မြန်မာလို ဘာသာပြန်နေသည်...")
                segments, translation_provider = translate_segments(
                    segments, grok_key, gemini_key, openrouter_key,
                    lambda fraction: set_progress(35 + fraction * 25, "မြန်မာဘာသာပြန်နေသည်"))
                set_progress(62, f"မြန်မာဘာသာပြန်ပြီးပါပြီ ({translation_provider})")

                voice = "my-MM-NilarNeural" if voice_label.startswith("နီလာ") else "my-MM-ThihaNeural"
                status.info("မြန်မာအသံ အပိုင်းလိုက် ဖန်တီးပြီး မူရင်းအချိန်နဲ့ ညှိနေသည်...")
                prepared_audio = build_voice_track(
                    segments, voice, work, duration,
                    lambda fraction: set_progress(62 + fraction * 20, "မြန်မာအသံ ဖန်တီးနေသည်"),
                    grok_key=grok_key)
                write_srt(segments, duration, output_srt)
                set_progress(84, "မြန်မာအသံနှင့် SRT ပြင်ဆင်ပြီးပါပြီ")

                status.info("မူရင်းအသံ Track ကို ဖယ်ရှားပြီး မြန်မာအသံတစ်ခုတည်း ထည့်နေသည်...")
                render_dubbed_video(video, prepared_audio, duration, output_srt, output_mp4,
                                    burn_subtitles,
                                    blur_regions=blur_regions, subtitle_size=subtitle_size,
                                    subtitle_color=subtitle_color, subtitle_position=subtitle_position)
                # Export a standalone MP3 from the final MP4 audio, ensuring it matches the final mix.
                run_cmd([ffmpeg_path(), "-y", "-i", str(output_mp4), "-vn", "-c:a", "libmp3lame",
                         "-b:a", "160k", str(output_mp3)], timeout=1200)
                set_progress(100, "ပြီးပါပြီ")
                status.success("Dubbing ပြီးပါပြီ။ Final MP4 တွင် မူရင်းအသံ Track မပါဝင်ပါ။")
                st.video(str(output_mp4))
                st.download_button("⬇️ မြန်မာအသံ Video (MP4)", output_mp4.read_bytes(),
                                   file_name="myanmar_dubbed_video.mp4", mime="video/mp4")
                st.download_button("⬇️ မြန်မာအသံ (MP3)", output_mp3.read_bytes(),
                                   file_name="myanmar_voice.mp3", mime="audio/mpeg")
                st.download_button("⬇️ မြန်မာစာတန်းထိုး (SRT)", output_srt.read_bytes(),
                                   file_name="myanmar_subtitles.srt", mime="application/x-subrip")
                with st.expander("Transcript / မြန်မာဘာသာပြန်စာသား"):
                    st.text_area("မူရင်း Transcript", "\n".join(s["text"] for s in segments), height=180)
                    st.text_area("မြန်မာ Script", "\n".join(s["burmese"] for s in segments), height=220)
            except Exception as exc:
                progress.empty()
                status.error(f"လုပ်ဆောင်မှု မအောင်မြင်ပါ: {exc}")
                st.caption("API quota၊ Network၊ FFmpeg သို့မဟုတ် Streamlit Cloud resource ကန့်သတ်ချက်ကြောင့် ဖြစ်နိုင်ပါတယ်။")


if __name__ == "__main__":
    main()
