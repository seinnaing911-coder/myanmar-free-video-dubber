# Myanmar Free Video Dubber

Streamlit app: one video upload, timestamped speech recognition, Burmese translation and TTS, producing a clean dubbed MP4 with the original audio removed.

## Included
- One MP4 / MOV / MKV / WEBM upload.
- Grok Voice batch transcription with Groq Whisper and local faster-whisper fallbacks.
- Grok translation first, then the existing Gemini/OpenRouter fallback.
- Grok TTS first, with Burmese Edge TTS voices (Nilar and Thiha) as fallback.
- No-key Free mode using a tiny local Whisper model, free web translation, and Edge TTS.
- Original audio is removed from the final video; only generated Burmese voice is mixed.
- Final dubbed video preview plus MP4 and MP3 exports; no source preview, blur, or burned subtitles.
- Speech segments are fitted into their time slots to reduce overlap.

## Deploy
Upload `app.py`, `requirements.txt`, `packages.txt`, and `.streamlit/config.toml` to the connected GitHub repository.

Set these in Streamlit Cloud > App > Settings > Secrets:
```toml
XAI_API_KEY = "your-xai-key"
# Optional. The default tiny model is fastest for free local mode.
LOCAL_WHISPER_MODEL = "tiny"
GEMINI_API_KEY = "your-gemini-key"
GROQ_API_KEY = "your-groq-key"
OPENROUTER_API_KEY = "your-openrouter-key"
APP_PASSWORD = "your-private-password"
```
Keep API keys private. Do not commit real keys to GitHub.

## Provider order

The app uses Grok as the primary provider when `XAI_API_KEY` is configured. Grok Voice STT transcribes the uploaded audio, Grok translates the timestamped segments into concise spoken Burmese, and Grok TTS generates the replacement voice. If no API key is configured, the app runs a no-key Free mode using local faster-whisper, free web translation, and Edge TTS. If Grok fails, it falls back to Groq, local Whisper, free translation, or Edge TTS. The final MP4 never maps the original audio track; only the generated Burmese voice is included.

Free mode has no AI API bill, but the first local Whisper run downloads a model and CPU processing is slower. Public free translation endpoints can also rate-limit heavy use. For the fastest mode, configure `XAI_API_KEY` or `GROQ_API_KEY`.

## Notes
- The uploaded video must contain an audio track with speech for transcription.
- Speech recognition timestamps may not be perfect. Very long Burmese lines can be shortened by fitting them into the available slot.
- Streamlit Cloud CPU, memory, execution time, network access and provider quotas can limit large-video processing.

Free mode does not require `XAI_API_KEY` or `GROQ_API_KEY`; it uses local faster-whisper, free translation, and Edge TTS. The app must be allowed to process without an API key.
