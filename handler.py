"""
RunPod Serverless Handler for Whisper Transcription
Last updated: Oct 15, 2025 - Fixed for Contabo client
"""
import runpod
import whisper
import tempfile
import os
import base64
import traceback

# Load Whisper model once (during cold start)
print("🔄 Loading Whisper model...")
MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")  # Allow env override
model = whisper.load_model(MODEL_NAME)
print(f"✅ Model '{MODEL_NAME}' loaded successfully!")


def transcribe_handler(job):
    """
    Handler for RunPod serverless transcription
    Input format:
    {
        "audio_base64": "base64_encoded_audio_data",
        "file_extension": "wav"  # optional
    }

    Output format (✅ Fixed to match Contabo client expectations):
    {
        "text": "full transcribed text",
        "segments": [...],
        "detected_language": "en",
        "duration": 123.45
    }
    """
    try:
        job_input = job.get('input', {})

        # Validate input
        audio_base64 = job_input.get('audio_base64')
        if not audio_base64:
            return {"error": "No audio_base64 provided"}

        print(f"📥 Received audio data: {len(audio_base64)} chars")

        # Decode base64 audio
        try:
            audio_data = base64.b64decode(audio_base64)
            print(f"📦 Decoded audio size: {len(audio_data)/1024:.1f}KB")
        except Exception as e:
            return {"error": f"Invalid base64 encoding: {str(e)}"}

        # 🔧 FIX: Check file size limit (50MB)
        file_size_mb = len(audio_data) / (1024 * 1024)
        if file_size_mb > 50:
            return {"error": f"File too large: {file_size_mb:.1f}MB (max 50MB)"}

        # Save to temporary file
        file_ext = job_input.get('file_extension', 'wav')
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name

            file_size_kb = os.path.getsize(tmp_path) / 1024
            print(f"🎵 Transcribing: {file_size_kb:.1f}KB")

            # Transcribe with Whisper
            result = model.transcribe(tmp_path, verbose=False)

            detected_lang = result.get("language", "unknown")
            duration = result.get("duration", 0)

            # 🔧 FIX: Check duration limit (10 minutes = 600 seconds)
            if duration > 600:
                return {"error": f"Audio too long: {duration/60:.1f} minutes (max 10 minutes)"}

            print(f"✅ Transcription complete!")
            print(f"   Language: {detected_lang}")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Segments: {len(result.get('segments', []))}")

            # 🔧 FIX: Return format matching subtitle_utils.py expectations
            return {
                "text": result.get("text", ""),
                "segments": result.get("segments", []),
                "detected_language": detected_lang,
                "duration": duration
            }

        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                print(f"🗑️ Cleaned up temp file")

    except Exception as e:
        error_msg = f"Transcription error: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return {"error": error_msg}


# Start the RunPod serverless handler
if __name__ == "__main__":
    print("🚀 Starting RunPod Whisper handler...")
    print(f"📦 Model: {MODEL_NAME}")
    print(f"⏱️ Max duration: 10 minutes")
    print(f"📏 Max size: 50MB")
    runpod.serverless.start({"handler": transcribe_handler})
