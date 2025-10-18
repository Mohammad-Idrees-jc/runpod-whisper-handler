"""
RunPod Serverless Handler for Whisper Transcription
FIXED VERSION - Complete with all imports and startup code
"""
import runpod
import whisper
import tempfile
import os
import base64
import traceback

print("=" * 60)
print("🚀 STARTING RUNPOD WHISPER HANDLER")
print("=" * 60)

# Load Whisper model once (during cold start)
MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3")
print(f"📦 Loading Whisper model: {MODEL_NAME}")

try:
    model = whisper.load_model(MODEL_NAME)
    print(f"✅ Model '{MODEL_NAME}' loaded successfully!")
except Exception as e:
    print(f"❌ FATAL ERROR: Failed to load Whisper model!")
    print(f"   Error: {e}")
    traceback.print_exc()
    model = None


def transcribe_handler(job):
    """
    Handler for RunPod serverless transcription
    """
    print("\n" + "=" * 60)
    print("📥 NEW JOB RECEIVED")
    print("=" * 60)
    
    try:
        # Check if model loaded successfully
        if model is None:
            error_msg = "Whisper model failed to load during initialization"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

        job_input = job.get('input', {})
        
        # Validate input
        audio_base64 = job_input.get('audio_base64')
        if not audio_base64:
            print("❌ No audio_base64 provided in job input")
            return {"error": "No audio_base64 provided"}

        # Get forced language (if provided)
        force_language = job_input.get('force_language')
        
        print(f"📥 Received audio data: {len(audio_base64)} chars")
        if force_language:
            print(f"🌐 Forced language: {force_language}")

        # Decode base64 audio
        try:
            audio_data = base64.b64decode(audio_base64)
            print(f"📦 Decoded audio size: {len(audio_data)/1024:.1f}KB")
        except Exception as e:
            error_msg = f"Invalid base64 encoding: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

        # Check file size limit (50MB)
        file_size_mb = len(audio_data) / (1024 * 1024)
        print(f"📏 File size: {file_size_mb:.2f}MB")
        
        if file_size_mb > 50:
            error_msg = f"File too large: {file_size_mb:.1f}MB (max 50MB)"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

        # Save to temporary file
        file_ext = job_input.get('file_extension', 'wav')
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name

            file_size_kb = os.path.getsize(tmp_path) / 1024
            print(f"💾 Saved to temp file: {tmp_path} ({file_size_kb:.1f}KB)")

            # Prepare transcription options
            transcribe_options = {"verbose": False}
            if force_language:
                transcribe_options["language"] = force_language
                print(f"🔧 Using forced language: {force_language}")

            # Transcribe
            print(f"🎵 Starting transcription...")
            result = model.transcribe(tmp_path, **transcribe_options)

            # Get language and duration
            detected_lang = force_language if force_language else result.get("language", "unknown")
            duration = result.get("duration", 0)

            print(f"✅ Transcription complete!")
            print(f"   Language: {detected_lang}")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Segments: {len(result.get('segments', []))}")

            # Check duration limit (10 minutes = 600 seconds)
            if duration > 600:
                error_msg = f"Audio too long: {duration/60:.1f} minutes (max 10 minutes)"
                print(f"❌ {error_msg}")
                return {"error": error_msg}

            # Return result
            return {
                "text": result.get("text", ""),
                "segments": result.get("segments", []),
                "detected_language": detected_lang,
                "duration": duration
            }

        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                    print(f"🗑️ Cleaned up temp file: {tmp_path}")
                except Exception as cleanup_error:
                    print(f"⚠️ Failed to cleanup temp file: {cleanup_error}")

    except Exception as e:
        error_msg = f"Transcription error: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return {"error": error_msg}


# Start the RunPod serverless handler
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎯 INITIALIZING RUNPOD HANDLER")
    print("=" * 60)
    print(f"📦 Model: {MODEL_NAME}")
    print(f"⏱️ Max duration: 10 minutes")
    print(f"📏 Max size: 50MB")
    print("=" * 60 + "\n")
    
    try:
        runpod.serverless.start({"handler": transcribe_handler})
    except Exception as e:
        print(f"❌ FATAL ERROR: Failed to start RunPod handler!")
        print(f"   Error: {e}")
        traceback.print_exc()
        raise
