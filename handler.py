def transcribe_handler(job):
    """
    Handler for RunPod serverless transcription
    """
    try:
        job_input = job.get('input', {})

        # Validate input
        audio_base64 = job_input.get('audio_base64')
        if not audio_base64:
            return {"error": "No audio_base64 provided"}

        # 🔧 GET FORCED LANGUAGE
        force_language = job_input.get('force_language')  # NEW

        print(f"📥 Received audio data: {len(audio_base64)} chars")
        if force_language:
            print(f"🌐 Forced language: {force_language}")

        # Decode base64 audio
        try:
            audio_data = base64.b64decode(audio_base64)
            print(f"📦 Decoded audio size: {len(audio_data)/1024:.1f}KB")
        except Exception as e:
            return {"error": f"Invalid base64 encoding: {str(e)}"}

        # Check file size limit
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

            # 🔧 TRANSCRIBE WITH FORCED LANGUAGE
            transcribe_options = {"verbose": False}
            if force_language:
                transcribe_options["language"] = force_language  # FORCE LANGUAGE
                print(f"🔧 Using forced language: {force_language}")
            
            result = model.transcribe(tmp_path, **transcribe_options)

            # 🔧 USE FORCED LANGUAGE IF PROVIDED
            detected_lang = force_language if force_language else result.get("language", "unknown")
            duration = result.get("duration", 0)

            # Check duration limit
            if duration > 600:
                return {"error": f"Audio too long: {duration/60:.1f} minutes (max 10 minutes)"}

            print(f"✅ Transcription complete!")
            print(f"   Language: {detected_lang}")
            print(f"   Duration: {duration:.1f}s")

            return {
                "text": result.get("text", ""),
                "segments": result.get("segments", []),
                "detected_language": detected_lang,  # 🔧 WILL BE FORCED LANGUAGE
                "duration": duration
            }

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                print(f"🗑️ Cleaned up temp file")

    except Exception as e:
        error_msg = f"Transcription error: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return {"error": error_msg}
