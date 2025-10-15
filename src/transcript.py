from faster_whisper import WhisperModel

def stt(clip):
    """Transcribe an audio clip into structured text using Faster Whisper.

    Loads a lightweight Whisper model, applies VAD-based segmentation,
    and returns the combined transcript as a single multiline string.

    Args:
        clip (str): Path to the input audio file.

    Returns:
        str: Transcribed conversation string.
    """
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(clip, vad_filter=True, beam_size=1)

    dialogue = []
    #for i, s in enumerate(segments, start=1):
    #    dialogue.append(f"Anonymous {i}: {s.text.strip()}")
    dialogue = [s.text.strip() for s in segments]
    structured_transcript = "\n".join(dialogue)
    return structured_transcript

#for testing
if __name__ == "__main__":
    import sys, os
    audio_path = 'data/harmful/bullying17.mp3'

    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)

    result = stt(audio_path)
    print("TRANSCRIPT:\n")
    print(result)