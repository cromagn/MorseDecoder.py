from pydub import AudioSegment
from pydub import effects
from pydub.silence import split_on_silence

import matplotlib.pyplot as plt

file_ogg = AudioSegment.from_ogg("C:\\Users\\romagnolic\\glouton-satnogs-data-downloader\\payload__03-20-2020T00-51-54__04-01-2020T10-51-54\\satnogs_1889358_2020-03-21T06-56-00.ogg")
chunks=split_on_silence(file_ogg)
for i, chunk in enumerate(chunks):
    # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
    silence_chunk = AudioSegment.silent(duration=500)

    # Add the padding chunk to beginning and end of the entire chunk.
    audio_chunk = silence_chunk + chunk + silence_chunk

    # Normalize the entire chunk.
    normalized_chunk = match_target_amplitude(audio_chunk, -20.0)

    # Export the audio chunk with new bitrate.
    print("Exporting chunk{0}.mp3.".format(i))
    normalized_chunk.export(
        "c:\\tmp\\Fuffa\\chunk{0}.mp3".format(i),
        bitrate = "192k",
        format = "mp3"
    )
