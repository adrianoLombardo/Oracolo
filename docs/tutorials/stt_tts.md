# STT e TTS locali

Esempio minimale per usare i modelli locali:

```python

from OcchioOnniveggente.src.hardware.local_audio import transcribe, synthesize

from OcchioOnniveggente.src.audio.local_audio import transcribe, synthesize


text = transcribe("audio.wav")
print(text)

synthesize("ciao", output="speech.wav")
```
