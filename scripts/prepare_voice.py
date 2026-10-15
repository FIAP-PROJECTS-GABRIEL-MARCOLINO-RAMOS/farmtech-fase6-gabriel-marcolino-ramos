"""Edita a gravação original localmente, sem síntese de voz.

Requer video/transcricao-original.json gerado com timestamps por palavra.
Preserva a origem, encurta pausas detectadas e remove uma frase redundante.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import wave

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'video/voz-gabriel.ogg'
CACHE = ROOT / '.cache'
OUT = ROOT / 'output/video'
OUT.mkdir(parents=True, exist_ok=True)
transcript = json.loads((ROOT / 'video/transcricao-original.json').read_text())
info = json.loads(subprocess.check_output(['ffprobe', '-v', 'quiet', '-show_format', '-of', 'json', str(SOURCE)]))
original_duration = float(info['format']['duration'])
start = max(0, transcript[0]['words'][0]['start'] - .25)
end = min(original_duration, transcript[-1]['end'] + .35)

log = (CACHE / 'voice-silence.log').read_text()
silences = re.findall(r'silence_start: ([0-9.]+).*?silence_end: ([0-9.]+)', log, re.S)
cuts = [(float(a)+.175, float(b)-.175) for a, b in silences if float(b)-float(a) > .6]
# O áudio já explica os dois treinos independentes antes desta frase repetida.
cuts.append((149.05, 156.35))
cuts = sorted((max(start, a), min(end, b)) for a, b in cuts if b > start and a < end and b > a)
merged = []
for a, b in cuts:
    if merged and a <= merged[-1][1]:
        merged[-1][1] = max(b, merged[-1][1])
    else:
        merged.append([a, b])
kept, cursor = [], start
for a, b in merged:
    if a > cursor:
        kept.append([cursor, a])
    cursor = max(cursor, b)
if cursor < end:
    kept.append([cursor, end])
kept_duration = sum(b-a for a, b in kept)
target_duration = 297.5
rate = max(1.0, kept_duration/target_duration)

raw = CACHE / 'voice-original.wav'
trimmed = CACHE / 'voice-pauses-trimmed.wav'
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(SOURCE), '-ar', '48000', '-ac', '1', '-c:a', 'pcm_s16le', str(raw)], check=True)
with wave.open(str(raw), 'rb') as source:
    params = source.getparams()
    with wave.open(str(trimmed), 'wb') as dest:
        dest.setparams(params)
        for a, b in kept:
            source.setpos(round(a*params.framerate))
            dest.writeframes(source.readframes(round(b*params.framerate)-round(a*params.framerate)))
audio = OUT / 'voz-gabriel-editada.wav'
subprocess.run(['ffmpeg', '-y', '-hide_banner', '-i', str(trimmed),
                '-af', f'atempo={rate:.9f},loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json',
                '-ar', '48000', '-ac', '1', '-c:a', 'pcm_s16le', str(audio)],
                check=True, stdout=subprocess.DEVNULL, stderr=(CACHE/'voice-normalization.log').open('w'))
edited = json.loads(subprocess.check_output(['ffprobe', '-v', 'quiet', '-show_format', '-of', 'json', str(audio)]))
edited_duration = float(edited['format']['duration'])
assert edited_duration < 299

def map_time(t):
    return sum(max(0, min(t, b)-a) for a, b in kept if t > a)/rate

original_starts = [start, 40.0, 98.0, 125.5, 181.6, 190.0, 202.4, 268.5, 281.8, 296.1, 323.9]
starts = [map_time(t) for t in original_starts]
scenes = [{'frame':i, 'original_start':t, 'start':starts[i],
           'end':starts[i+1] if i+1 < len(starts) else edited_duration}
          for i, t in enumerate(original_starts)]
record = {'original_file':'video/voz-gabriel.ogg', 'original_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'original_duration':original_duration, 'trim_bounds':[start,end], 'removed_intervals':merged,
          'redundant_sentence_removed':[149.05,156.35], 'kept_intervals':kept,
          'kept_duration':kept_duration, 'tempo_multiplier':rate, 'edited_duration':edited_duration,
          'processing':'voz original; pausas encurtadas; frase redundante removida; atempo preserva tom; loudnorm -16 LUFS e pico verdadeiro alvo -1,5 dBTP',
          'scene_timestamps':'limites por palavras na transcrição automática, conferidos com o roteiro; transcrição não é legenda revisada',
          'scenes':scenes}
(ROOT/'video/sincronizacao.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['original_duration','kept_duration','tempo_multiplier','edited_duration']},indent=2))
