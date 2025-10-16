# -*- coding: utf-8 -*-
"""
GLOBINHO - Sistema de IA para Criação de Conteúdo
Desenvolvido para o Desafio Globo 4.0
VERSÃO FINAL - PRONTA PARA PRODUÇÃO
"""

import os, sys, subprocess, cv2, torch, whisper, json, numpy as np
import tempfile, shutil, uuid, zipfile, librosa, threading, time, pickle, random
from pathlib import Path
from flask import Flask, request, render_template, send_file, jsonify
from datetime import datetime
from werkzeug.utils import secure_filename
import moviepy.editor as mpy
import moviepy.config as mpy_config
import moviepy.video.fx.all as vfx
import qrcode
from PIL import Image
from collections import Counter

# ==============================================================================
## ⚙️ CONFIGURAÇÕES GLOBAIS (Editável pela Equipe Globo) ⚙️
# ==============================================================================

KEYWORDS_SCORING = {
    "exclusivo": 20, "urgente": 18, "polêmica": 15, "agora": 10, 
    "importante": 10, "!": 8, "?": 8
}
QR_CODE_LINK = "https://globoplay.globo.com/"
SMART_CAPTION_HOOKS = {
    "URGENTE": ["🔴 URGENTE:", "🚨 ATENÇÃO:", "PLANTÃO:"],
    "ALERTA": ["⚠️ ALERTA:", "FIQUE DE OLHO:", "IMPORTANTE:"],
    "SERIO": ["📢 FATO:", "ANÁLISE:", "ENTENDA O CASO:"],
    "POSITIVO": ["✨ BOA NOTÍCIA:", "INSPIRADOR:", "VEJA QUE LEGAL:"],
    "NEUTRO": ["📰 NOTÍCIA:", "ACONTECEU:", "FIQUE POR DENTRO:"]
}
SMART_CAPTION_CTA = [
    "O que você acha disso? Comente abaixo! 👇", "Deixe sua opinião nos comentários!",
    "Qual a sua visão sobre o assunto? Participe do debate.",
    "Concorda? Discorda? Queremos saber o que você pensa!"
]
DEFAULT_HASHTAGS = "#Globo #Jornalismo #Brasil #Noticias"
OUTPUT_FILENAME_FORMAT = {
    "post": "{base_name}_post.txt",
    "analysis": "{base_name}_analise.txt"
}

# ==============================================================================
## FIM DAS CONFIGURAÇÕES GLOBAIS
# ==============================================================================

# Configurações do Sistema
def find_imagemagick():
    program_files = Path(r"C:\Program Files")
    if program_files.exists():
        for folder in program_files.iterdir():
            if folder.is_dir() and folder.name.startswith("ImageMagick"):
                magick_path = folder / "magick.exe"
                if magick_path.exists(): return str(magick_path)
    return None

imagemagick_path = find_imagemagick()
if imagemagick_path: mpy_config.IMAGEMAGICK_BINARY = imagemagick_path

ROOT = Path(__file__).resolve().parent
UPLOAD, DONE, TEMP = ROOT / "upload", ROOT / "done", ROOT / "temp"
LOG, ANALYTICS, HISTORY, PREFERENCES = ROOT / "log.txt", ROOT / "analytics.json", ROOT / "history.json", ROOT / "preferences.pkl"

for p in (UPLOAD, DONE, TEMP): p.mkdir(exist_ok=True)

processing_status = {}
status_lock = threading.Lock()
processing_lock = threading.Lock()
is_processing = False

default_preferences = {
    "subtitle_font": "Courier-New-Bold", "subtitle_size": 70, "with_subtitles": True,
    "video_speed": 1.0, "num_clips": 3, "min_duration": 30, "max_duration": 120,
    "whisper_model": "base", "fast_mode": False, "watermark_path": ""
}
AVAILABLE_WHISPER_MODELS = {"tiny": "tiny", "base": "base", "small": "small", "medium": "medium", "large": "large"}

print("Carregando modelo de IA Whisper...")
device = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = "base"
model = whisper.load_model(WHISPER_MODEL, device=device)
current_model_name = WHISPER_MODEL
print(f"Modelo carregado: {WHISPER_MODEL} | {device.upper()}")

app = Flask(__name__)
app.secret_key = "globinho_secret_2025"
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024

try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    USE_MEDIAPIPE = True
    print("✓ MediaPipe Face Detection carregado")
except ImportError:
    USE_MEDIAPIPE = False
    CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(CASCADE)
    print("⚠ MediaPipe não disponível, usando Haar Cascade")

def log(msg):
    linha = f"{datetime.now():%d/%m/%Y %H:%M:%S} | {msg}"
    print(linha)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")

def load_preferences():
    if PREFERENCES.exists():
        with open(PREFERENCES, 'rb') as f:
            prefs = pickle.load(f)
            for key, value in default_preferences.items():
                if key not in prefs: prefs[key] = value
            return prefs
    return default_preferences.copy()

def save_preferences(prefs):
    with open(PREFERENCES, 'wb') as f:
        pickle.dump(prefs, f)

def load_whisper_model(model_name):
    global model, current_model_name
    if model_name not in AVAILABLE_WHISPER_MODELS: model_name = "base"
    if model_name != current_model_name:
        log(f"Carregando modelo Whisper: {model_name}")
        try:
            model = whisper.load_model(model_name, device=device)
            current_model_name = model_name
            log(f"✓ Modelo {model_name} carregado com sucesso")
        except Exception as e:
            log(f"✗ Erro ao carregar modelo {model_name}: {e}")

def save_to_history(video_name, session_id, clips, analytics):
    try:
        history = json.load(open(HISTORY, encoding="utf-8")) if HISTORY.exists() else []
        history.insert(0, {
            "video_name": video_name, "session_id": session_id,
            "date": datetime.now().isoformat(), "clips_count": len(clips),
            "avg_score": analytics.get("avg_score", 0),
            "clips": [{"file": c} for c in clips]
        })
        with open(HISTORY, "w", encoding="utf-8") as f:
            json.dump(history[:50], f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"Erro ao salvar histórico: {e}")

def get_history():
    if HISTORY.exists():
        with open(HISTORY, "r", encoding="utf-8") as f: return json.load(f)
    return []

def clear_temp_folders():
    for folder in [UPLOAD, TEMP]:
        if folder.exists():
            for filename in os.listdir(folder):
                file_path = folder / filename
                try:
                    if file_path.is_file(): file_path.unlink()
                    elif file_path.is_dir(): shutil.rmtree(file_path)
                except Exception as e:
                    log(f"Não foi possível limpar {file_path}: {e}")

def update_status(session_id, stage, progress, log_message=None):
    with status_lock:
        if session_id not in processing_status:
            processing_status[session_id] = {
                "stage": "", "progress": 0, "timestamp": time.time(), "logs": []
            }
        
        processing_status[session_id]["stage"] = stage
        processing_status[session_id]["progress"] = min(progress, 100)
        processing_status[session_id]["timestamp"] = time.time()
        
        if log_message:
            processing_status[session_id]["logs"].append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": log_message
            })

def analyze_sentiment_from_audio(audio_path):
    try:
        y, sr = librosa.load(str(audio_path), sr=22050, duration=60)
        energy = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = sr * 60.0 / np.mean(np.diff(tempo)) if len(tempo) > 1 else 120.0
        sentiment, confidence = "NEUTRO", 0.5
        if energy > 0.05 and tempo_val > 120: sentiment, confidence = "URGENTE", 0.85
        elif energy < 0.02 and tempo_val < 100: sentiment, confidence = "SERIO", 0.75
        elif zcr > 0.1: sentiment, confidence = "ALERTA", 0.70
        elif spectral_centroid > 2000: sentiment, confidence = "POSITIVO", 0.65
        log(f"Sentimento: {sentiment} ({confidence:.2f})")
        return {"sentiment": sentiment, "confidence": confidence, "energy": float(energy), "tempo": float(tempo_val)}
    except Exception as e:
        log(f"Erro na análise de sentimento: {e}")
        return {"sentiment": "NEUTRO", "confidence": 0.5}

def detect_faces_mediapipe(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = face_detector.process(rgb_frame)
    faces = []
    if results.detections:
        h, w = frame.shape[:2]
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
            faces.append({'center_x': x + width / 2, 'confidence': detection.score[0], 'bbox': (x, y, width, height)})
    return faces

def detect_faces_haar(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    return [{'center_x': x + w / 2, 'confidence': 0.7, 'bbox': (x, y, w, h)} for (x, y, w, h) in detected]

def smart_crop(clip, zoom_factor=1.3):
    w, h = clip.size
    target_w, target_h = 1080, 1920
    target_ratio = target_w / target_h
    crop_width = min(w, max(int(h * target_ratio / zoom_factor), 800))
    log(f"Detectando rostos - Video: {w}x{h}, Crop: {crop_width}px")
    sample_times = np.linspace(0, min(clip.duration, 120), 30)
    face_positions = []
    for t in sample_times:
        try:
            frame = clip.get_frame(t)
            faces = detect_faces_mediapipe(frame) if USE_MEDIAPIPE else detect_faces_haar(frame)
            best_face = max(faces, key=lambda f: f['confidence']) if faces else {'center_x': w / 2}
            face_positions.append((t, best_face['center_x']))
        except Exception:
            face_positions.append((t, w / 2))
    if len(face_positions) > 3:
        times, positions = np.array([fp[0] for fp in face_positions]), np.array([fp[1] for fp in face_positions])
        window = min(7, len(positions))
        smoothed = np.convolve(positions, np.ones(window)/window, mode='same')
        smoothed = np.clip(smoothed, crop_width / 2, w - crop_width / 2)
    else:
        times, smoothed = np.array([0, clip.duration]), np.array([w/2, w/2])
    def apply_dynamic_crop(get_frame, t):
        frame = get_frame(t)
        center_x = np.interp(t, times, smoothed)
        x1 = int(center_x - crop_width / 2)
        x1 = max(0, min(x1, w - crop_width))
        x2 = x1 + crop_width
        return frame[:, x1:x2]
    cropped_clip = clip.fl(apply_dynamic_crop)
    new_h = int(crop_width / target_ratio)
    if new_h <= h:
        y_start = (h - new_h) // 2
        cropped_clip = cropped_clip.crop(y1=y_start, y2=y_start + new_h)
    return cropped_clip.resize((target_w, target_h))

def group_words_for_subtitles(words_data, t0, words_per_group=3):
    subtitle_groups = []
    for i in range(0, len(words_data), words_per_group):
        group = words_data[i:i+words_per_group]
        if not group: continue
        text = " ".join([w['word'].strip() for w in group]).upper()
        start_time = (group[0]['start'] - t0)
        end_time = (group[-1]['end'] - t0)
        if start_time < 0: continue
        subtitle_groups.append({'text': text, 'start': start_time, 'end': end_time, 'duration': end_time - start_time})
    return subtitle_groups

def classify_clip_narrative(clips):
    clips = sorted(clips, key=lambda x: x['start'])
    if len(clips) > 0: clips[0]['narrative'] = "INTRODUCAO"
    if len(clips) > 1:
        climax_clip = max(clips[1:], key=lambda x: x['score'])
        climax_clip['narrative'] = "CLIMAX"
    for clip in clips:
        if 'narrative' not in clip:
            clip['narrative'] = "CONTEXTO"
    return sorted(clips, key=lambda x: ['INTRODUCAO', 'CONTEXTO', 'CLIMAX'].index(x.get('narrative', 'CONTEXTO')))

def generate_smart_caption(text_content, sentiment):
    hooks = SMART_CAPTION_HOOKS.get(sentiment.get("sentiment", "NEUTRO"), SMART_CAPTION_HOOKS["NEUTRO"])
    hook = random.choice(hooks)
    sentences = text_content.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() for s in sentences if len(s.split()) > 5]
    best_sentence = sentences[0] if sentences else ""
    if len(sentences) > 1:
        max_score = -1
        for s in sentences[:5]:
            score = sum(weight for keyword, weight in KEYWORDS_SCORING.items() if keyword in s.lower())
            if score > max_score:
                max_score = score
                best_sentence = s
    summary = best_sentence[:220] + "..." if len(best_sentence) > 220 else best_sentence
    cta = random.choice(SMART_CAPTION_CTA)
    return f"{hook}\n\n{summary}.\n\n{cta}"

def generate_strategic_report(score, narrative_type, sentiment_data, clip_data, text_content):
    engagement_score = "ALTO" if score > 70 else "MÉDIO" if score > 40 else "BAIXO"
    best_time = "18h-21h" if sentiment_data.get("sentiment") == "URGENTE" else "12h-14h"
    platforms = []
    if clip_data.get('duration', 0) < 60: platforms.extend(["Instagram Reels", "TikTok"])
    if clip_data.get('duration', 0) < 180: platforms.extend(["YouTube Shorts", "Facebook"])
    words = text_content.lower().split()
    hashtags = set([f"#{word.capitalize()}" for word in words if len(word) > 6 and word.isalpha()])
    final_hashtags = " ".join(list(hashtags)[:5])
    return f"""📊 MÉTRICAS DE PERFORMANCE:
• Score de Relevância: {score}/100
• Potencial de Engajamento: {engagement_score}
• Categoria Narrativa: {narrative_type}
• Duração: {clip_data.get('duration', 0):.1f}s
• Ritmo: {len(text_content.split())/clip_data.get('duration', 1):.1f} palavras/seg

🎭 ANÁLISE DE SENTIMENTO:
• Tom Detectado: {sentiment_data.get('sentiment', 'NEUTRO')}
• Confiança: {sentiment_data.get('confidence', 0)*100:.0f}%
• Energia do Áudio: {sentiment_data.get('energy', 0)*100:.1f}%
• Tempo (BPM): {sentiment_data.get('tempo', 0):.0f}

🎯 RECOMENDAÇÕES DE PUBLICAÇÃO:
• Melhor Horário: {best_time}
• Plataformas Ideais: {', '.join(platforms)}
• Tipo de Público: {"Amplo" if score > 60 else "Segmentado"}

#️⃣ HASHTAGS ESTRATÉGICAS:
{final_hashtags} {DEFAULT_HASHTAGS}

💡 DICA: Use este conteúdo para iniciar conversas nas redes sociais e direcionar tráfego para o conteúdo completo no Globoplay.

Gerado automaticamente por GLOBINHO AI 🤖"""

def generate_social_media_post_file(text_content, sentiment_data):
    title = text_content.split('.')[0][:100].upper()
    smart_caption = generate_smart_caption(text_content, sentiment_data)
    return f"""📝 TÍTULO SUGERIDO:
{title}

📱 LEGENDA PARA REDES SOCIAIS:
{smart_caption}

👉 Assista ao conteúdo completo no Globoplay!"""

def save_analytics(clips_data, session_id, sentiment_data):
    try:
        data = json.load(open(ANALYTICS, encoding="utf-8")) if ANALYTICS.exists() else {
            "total_clips": 0, "keywords": [], "durations": [], "avg_score": 0,
            "sessions": [], "narratives": {"INTRODUCAO": 0, "CONTEXTO": 0, "CLIMAX": 0},
            "sentiments": {}, "total_duration": 0, "avg_duration": 0,
            "score_distribution": {"0-30": 0, "31-50": 0, "51-70": 0, "71-100": 0},
            "duration_by_narrative": {"INTRODUCAO": [], "CONTEXTO": [], "CLIMAX": []}
        }
        
        session_scores, session_duration = [], 0
        
        for clip in clips_data:
            data["total_clips"] += 1
            data["durations"].append(clip["duration"])
            data["total_duration"] += clip["duration"]
            session_duration += clip["duration"]
            session_scores.append(clip["score"])
            
            score = clip["score"]
            if score <= 30:   data["score_distribution"]["0-30"] += 1
            elif score <= 50: data["score_distribution"]["31-50"] += 1
            elif score <= 70: data["score_distribution"]["51-70"] += 1
            else:             data["score_distribution"]["71-100"] += 1
            
            narrative = clip.get("narrative", "CONTEXTO")
            if narrative in data["narratives"]: data["narratives"][narrative] += 1
            if narrative in data["duration_by_narrative"]:
                data["duration_by_narrative"][narrative].append(clip["duration"])
            
            words = clip["text"].lower().split()
            data["keywords"].extend([w for w in words if len(w) > 5])
        
        if data["durations"]: data["avg_duration"] = sum(data["durations"]) / len(data["durations"])
        
        sent = sentiment_data.get("sentiment", "NEUTRO")
        data["sentiments"][sent] = data["sentiments"].get(sent, 0) + 1
        
        keyword_count = Counter(data["keywords"])
        top_keywords = list(dict(keyword_count.most_common(10)).keys())
        
        data["sessions"].append({
            "id": session_id[:8], "date": datetime.now().isoformat(), "clips": len(clips_data),
            "avg_score": sum(session_scores) / len(session_scores) if session_scores else 0,
            "total_duration": session_duration, "top_keywords": top_keywords, "sentiment": sent
        })
        
        all_scores = [s["avg_score"] for s in data["sessions"]]
        data["avg_score"] = sum(all_scores) / len(all_scores) if all_scores else 0
        
        with open(ANALYTICS, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return data
    except Exception as e:
        log(f"Erro ao salvar analytics: {e}")
        return {}

def hms_to_seconds(t):
    try:
        parts = list(map(int, t.split(':')))
        if len(parts) == 3:
            h, m, s = parts
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = parts
            return m * 60 + s
        elif len(parts) == 1:
            return parts[0]
    except:
        return None
    return None

def process_video_thread(path, session_id, preferences):
    global is_processing
    try:
        result = process(path, session_id, preferences)
        with status_lock:
            processing_status[session_id]["result"] = result
            processing_status[session_id]["done"] = True
    except Exception as e:
        log(f"ERRO no processamento: {e}")
        import traceback
        traceback.print_exc()
        with status_lock:
            processing_status[session_id]["error"] = str(e)
            processing_status[session_id]["done"] = True
    finally:
        with processing_lock:
            is_processing = False
            log("Fila de processamento liberada.")

def process(path, session_id, preferences):
    name = path.stem
    final_files = []
    MIN_DURATION, MAX_DURATION = preferences.get('min_duration', 30), preferences.get('max_duration', 120)
    
    NUM_BEST_CLIPS = int(preferences.get('num_clips', 3))
    
    video_speed_raw = preferences.get('video_speed')
    VIDEO_SPEED = float(video_speed_raw) if video_speed_raw else 1.0

    WHISPER_MODEL_NAME = preferences.get('whisper_model', 'base')
    FAST_MODE = preferences.get('fast_mode', False)
    WATERMARK_PATH = preferences.get('watermark_path', '')
    WITH_SUBTITLES = preferences.get('with_subtitles', True)
    
    start_time_str = preferences.get('start_time')
    end_time_str = preferences.get('end_time')

    load_whisper_model(WHISPER_MODEL_NAME)
    start_time = time.time()
    clip = None
    
    try:
        update_status(session_id, "Analisando mídia", 5, "📹 Carregando vídeo...")
        clip = mpy.VideoFileClip(str(path))
        target_duration = 300 if FAST_MODE and clip.duration > 300 else None
        
        update_status(session_id, "Extraindo audio", 10, "🎵 Extraindo áudio do vídeo...")
        audio_path = TEMP / f"audio_{session_id}.wav"
        if target_duration:
            with clip.subclipped(0, target_duration) as sub_clip:
                sub_clip.audio.write_audiofile(str(audio_path), codec='pcm_s16le', fps=16000, logger=None)
        else:
            clip.audio.write_audiofile(str(audio_path), codec='pcm_s16le', fps=16000, logger=None)
        log(f"⏱️ Extração de áudio: {time.time() - start_time:.2f}s")
        
        update_status(session_id, "Analisando sentimento", 15, "🎭 Analisando sentimento do áudio...")
        sentiment_data = analyze_sentiment_from_audio(audio_path)
        
        update_status(session_id, "IA transcrevendo", 30, f"🤖 Transcrevendo com Whisper {WHISPER_MODEL_NAME}...")
        with torch.no_grad():
            result = model.transcribe(str(audio_path), language="pt", verbose=False, word_timestamps=True)
        log(f"⏱️ Transcrição (Whisper {WHISPER_MODEL_NAME}): {time.time() - start_time:.2f}s")
        
        all_segments = result.get("segments", [])
        if not all_segments: 
            raise ValueError("Transcrição não retornou segmentos. Verifique se o áudio contém fala.")
        
        if start_time_str and end_time_str:
            log("Modo de corte: MANUAL")
            update_status(session_id, "Processando corte manual", 40, "✂️ Aplicando corte manual selecionado...")
            start_sec = hms_to_seconds(start_time_str)
            end_sec = hms_to_seconds(end_time_str)

            if start_sec is None or end_sec is None or start_sec >= end_sec:
                raise ValueError("Tempo de início ou fim do corte manual é inválido.")
            
            manual_segments = [s for s in all_segments if s['start'] < end_sec and s['end'] > start_sec]
            manual_text = " ".join(s['text'] for s in manual_segments).strip()
            
            if not manual_text:
                raise ValueError("Nenhuma fala detectada no intervalo selecionado.")
            
            top_clips = [{
                "start": start_sec, "end": end_sec, "segments": manual_segments,
                "text": manual_text, "duration": end_sec - start_sec, "score": 99,
                "narrative": "MANUAL"
            }]
        else:
            log("Modo de corte: AUTOMÁTICO (IA)")
            update_status(session_id, "Analisando trechos", 50, "🔍 IA analisando os melhores momentos do vídeo...")
            potential_clips, current_clip_segments = [], []
            
            for segment in all_segments:
                if not current_clip_segments or (segment['start'] - current_clip_segments[-1]['end']) > 3.0 or (segment['end'] - current_clip_segments[0]['start']) > MAX_DURATION:
                    if current_clip_segments:
                        start, end = current_clip_segments[0]['start'], current_clip_segments[-1]['end']
                        duration = end - start
                        if duration >= MIN_DURATION:
                            full_text = " ".join([s['text'] for s in current_clip_segments]).strip()
                            potential_clips.append({"start": start, "end": end, "segments": current_clip_segments, "text": full_text, "duration": duration, "score": 0})
                    current_clip_segments = [segment]
                else:
                    current_clip_segments.append(segment)
            
            if current_clip_segments:
                start, end = current_clip_segments[0]['start'], current_clip_segments[-1]['end']
                duration = end - start
                if duration >= MIN_DURATION:
                    full_text = " ".join([s['text'] for s in current_clip_segments]).strip()
                    potential_clips.append({"start": start, "end": end, "segments": current_clip_segments, "text": full_text, "duration": duration, "score": 0})
            
            if not potential_clips: 
                raise ValueError("Nenhum clipe com a duração mínima foi encontrado. Tente ajustar as configurações.")
            
            for p_clip in potential_clips:
                text_lower = p_clip["text"].lower()
                score = sum(text_lower.count(k) * w for k, w in KEYWORDS_SCORING.items())
                words_per_second = len(p_clip["text"].split()) / p_clip["duration"]
                if words_per_second > 2.5: score += 15
                if 40 <= p_clip["duration"] <= 90: score += 10
                if sentiment_data.get('sentiment') == 'URGENTE': score += 10
                p_clip["score"] = score
            
            top_clips = sorted(potential_clips, key=lambda c: c['score'], reverse=True)[:NUM_BEST_CLIPS]
            top_clips = classify_clip_narrative(top_clips.copy())
            log(f"⏱️ Análise de trechos: {time.time() - start_time:.2f}s")
        
        update_status(session_id, f"Renderizando {len(top_clips)} clipes", 60, f"🎬 Iniciando renderização de {len(top_clips)} clipes...")
        
        for idx, p_clip in enumerate(top_clips, 1):
            progress = 60 + (idx / len(top_clips)) * 35
            update_status(session_id, f"Renderizando clipe {idx}/{len(top_clips)}", progress, f"⚙️ Renderizando clipe {idx} de {len(top_clips)}...")
            render_start_time = time.time()
            
            part, video_part_speed, part_vertical, final_video_part, final_part = [None] * 5
            txt_clips = []
            
            try:
                # CORREÇÃO AQUI: subclip -> subclipped
                part = clip.subclip(p_clip['start'], p_clip['end'])
                video_part_speed = part.fx(vfx.speedx, VIDEO_SPEED)
                part_vertical = smart_crop(video_part_speed.without_audio())
                
                # Lista de clips para composição
                composition_clips = [part_vertical]
                
                # Adicionar legendas se ativado
                if WITH_SUBTITLES:
                    words_in_clip = [word for seg in p_clip['segments'] if 'words' in seg for word in seg['words']]
                    subtitle_groups = group_words_for_subtitles(words_in_clip, p_clip['start'])
                    
                    for sub_group in subtitle_groups:
                        start_time_adjusted = sub_group['start'] / VIDEO_SPEED
                        duration_adjusted = sub_group['duration'] / VIDEO_SPEED
                        if start_time_adjusted >= part_vertical.duration: continue
                        
                        txt_clip = mpy.TextClip(
                            sub_group['text'], 
                            fontsize=preferences.get('subtitle_size', 70), 
                            color='yellow', 
                            font=preferences.get('subtitle_font', 'Courier-New-Bold'), 
                            stroke_color='black', 
                            stroke_width=3, 
                            bg_color='rgba(0,0,0,0.7)', 
                            method='caption', 
                            size=(part_vertical.w * 0.9, None)
                        )
                        txt_clip = txt_clip.set_duration(duration_adjusted).set_start(start_time_adjusted).set_position(('center', 0.8), relative=True)
                        txt_clips.append(txt_clip)
                        composition_clips.append(txt_clip)
                
                # Adicionar marca d'água se fornecida
                if WATERMARK_PATH and Path(WATERMARK_PATH).exists():
                    try:
                        watermark = mpy.ImageClip(WATERMARK_PATH)
                        # Redimensionar marca d'água para 15% da largura do vídeo
                        watermark_width = int(part_vertical.w * 0.15)
                        watermark = watermark.resize(width=watermark_width)
                        # Posicionar no canto superior direito com margem
                        watermark = watermark.set_duration(part_vertical.duration)
                        watermark = watermark.set_position(('right', 'top')).margin(right=20, top=20, opacity=0)
                        watermark = watermark.set_opacity(0.7)
                        composition_clips.append(watermark)
                        log(f"✓ Marca d'água adicionada: {WATERMARK_PATH}")
                    except Exception as e:
                        log(f"⚠ Erro ao adicionar marca d'água: {e}")

                final_video_part = mpy.CompositeVideoClip(composition_clips)
                final_part = final_video_part.set_audio(part.audio)

                out_name = f"{name}_{session_id[:8]}_corte{idx}"
                out_path_mp4 = DONE / f"{out_name}.mp4"

                ffmpeg_params = []
                if VIDEO_SPEED != 1.0:
                    atempo_filter = f"atempo={VIDEO_SPEED}"
                    if VIDEO_SPEED > 2.0: atempo_filter = f"atempo=2.0,atempo={VIDEO_SPEED/2.0}"
                    ffmpeg_params.extend(['-af', atempo_filter])

                final_part.write_videofile(
                    str(out_path_mp4), codec="libx264", audio_codec="aac", fps=30,
                    verbose=False, logger=None, threads=os.cpu_count(), preset='medium',
                    ffmpeg_params=ffmpeg_params
                )
                final_files.append(out_path_mp4.name)

                post_content = generate_social_media_post_file(p_clip['text'], sentiment_data)
                (DONE / OUTPUT_FILENAME_FORMAT["post"].format(base_name=out_name)).write_text(post_content, encoding="utf-8")
                
                report_content = generate_strategic_report(p_clip['score'], p_clip.get('narrative', ''), sentiment_data, p_clip, p_clip['text'])
                (DONE / OUTPUT_FILENAME_FORMAT["analysis"].format(base_name=out_name)).write_text(report_content, encoding="utf-8")
                
                log(f"⏱️ Renderização clipe {idx}: {time.time() - render_start_time:.2f}s")
            
            except Exception as e:
                log(f"Erro ao renderizar clipe {idx}: {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                clips_to_close = [part, video_part_speed, part_vertical, final_video_part, final_part] + txt_clips
                for c in clips_to_close:
                    if c:
                        try: c.close()
                        except: pass
        
        if not final_files:
            raise ValueError("Nenhum clipe foi gerado com sucesso. Verifique o vídeo e as configurações.")
        
        analytics_data = save_analytics(top_clips, session_id, sentiment_data)
        save_to_history(path.name, session_id, [f for f in final_files if f.endswith('.mp4')], analytics_data)
        update_status(session_id, "Concluido", 100, "✅ Processamento concluído com sucesso!")
        
        return {
            "files": [f for f in final_files if f.endswith('.mp4')], "session_id": session_id,
            "analytics": {
                "total_clips": len(top_clips),
                "avg_score": sum(c['score'] for c in top_clips) / len(top_clips) if top_clips else 0,
                "total_duration": sum(c['duration'] for c in top_clips),
                "sentiment": sentiment_data, "processing_time": time.time() - start_time
            }
        }
    
    except Exception as e:
        log(f"ERRO: {e}")
        import traceback
        traceback.print_exc()
        return {"files": [], "session_id": session_id, "error": str(e)}
    
    finally:
        if clip:
            clip.close()

# ==============================================================================
# ROTAS FLASK E EXECUÇÃO
# ==============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global is_processing
    with processing_lock:
        if is_processing:
            log("⚠️ Tentativa de upload enquanto um processamento está em andamento.")
            return jsonify({"error": "Um vídeo já está sendo processado. Por favor, aguarde o término."}), 429
        is_processing = True
        log("Iniciando novo processamento, bloqueando a fila.")

    clear_temp_folders()
    session_id = str(uuid.uuid4())
    # DEPOIS (Com a correção)
    processing_status[session_id] = {
        "stage": "Iniciando",
        "progress": 0,
        "done": False, 
        "logs": [] # <-- CORREÇÃO AQUI
    }
    
    
    if 'video' not in request.files or not request.files['video'].filename:
        with processing_lock:
            is_processing = False
        return jsonify({"error": "Nenhum arquivo de vídeo enviado"}), 400
    
    file = request.files['video']
    preferences = json.loads(request.form.get('preferences', '{}')) or load_preferences()
    filename = secure_filename(file.filename)
    save_path = UPLOAD / filename
    file.save(save_path)
    log(f"UPLOAD: {filename}")
    
    thread = threading.Thread(
        target=process_video_thread, 
        args=(save_path, session_id, preferences)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({"session_id": session_id})

@app.route('/upload-watermark', methods=['POST'])
def upload_watermark():
    """Upload de marca d'água"""
    if 'watermark' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400
    
    file = request.files['watermark']
    if file.filename == '':
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400
    
    # Salvar marca d'água
    filename = secure_filename(file.filename)
    watermark_path = TEMP / f"watermark_{filename}"
    file.save(watermark_path)
    
    return jsonify({"path": str(watermark_path), "success": True})

@app.route('/status/<session_id>')
def get_status(session_id):
    with status_lock:
        status = processing_status.get(session_id)
    if not status:
        return jsonify({"error": "Sessao nao encontrada"}), 404
    return jsonify(status)

@app.route('/preview/<session_id>')
def preview_video(session_id):
    """Retorna informações do vídeo para preview"""
    video_files = list(UPLOAD.glob(f"*.mp4")) + list(UPLOAD.glob(f"*.mov")) + list(UPLOAD.glob(f"*.avi"))
    if video_files:
        video_path = video_files[0]
        try:
            clip = mpy.VideoFileClip(str(video_path))
            duration = clip.duration
            fps = clip.fps
            size = clip.size
            clip.close()
            return jsonify({
                "filename": video_path.name,
                "duration": duration,
                "fps": fps,
                "width": size[0],
                "height": size[1]
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Video nao encontrado"}), 404

@app.route('/video/<filename>')
def serve_video(filename):
    """Serve o vídeo para preview"""
    return send_file(UPLOAD / filename)

@app.route('/download-all/<session_id>')
def download_all(session_id):
    result = processing_status.get(session_id, {}).get("result", {})
    files = result.get("files", [])
    
    if not files:
        return "Nenhum arquivo encontrado para esta sessão.", 404
    
    zip_path = DONE / f"globinho_{session_id[:8]}.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for fname in files:
            base_name = fname.replace('.mp4', '')
            for ext in ['.mp4', '_post.txt', '_analise.txt']:
                fpath = DONE / (base_name + ext)
                if fpath.exists():
                    zf.write(fpath, arcname=fpath.name)
    
    return send_file(zip_path, as_attachment=True)

@app.route('/done/<filename>')
def download_file_route(filename):
    return send_file(DONE / filename, as_attachment=True)

@app.route('/analytics')
def view_analytics():
    if not ANALYTICS.exists():
        return jsonify({
            "total_clips": 0, "sessions": [], "narratives": {}, 
            "sentiments": {}, "keywords": [], "avg_score": 0, 
            "avg_duration": 0, 
            "score_distribution": {
                "0-30": 0, "31-50": 0, "51-70": 0, "71-100": 0
            }, 
            "duration_by_narrative": {
                "INTRODUCAO": [], "CONTEXTO": [], "CLIMAX": []
            }
        })
    return jsonify(json.load(open(ANALYTICS, encoding="utf-8")))

@app.route('/history')
def view_history():
    return jsonify(get_history())

@app.route('/save-preferences', methods=['POST'])
def save_prefs():
    try:
        save_preferences(request.json)
        return jsonify({"success": True})
    except:
        return jsonify({"error": "Erro ao salvar preferências"}), 500

@app.route('/get-preferences')
def get_prefs():
    return jsonify(load_preferences())

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)