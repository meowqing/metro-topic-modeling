from fastapi import FastAPI, HTTPException, UploadFile, File
from pathlib import Path
import joblib
import pandas as pd
import io
import uvicorn
from typing import Dict, List
from collections import Counter

# Импорт препроцессинга
import sys
sys.path.insert(0, str(Path(__file__).parent))
import preprocessing
preprocess_func = preprocessing.preprocess_text

# Инициализация FastAPI
app = FastAPI(
    title="Metro Text Classifier",
    description="Классификация текстов о метро с помощью BERTopic",
    version="1.0"
)

# Загрузка модели 
MODEL_PATH = Path(__file__).parent / "models" / "bertopic_model_fastapi.joblib"

artifacts = joblib.load(MODEL_PATH)
model = artifacts['model']
coherence = artifacts['coherence_c_v']
model_loaded = True

# Эндпоинты
@app.get("/health")
async def health_check() -> Dict:
    """Проверка состояния сервиса"""
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "topics_count": len(coherence) if model_loaded else 0
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    """Классификация текстов из CSV файла"""
    if not model_loaded:
        raise HTTPException(503, "Модель не загружена")
    try:
        # Чтение CSV файла
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Проверка колонки 'text'
        if 'text' not in df.columns:
            raise HTTPException(400, "Колонка 'text' не найдена")
        
        df = df.drop_duplicates(subset=['text'], keep='first')

        # Затем извлекаем тексты
        raw_texts = df['text'].fillna('').astype(str).tolist()
        raw_texts = [t.strip() for t in raw_texts if t.strip()]
        
        if not raw_texts:
            return {
                "status": "warning",
                "message": "Нет текстов для обработки"
            }
        

        
        # Препроцессинг
        processed_texts = []
        valid_indices = []
        
        for i, text in enumerate(raw_texts):
            processed = preprocess_func(text)
            if processed:
                processed_texts.append(processed)
                valid_indices.append(i)
        
        # Классификация
        results = []
        if processed_texts:
            topics, probs = model.transform(processed_texts)
            
            for idx, original_idx in enumerate(valid_indices):
                topic_num = int(topics[idx])
                
                # Расчет confidence
                if probs is not None and idx < len(probs):
                    if topic_num >= 0 and topic_num < len(probs[idx]):
                        confidence = float(probs[idx][topic_num])
                    else:
                        confidence = 0.5
                else:
                    confidence = 0.9
                
                result = {
                    "id": original_idx,
                    "text_preview": raw_texts[original_idx][:100],
                    "topic": topic_num,
                    "confidence": round(confidence, 3)
                }
                
                # Добавление когерентности темы
                if topic_num in coherence:
                    result["topic_coherence"] = round(coherence[topic_num], 3)
                
                results.append(result)
        
        # Статистика
        topic_counts = Counter([r["topic"] for r in results]) if results else {}
        
        return {
            "status": "success",
            "summary": {
                "file": file.filename,
                "total_texts": len(raw_texts),
                "processed_texts": len(processed_texts),
                "unique_topics": len(topic_counts),
                "topics_distribution": dict(topic_counts)
            },
            "results": results[:100]
        }
        
    except pd.errors.EmptyDataError:
        raise HTTPException(400, "Файл пуст")
    except pd.errors.ParserError:
        raise HTTPException(400, "Ошибка чтения файла")
    except Exception as e:
        raise HTTPException(500, f"Ошибка обработки: {str(e)}")


# Запуск
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)