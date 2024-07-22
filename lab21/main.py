''' Этот код демонстрирует базовую структуру приложения FastAPI для машинного обучения и веб-разработки.
    Он включает импорт необходимых библиотек и фреймворков, а также настройку приложения.'''

from fastapi import FastAPI, Request, File, UploadFile  # Основные компоненты FastAPI для создания API и работы с файлами
from fastapi.responses import HTMLResponse  # Класс для ответов в виде HTML
from fastapi.templating import Jinja2Templates  # Утилита для рендеринга HTML-шаблонов
from fastapi.middleware.cors import CORSMiddleware  # Middleware для управления политикой CORS
from fastapi.staticfiles import StaticFiles
import uvicorn  # ASGI сервер для запуска приложения
import io  # Ввод-вывод для работы с потоками
import os  # Работа с операционной системой, например, для доступа к переменным окружения
import base64  # Кодирование и декодирование данных в base64
from PIL import Image  # Работа с изображениями
from keras.models import load_model  # Загрузка предобученных моделей Keras
import numpy as np  # Работа с массивами
from typing import Dict  # Типизация для словарей
from pydantic import BaseModel  # Создание моделей данных для валидации
from PIL import ImageOps  # Работа с операциями над изображениями
from fastapi.responses import StreamingResponse  # Класс для потоковой передачи ответов


app = FastAPI()
# Здесь 'directory' должен указывать на папку, где находятся static файлы
app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")

# Разрешаем запросы CORS от любого источника
origins = ["*"]  # Для простоты можно разрешить доступ со всех источников
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_mnist = load_model('static/MnistConv2D.h5')
model_segment = load_model('static/segment.h5')

@app.get("/")
@app.get("/index.html", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.get("/mnist.html", response_class=HTMLResponse)
async def mnist(request: Request):
    return templates.TemplateResponse("mnist.html", {"request": request})

@app.get("/segment.html", response_class=HTMLResponse)
async def mnist(request: Request):
    return templates.TemplateResponse("segment.html", {"request": request})

@app.get("/tseries.html", response_class=HTMLResponse)
async def mnist(request: Request):
    return templates.TemplateResponse("tseries.html", {"request": request})


@app.post("/mnist_predict")
async def predict(data: Dict[str, str]):
    # Декодируем данные изображения из формата base64
    image_data = base64.b64decode(data['image_data'])
    # Преобразуем данные в объект изображения
    image = Image.open(io.BytesIO(image_data))
    # Преобразуем изображение в массив NumPy и изменяем его размер
    image = image.convert('L').resize((28, 28))
    # Нормализуем значения пикселей изображения
    image = np.array(image)
    # Разворачиваем массив в одномерный вектор
    processed_image = np.expand_dims(image, axis=0)
    # Используем модель для предсказания цифры
    predictions = model_mnist.predict(processed_image)
    # Получаем номер цифры с наибольшей вероятностью
    digit = np.argmax(predictions)
    # Возвращаем предсказанную цифру
    return {"digit": int(digit)}


def labels_to_rgb(image_list
                 ):
    AIRPLANE = (255, 255, 255)      # Самолет (белый)
    BACKGROUND = (0, 0, 0)          # Фон (черный)
    CLASS_LABELS = (AIRPLANE, BACKGROUND)
    result = []
    for y in image_list:
        temp = np.zeros((256, 456, 3), dtype='uint8')
        for i, cl in enumerate(CLASS_LABELS):
            temp[np.where(np.all(y==i, axis=-1))] = CLASS_LABELS[i]
        result.append(temp)  
    return np.array(result)
    

@app.post('/segment_predict')
async def segment_predict(image: UploadFile = File(...)):
    img_or = Image.open(image.file).resize((456, 256))
    img = np.array(img_or)
    processed_image = np.expand_dims(img, axis=0) 
    predict = np.argmax(model_segment.predict(processed_image), axis=-1)    
    mask = labels_to_rgb(predict[..., None])[0]
    # Создание канала альфа-слоя для изображения    
    alpha = Image.fromarray(mask).convert('L').point(lambda x: 0 if x == 0 else 255)
    # Применение маски к изображению
    result = img_or.copy()
    result.putalpha(alpha)

    # Создание нового изображения и вставка на него оригинального изображения и изображения с наложенной маской
    new_img = Image.new('RGBA', (img_or.width*2, img_or.height), (0, 0, 0, 0))
    new_img.paste(Image.fromarray(mask), (0, 0))
    new_img.paste(result, (img_or.width, 0))
    
    
    buffer_inverted = io.BytesIO()
    new_img.save(buffer_inverted, format="PNG")
    buffer_inverted.seek(0)

    return StreamingResponse(buffer_inverted, media_type="image/png")



if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=int(os.environ.get('PORT', 8000)))
