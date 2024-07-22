import gdown
import shutil
import os

# URL архивов для загрузки
urls = [
    'https://storage.yandexcloud.net/aiueducation/integration/lessons/l3/templates.zip',
    'https://storage.yandexcloud.net/aiueducation/integration/lessons/l3/static.zip'
]

for url in urls:
    # Загрузка файла
    file_name = gdown.download(url, None, quiet=True)
    
    # Распаковка архива в текущую директорию
    shutil.unpack_archive(file_name, '.')
    
    # Удаление архива после распаковки
    os.remove(file_name)

print("Архивы загружены, распакованы и удалены.")