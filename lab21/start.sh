
python3 download.py

docker rm -f $(docker ps -a | awk '{print$1}')

docker build  -t fastapi:latest .

docker run -d -p 8000:8000 fastapi
