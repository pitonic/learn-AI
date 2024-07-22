FROM python:3.8

WORKDIR /

COPY ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY ./main.py /main.py
COPY ./static/ /static
COPY ./templates/ /templates

CMD ["fastapi", "run", "/main.py"]
