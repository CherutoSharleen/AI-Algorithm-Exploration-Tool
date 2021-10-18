FROM python:3.9.7
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8501
COPY . /app
ENTRYPOINT ["streamlit", "run"]
CMD ["explore.py"]
