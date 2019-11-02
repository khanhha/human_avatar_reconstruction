FROM continuumio/miniconda3
RUN conda create --name hmbody --file /requirements.txt