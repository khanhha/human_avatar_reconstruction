FROM continuumio/anaconda3
COPY /environment.yml /tmp
COPY /src /tmp
COPY /third_parties /tmp
COPY /web_body /tmp
WORKDIR /tmp
RUN pwd
RUN conda env create --prefix /envs --file /environment.yml
RUN conda activate /tmp/envs 

