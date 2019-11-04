FROM continuumio/anaconda3
COPY /environment.yml /tmp
WORKDIR /tmp
#RUN conda env create --prefix ./envs --file ./environment.yml
RUN conda env create --name hmenv -f environment.yml
RUN echo "source activate hmenv" > ~/.bashrc
ENV PATH /opt/conda/envs/hmenv/bin:$PATH
