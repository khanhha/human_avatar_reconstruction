FROM continuumio/anaconda3
COPY /environment_cpu.yml /tmp
WORKDIR /tmp
#RUN conda env create --prefix ./envs --file ./environment.yml
RUN conda env create --name hmenv -f environment_cpu.yml
RUN echo "source activate hmenv" > ~/.bashrc
ENV PATH /opt/conda/envs/hmenv/bin:$PATH

RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get -y install cmake
RUN apt-get -y install libblas-dev liblapack-dev xorg-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev
#this line is for the bug missing glib.so libpthread
#https://github.com/matplotlib/matplotlib/issues/11058
RUN apt-get -y install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
COPY /third_parties /tmp/third_parties

WORKDIR /
WORKDIR /tmp/third_parties/tf-pose-estimation/
RUN python setup.py install

WORKDIR /
WORKDIR /tmp/third_parties/libigl/build
RUN cmake .. -DLIBIGL_WITH_EMBREE=OFF 
RUN make

WORKDIR /
COPY /google_auth /tmp/google_auth
WORKDIR /tmp/google_auth/
#download folder cnn_cnn_data from google drive to /tmp/cnn_run_data
RUN python ./download_run_data.py cnn_run_data ../

WORKDIR /
COPY /src /tmp/src/
COPY /web_body /tmp/web_body
WORKDIR /tmp/web_body

ENV PYTHONPATH "${PYTHONPATH}:/tmp/src/"
ENV PYTHONPATH "${PYTHONPATH}:/tmp/third_parties/"

RUN apt-get install -y xvfb

EXPOSE 8000
ENV DISPLAY :1
CMD ["./run_webapp.sh"]
