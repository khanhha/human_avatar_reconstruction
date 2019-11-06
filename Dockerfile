FROM human:lib 
RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get -y install cmake
RUN apt-get -y install libblas-dev liblapack-dev xorg-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev

COPY /third_parties /tmp/third_parties

WORKDIR /
WORKDIR /tmp/third_parties/tf-pose-estimation/
RUN python setup.py install

WORKDIR /
WORKDIR /tmp/third_parties/libigl/build
RUN cmake .. -DLIBIGL_WITH_EMBREE=OFF 
RUN make

WORKDIR /
COPY /src /tmp/src/
COPY /web_body /tmp/web_body
COPY /google_auth /tmp/google_auth

RUN conda install -c conda-forge google-auth-oauthlib
RUN conda install -c conda-forge google-api-python-client

WORKDIR /tmp
RUN python /google_auth/download_run_data.py cnn_run_data ./

WORKDIR /tmp/web_body

ENV PYTHONPATH "${PYTHONPATH}:/tmp/src/"
ENV PYTHONPATH "${PYTHONPATH}:/tmp/third_parties/"

RUN apt-get install -y xvfb

EXPOSE 8000
CMD ["./run_webapp.sh"]
