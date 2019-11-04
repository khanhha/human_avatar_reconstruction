FROM human:lib 
COPY /environment.yml /tmp
COPY /src /tmp/src/
COPY /third_parties /tmp/third_parties
COPY /web_body /tmp/web_body
WORKDIR /tmp
RUN pwd
RUN ls

RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get -y install cmake
RUN apt-get -y install libblas-dev liblapack-dev xorg-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev

ENV PYTHONPATH "${PYTHONPATH}:/tmp/src/"
ENV PYTHONPATH "${PYTHONPATH}:/tmp/third_parties/"

WORKDIR /
WORKDIR /tmp/third_parties/tf-pose-estimation/
RUN python setup.py install

WORKDIR /
WORKDIR /tmp/third_parties/libigl/build
RUN cmake .. -DLIBIGL_WITH_EMBREE=OFF 
RUN make

WORKDIR /
WORKDIR /tmp/web_body

CMD ["python", "manage.py", "makemigrations"]
CMD ["python", "manage.py", "migrate"]

CMD ["python","manage.py", "runserver", "3000"]
