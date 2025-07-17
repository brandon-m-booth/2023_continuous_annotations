### Start with Ubuntu LTS
FROM ubuntu:24.04

### The main application directory
WORKDIR /app

### Copy local directories to the current workdir of the docker image
COPY ./dataset ./dataset
COPY ./figures ./figures
COPY ./scripts ./scripts
COPY ./requirements.txt ./requirements.txt
COPY ./LICENSE ./LICENSE

### OS updates
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y wget bzip2

### Install Anaconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/root/miniconda3/bin:$PATH
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda update conda -y
RUN conda update --all -y

### CUBES Annotation Tool Setup
# Python install
RUN conda install -c conda-forge python=3.11
RUN pip install -r requirements.txt

# Rscript install
RUN apt-get install r-base -y
RUN Rscript -e "install.packages(c('data.table', 'irr', 'optparse', 'psych'))"

# Start the app using serve command
ENTRYPOINT ["tail", "-f", "/dev/null"]
