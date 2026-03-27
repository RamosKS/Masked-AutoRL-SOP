FROM continuumio/miniconda3

WORKDIR /app

COPY . /app

RUN conda env create -f environment.yml

ENV PATH=/opt/conda/envs/rl/bin:$PATH

RUN pip install notebook ipykernel

EXPOSE 8888

CMD ["python", "Masked AutoRL-SOP.py"]