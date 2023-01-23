# hash:sha256:bb1bc0c3f5d7de7247846b6b1792fa69f11cefb009537b783a306fcf7eb2ebf3
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    astropy==5.2.1 \
    datajoint==0.12.9 \
    dspca==1.0.3 \
    globus-sdk==3.15.1 \
    h5py==3.3.0 \
    matplotlib==3.6.3 \
    numpy==1.24.1 \
    nwb-conversion-tools==0.11.38 \
    nwbinspector==0.4.25 \
    opencv-python==4.7.0.68 \
    openpyxl==3.0.10 \
    pandas==1.5.3 \
    pillow==9.4.0 \
    pybpod-gui-api==1.8.2 \
    pynrrd==1.0.0 \
    pynwb==2.2.0 \
    scipy==1.10.0 \
    seaborn==0.11.0 \
    spikeinterface==0.96.1 \
    statannot==0.2.3 \
    statsmodels==0.13.5 \
    streamlit==1.17.0 \
    streamlit-aggrid==0.3.3 \
    tifffile==2023.1.23.1 \
    tqdm==4.64.1 \
    watchdog==2.2.1 \
    xlrd==2.0.1
