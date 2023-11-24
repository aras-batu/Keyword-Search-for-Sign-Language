FROM pytorchignite/base:1.11.0-0.4.9

RUN mkdir /var/run/sshd



RUN apt-get update && \
    apt-get install openssh-server sssd -y && \
    apt-get install vim -y && \
    apt-get -y install --no-install-recommends  libglib2.0 \
                                                libsm6 \
                                                htop \
                                                libxext6 \
                                                tmux \
                                                libxrender-dev \
                                                libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

RUN conda install -y av==7.0.1 -c conda-forge

RUN pip install --no-cache-dir albumentations==0.5.2 \
                                         numpy==1.20.2 \
                                         opencv-python-headless==4.5.1.48 \
                                         torch-tb-profiler==0.1.0 \
                                         scikit-learn==0.24.1 \
                                         Pillow==8.4.0 \
                                         tqdm==4.60.0 \
                                         pandas==1.1.5 \
                                         lmdb==1.2.1 \
                                         transformers==4.5.1 \
                                         timm==0.5.4 \
                                         requests \
                                         tensorboard==2.5.0 \ 
                                         tokenizers==0.10.2 \
                                         wandb==0.12.10 \
                                         seaborn==0.11.2 \ 
                                         nlpaug==1.1.4  \
                                         nltk==3.6.2 \
                                         ml-collections==0.1.1
                                         

RUN apt-get autoremove -y && apt-get autoclean -y


ENV PYTHONPATH .

