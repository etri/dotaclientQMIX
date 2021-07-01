FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

RUN pip install --user tensorboardX==2.0 pika==1.1.0 aioamqp==0.14.0 grpcio==1.27.2 scipy==1.4.0 pypng==0.0.19 pillow==7.0.0
 
RUN pip install --user dotaservice==0.3.9

RUN mkdir /root/dotaclient /root/dotaclient/logs

WORKDIR /root/dotaclient

COPY agent.py optimizer.py policy.py qmix.py /root/dotaclient/
