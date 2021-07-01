# dotaclientQMIX
DotaClient with QMIX extension
- Multi-Agent Reinforcement Learning for Dota 2
- This is an extension of dotaclient(https://github.com/TimZaman/dotaclient) to support multiplayers (unto 5), and Multi-agent Reinforcement Algorithms (QMIX and IQL)

## Components

* [DotaService](https://github.com/TimZaman/dotaservice) (dotarl/dotaservice)
* [DotaClient](https://github.com/TimZaman/dotaclient) (dotarl/dotaclient)
* [PyMARL](https://github.com/oxwhirl/pymarl) (dotarl/dotaclient/pymarl)

## Installation

### DotaService

Read dotaservice/docker/README.md

### RabbitMQ

```bash
sudo apt install rabbitmq-server
sudo rabbitmq-plugins enable rabbitmq_management
sudo rabbitmq-plugins enable rabbitmq_recent_history_exchange
```

### Python packages

```bash
pip install torch tensorboardX pika aioamqp grpcio scipy pypng==0.0.19 pillow pprint
```

## Launch

```bash
docker run -dp 13337:13337 dotaservice
python optimizer.py
python agent.py
```
