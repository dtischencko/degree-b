# Исследование уязвимостей в моделях машинного обучения
## Model Cards
* **Censor's model** -> [Bi-Encoder](https://huggingface.co/DiTy/bi-encoder-russian-msmarco)
* **LLM model** -> [Autoregressive decoder-only](https://huggingface.co/Intel/neural-chat-7b-v3-2)

## Цензор намерений

Реализация самого цензора находится тут [censor_app](./src/censor_app/)

Ниже представлен псевдокод реализации цензора:
![pseudocode](./imgs/pseudocode.png)

## Start services

После пуллинга репозитория выполнить:
```sh
docker-compose up
```


