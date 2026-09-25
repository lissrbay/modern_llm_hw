# modern_llm_hw
Репозиторий для домашек по курсу Современные LLM

Модель обучалась с параметрами:

ARGS_DEFAULT(
    
    DIM=256,

    N_LAYERS=4,

    N_HEADS=8,

    N_KV_HEADS=2,

    BATCH_SIZE=128,

    EPOCHS=4,

    LR=1e-4,

    LOG_EVERY=40,

    LOG_DIR="runs/llama"
)

Эпсилон для RoPE захардкожено в 1e-6 и внутренняя размерность FFN х4 от входа согласно оригинальной архитектуре.

В качестве датасета взяла тексты гороскопов, в качестве токенизатора - тиктокен cl100k_base. 

Запуск обучения с параметрами:

![plot](./imgs/nqCLD3tcizsCxdIdpKJy7iLjjKv3Mo9TUb96_yi_5cSendwl3sTsmrLOcg2GC1iRBjSjKJTNRxz0U9c0n9-SdrAZ.jpg)
![plot](./imgs/VN3pyLMDrsPHUCcDEBLE4uLL1OhXe8xCTeCX_lgPzvutXAFZa4LZhaBjgzjnGjsbdHaZdvlviTl4demFoLu-MFl4.jpg)

Тензорборд графики:

Эпохи:
![plot](./imgs/Hsb14MWwAvvsfZ-NmXAm4Hee1sc2tXKDI8x3O-k6XrgNzsCMBsMUuWK8AobQLIRj-u2H6z2R6y5ffXM7UcxrWVQ0.jpg)

Лосс на обучении:
![plot](./imgs/Me5IwD9QhyY9Zx81jvO2HhCTXQN7kuO1SRgolgn9LACLdm5Y8-iesMlZ4oFxDM8i38cMGxtiZB3ktb4WddQSADFh.jpg)

Лосс на валидации:
![plot](./imgs/2GCrKuZEXuSex-PNL-f2DuVojufW8bvEt0UDkb-l9u-SFxScXKdGK_IMFe7YOvmtqyppoDJmQ0apuU7-L-Cgf3Wb.jpg)