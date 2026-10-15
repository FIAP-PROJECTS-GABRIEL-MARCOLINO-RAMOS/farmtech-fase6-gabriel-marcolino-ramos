# FarmTech Solutions — Fase 6: O despertar da Rede Neural

**Gabriel Marcolino Ramos — RM565717**  
Inteligência Artificial — FIAP | Integrante único

Experimento de visão computacional com 80 imagens reais de gatos e carros. Compara dois YOLOv5n ajustados independentemente por 30 e 60 épocas, o YOLOv5n padrão e uma CNN treinada do zero. O notebook apresenta o protocolo, as curvas, os resultados reais de validação/teste e a análise crítica.

- [Notebook executado](notebooks/GabrielMarcolinoRamos_rm565717_pbl_fase6.ipynb)
- [Abrir no Google Colab](https://colab.research.google.com/github/FIAP-PROJECTS-GABRIEL-MARCOLINO-RAMOS/farmtech-fase6-gabriel-marcolino-ramos/blob/main/notebooks/GabrielMarcolinoRamos_rm565717_pbl_fase6.ipynb)
- [Imagens e rótulos no Drive](https://drive.google.com/drive/folders/1VGalo7vTeXqBAu0D-RESBMgMX6a5iobW) — cópia na conta do titular; a base também está neste repositório.
- [Vídeo demonstrativo no YouTube](https://youtu.be/fcv1dElMhMU) — não listado, 4min58s, com a voz de Gabriel. [Roteiro](docs/narracao.txt).
- [Prévia silenciosa da demonstração](output/video/previa_sem_narracao.mp4) — montagem das saídas reais, ainda sem narração.

## Arquivos

| Caminho | Conteúdo |
|---|---|
| `notebooks/` | Notebook com todas as células executadas |
| `data/` | Imagens 40+40, divisão 32/4/4 por classe, caixas YOLO/VOC e manifesto com fontes/licenças |
| `src/` | Preparação, auditoria, treinamento e avaliação |
| `models/` | Checkpoints dos dois YOLOs ajustados e da CNN |
| `results/` | Históricos, logs, métricas, predições e figuras |
| `docs/` | Requisitos, rotulação, evidência do Drive e roteiro do vídeo |
| `scripts/` | Bootstrap e construção do notebook/material de entrega |

As fotografias são do COCO e têm licenças próprias, incluindo restrições não comerciais e compartilhamento pela mesma licença. Consulte [data/README.md](data/README.md) e [manifest.csv](data/manifest.csv) antes de reutilizar. O código e os pesos YOLOv5 seguem as condições do projeto de origem; consulte [THIRD_PARTY.txt](THIRD_PARTY.txt).

## Executar

O notebook foi executado em Python 3.12, PyTorch 2.5.1 e GPU RTX 3050 Laptop. No Colab, selecionar GPU e executar as células em ordem. A montagem do Drive exige autenticação; o fluxo específico do Colab não foi executado nesta validação local.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/bootstrap.py
python -m ipykernel install --user --name farmtech-fase6
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=farmtech-fase6 \
  notebooks/GabrielMarcolinoRamos_rm565717_pbl_fase6.ipynb
```

`REUTILIZAR_TREINOS=True` preserva os treinos originais e repete a avaliação; usar `False` para treinar novamente. Tempos preservados são dos treinos originais, e novas execuções podem variar. O bootstrap fixa o YOLOv5 v7.0 e baixa seu checkpoint oficial. Os dados estão versionados, permitindo execução sem acesso à cópia privada do Drive.

A execução integral também foi conferida em uma cópia limpa dos arquivos versionados, com o ambiente Python existente: 14 células sem erros e reprodução das métricas de acurácia, F1 e mAP. [Registro da verificação](results/reproduction_check.json).

## Material para submissão

Prazo original: **13/10/2026 às 23h59**. Material finalizado em **15/10/2026**, sem alteração artificial de datas. O portal informa recebimento até 3 dias após o prazo, com nota limitada a 70%. O link “Entregar atividade” abriu o formulário de envio apesar do aviso de prazo encerrado. A confirmação da submissão deve ser verificada no próprio portal; este repositório não é comprovante de entrega.

A importação de 64 imagens e das caixas VOC no MakeSense, a conferência visual por amostragem e a exportação YOLO foram concluídas. Os 64 arquivos exportados foram comparados com as caixas importadas; a diferença para os rótulos de treino decorre apenas do arredondamento das coordenadas VOC. [Evidências](data/makesense/) e [metodologia](docs/rotulacao.md).

[PDF com os links para submissão](output/pdf/GabrielMarcolinoRamos_rm565717_pbl_fase6.pdf). Após o envio acadêmico confirmado, não realizar novos commits.
