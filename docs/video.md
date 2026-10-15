# Vídeo com a voz de Gabriel

Texto de narração: `narracao.txt`. Ler em ritmo natural, visando de 4min20s a 4min50s; o limite total é cinco minutos. O texto tem aproximadamente 590 palavras e deve ser ensaiado para caber no limite. Se necessário, gravar em blocos, sem ler os títulos do roteiro visual.

| Trecho | Demonstração |
|---|---|
| Identificação e objetivo | README e identificação do notebook |
| Base e anotações | Manifesto, divisão 32/4/4 e exemplo real com caixas |
| Notebook | Células executadas e conexão condicional com o Drive |
| Treinamentos | Chamadas de 30/60 épocas e arquitetura CNN |
| Curvas | `results/yolo_learning_curves.png` e `cnn_learning_curves.png` |
| Métricas | `results/comparison.csv`: teste, acurácia e mAP distintos |
| Demonstração | Montagem das oito imagens de teste, sem selecionar só acertos |
| Conclusão | Limitações e arquivos disponíveis no repositório |

Gravar em ambiente silencioso, com o celular ou microfone habitual. Preservar o áudio original em `video/voz-gabriel.*` (não versionar dados desnecessários). O roteiro não afirma que a etapa MakeSense foi concluída antes da evidência. Ao concluir essa etapa, a frase correspondente pode ser atualizada para descrevê-la precisamente.

Após receber a gravação: conferir duração e inteligibilidade, sincronizar as telas reais do projeto com a fala, exportar MP4 e conferir o vídeo completo. Publicar no YouTube como **não listado**, inserir o endereço no README e verificar acesso. Publicado em 15/10/2026: https://youtu.be/fcv1dElMhMU (não listado).

## Gravação recebida em 15/10/2026

O áudio original foi preservado localmente em `video/voz-gabriel.ogg` e está excluído do Git. Duração original: 357,99 s. A edição encurta pausas, retira uma frase redundante sobre os treinos e aplica ajuste de ritmo de aproximadamente 1,129 vez, preservando o tom. A faixa editada tem 297,51 s. A sincronização usa limites por palavra de uma transcrição automática local; essa transcrição não é uma legenda revisada.

O MP4 narrado fica em `output/video/GabrielMarcolinoRamos_rm565717_fase6.mp4`. Publicado no YouTube como não listado: https://youtu.be/fcv1dElMhMU. O vídeo com voz não é enviado automaticamente ao repositório público.

A montagem usa `scripts/prepare_voice.py` e `scripts/build_video.py --audio output/video/voz-gabriel-editada.wav --timeline video/sincronizacao.json`. A montagem possui cenas estáticas com saídas reais, sem simular execução ao vivo. A gravação antecede a revisão no MakeSense, concluída depois e documentada em `docs/rotulacao.md`.
