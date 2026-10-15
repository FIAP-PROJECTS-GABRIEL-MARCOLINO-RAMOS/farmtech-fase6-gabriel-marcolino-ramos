# Rotulação e conferência no Make Sense AI

As caixas de referência são anotações humanas originais do COCO 2017. O script `src/dataset.py` seleciona as imagens e converte as coordenadas para YOLO e Pascal VOC.

Em 15/10/2026, foram importadas as 64 imagens de treinamento em https://www.makesense.ai/, iniciado um projeto Object Detection com classes `gato` e `carro`, nessa ordem, e importados os 64 arquivos Pascal VOC correspondentes. A conferência visual por amostragem preservou as caixas de referência, incluindo objetos pequenos e parcialmente ocluídos. As anotações foram exportadas pela própria ferramenta em YOLO.

- Exportação original: `data/makesense/rotulos_makesense_yolo.zip` (64 arquivos, 131 caixas).
- Captura da interface: `data/makesense/revisao-interface.jpg`.
- Comparação reproduzível: `python scripts/verify_makesense.py`.
- Registro e hash: `data/makesense/verificacao.json`.
- Cópia da exportação no Drive: https://drive.google.com/file/d/1xZhWqPzxayZzHaLu8LjB5bJwDe7uKN3Z/view.

Todos os arquivos exportados correspondem às caixas VOC importadas, dentro da precisão do exportador. O VOC arredonda as coordenadas COCO para pixels inteiros; os rótulos usados no treino preservam as coordenadas fracionárias originais. A diferença máxima nos cantos das caixas é de 0,501 pixel, incluindo a precisão de exportação. Não houve alteração semântica de caixas ou classes; os treinos originais e seus resultados foram preservados.

A revisão foi feita sobre anotações humanas preexistentes, sem desenhar novas caixas do zero. Validação e teste possuem rótulos de referência para cálculo das métricas; suas imagens não participam dos gradientes de treinamento.

O vídeo foi gravado antes desta conferência na ferramenta. O estado atualizado e a evidência estão nos arquivos acima.
