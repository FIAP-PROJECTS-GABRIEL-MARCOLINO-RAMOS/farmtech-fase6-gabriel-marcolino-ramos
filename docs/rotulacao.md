# Rotulação e conferência no Make Sense AI

As caixas de referência são anotações humanas originais do COCO 2017. O script `src/dataset.py` seleciona as imagens e converte as coordenadas para YOLO e Pascal VOC. A conversão automática está concluída. A importação, revisão e exportação no Make Sense AI devem ser comprovadas antes de declarar atendida essa etapa do enunciado.

1. Abrir https://www.makesense.ai/ e selecionar as 64 imagens de `data/images/train/`.
2. Escolher **Object Detection** e criar as classes nesta ordem: `gato`, `carro`.
3. Importar as anotações Pascal VOC de `data/annotations_voc/train/`.
4. Conferir o enquadramento das caixas e a classe de cada objeto. Preservar também os objetos pequenos da mesma classe: não remover anotações só para melhorar a métrica.
5. Exportar em formato YOLO, guardar o ZIP original em `data/makesense/` e salvar uma captura da ferramenta.
6. Conferir a correspondência dos IDs e comparar as coordenadas exportadas com os rótulos usados. Se houver correções reais, atualizar os rótulos e executar novamente os experimentos antes da entrega.

Validação e teste possuem rótulos de referência para cálculo das métricas; suas imagens nunca participam dos gradientes de treinamento.

Em 15/10/2026, a importação foi autorizada pelo titular e tentada no makesense.ai. O seletor de arquivos não emitiu o evento de abertura pelas interfaces disponíveis; nenhuma revisão/exportação foi concluída. Não confundir a tentativa com atendimento do requisito.
