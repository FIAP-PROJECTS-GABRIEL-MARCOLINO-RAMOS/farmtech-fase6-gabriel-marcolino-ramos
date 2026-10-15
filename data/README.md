# Base de imagens

80 imagens reais do conjunto **COCO 2017 validation**, selecionadas antes dos experimentos com semente 565717. Classes do projeto: `0=gato` e `1=carro`. IDs originais COCO: `cat=17` e `car=3`; IDs contíguos do YOLO pré-treinado: `cat=15` e `car=2`.

| Classe | Treino | Validação | Teste | Total |
|---|---:|---:|---:|---:|
| Gato | 32 | 4 | 4 | 40 |
| Carro | 32 | 4 | 4 | 40 |
| Total | 64 | 8 | 8 | 80 |

`manifest.csv` registra o ID, partição, classe, dimensão, quantidade de caixas, URL de origem, página Flickr, licença e SHA-256 de cada imagem. Os arquivos originais foram preservados. As fotografias continuam pertencendo aos respectivos titulares indicados pelas fontes; não são imagens autorais deste projeto.

Critérios: uma única classe de interesse por imagem (pode haver vários objetos da mesma classe); ausência de caixas `crowd`; maior caixa de interesse com pelo menos 10% da área da imagem; licenças CC BY, BY-SA, BY-NC ou BY-NC-SA. Esses filtros simplificam o problema e limitam a generalização para cenários mais difíceis. A finalidade desta seleção é acadêmica e não comercial; respeitar as restrições de cada imagem antes de qualquer reutilização.

Anotações: `coco_subset.json` preserva as caixas originais; `labels/` contém o formato YOLO; `annotations_voc/` permite revisão no Make Sense AI. As anotações COCO têm licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), conforme os [termos do COCO](https://cocodataset.org/#termsofuse). Fonte: [COCO](https://cocodataset.org/).

Não há sobreposição de IDs ou de SHA-256 entre partições. O teste tem somente oito imagens, sem imagens negativas e sem cenas contendo as duas classes juntas. Não representa validação de segurança, controle de acesso ou uso em produção.
