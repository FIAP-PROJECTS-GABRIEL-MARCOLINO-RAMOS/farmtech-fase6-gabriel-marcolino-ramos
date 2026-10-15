"""Monta uma demonstração das saídas reais; recebe a voz original com --audio.

Sem áudio, gera uma prévia silenciosa de 270 s. Não simula execução ou interface.
"""
from pathlib import Path
import argparse
import json
import subprocess
import textwrap
from PIL import Image, ImageDraw, ImageFont, ImageOps
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/video'
FRAMES = ROOT / '.cache/video'
OUT.mkdir(parents=True, exist_ok=True)
FRAMES.mkdir(parents=True, exist_ok=True)
FONT = Path('/usr/share/fonts/truetype/dejavu')
W, H = 1920, 1080

def font(size, bold=False, mono=False):
    name = 'DejaVuSansMono' if mono else 'DejaVuSans'
    return ImageFont.truetype(str(FONT / (name + ('-Bold' if bold else '') + '.ttf')), size)

def page(title, subtitle=''):
    image = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(image)
    d.text((70, 44), title, font=font(46, True), fill='#132d40')
    d.text((70, 112), subtitle, font=font(27), fill='#4c5963')
    d.line((70, 163, 1850, 163), fill='#c1cad1', width=2)
    d.text((70, 1025), 'FIAP • Fase 6 | Gabriel Marcolino Ramos • RM565717 | Saídas reais do experimento',
           font=font(24), fill='#4c5963')
    return image

def lines(image, text, xy=(85, 205), size=33, width=82, mono=False):
    d = ImageDraw.Draw(image)
    x, y = xy
    for line in text.splitlines():
        for wrapped in textwrap.wrap(line, width=width, replace_whitespace=False) or ['']:
            d.text((x, y), wrapped, font=font(size, mono=mono), fill='#202c35')
            y += size + 15
    return y

def fitted(image, path, box):
    source = Image.open(path).convert('RGB')
    source = ImageOps.contain(source, (box[2]-box[0], box[3]-box[1]))
    image.paste(source, (box[0]+(box[2]-box[0]-source.width)//2, box[1]+(box[3]-box[1]-source.height)//2))

def save(image, number):
    path = FRAMES / f'{number:02d}.png'
    image.save(path)
    return path

def build_frames():
    frames = []
    df = pd.read_csv(ROOT / 'data/manifest.csv')
    image = page('O despertar da Rede Neural', 'YOLO ajustado, YOLO padrão e CNN do zero')
    lines(image, 'Gabriel Marcolino Ramos • RM565717\n80 imagens reais: gatos e carros\nDuas simulações independentes: 30 e 60 épocas', size=36)
    for i, cls in enumerate(['gato', 'carro']):
        row = df[(df.class_name == cls) & (df.split == 'train')].iloc[0]
        fitted(image, ROOT / row['file'], (160+830*i, 450, 930+830*i, 965))
    frames.append(save(image, 0))

    image = page('Base e partições fixas', 'Fonte: COCO val2017 • semente 565717 • manifesto com licenças e SHA-256')
    lines(image, 'Classe          Treino      Validação      Teste      Total\nGato               32              4          4         40\nCarro              32              4          4         40\nTotal              64              8          8         80', size=37, mono=True)
    lines(image, '155 caixas de referência\nSem IDs ou arquivos idênticos entre partições\nDuas fotos relacionadas inteiramente no treino\nAnotações COCO convertidas para YOLO e Pascal VOC\nRevisão no MakeSense ainda pendente', xy=(85, 600), size=32)
    frames.append(save(image, 1))

    nb = json.loads((ROOT / 'notebooks/GabrielMarcolinoRamos_rm565717_pbl_fase6.ipynb').read_text())
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    image = page('Notebook executado', f'{len(cells)} células de código • execução local • persistência condicional no Colab')
    lines(image, 'notebooks/GabrielMarcolinoRamos_rm565717_pbl_fase6.ipynb\n\nfrom src.dataset import audit\nfrom src.experiments import train_yolo, train_cnn\nfrom src.evaluation import evaluate_all\n\naudit()\ntrain_yolo(30, reuse=REUTILIZAR_TREINOS)\ntrain_yolo(60, reuse=REUTILIZAR_TREINOS)\ntrain_cnn(60, reuse=REUTILIZAR_TREINOS)\ncomparison = evaluate_all()', size=31, mono=True, width=85)
    frames.append(save(image, 2))

    image = page('Treinamentos e seleção', 'Treino e validação primeiro; teste reservado para a comparação final')
    lines(image, 'YOLOv5n • dois ajustes independentes\n30 e 60 épocas; mesmo checkpoint inicial COCO\n416 px • batch 8 • SGD • semente 565717\nSeleção customizada: maior mAP50–95 na validação\n\nCNN com inicialização aleatória\n3 blocos Conv + ReLU + MaxPool; 89.314 parâmetros\n128 px • batch 8 • Adam • 60 épocas\nMelhor checkpoint: época 7, menor perda de validação', size=36, width=85)
    frames.append(save(image, 3))

    for n, title, filename in [(4, 'Curvas dos dois YOLOs', 'yolo_learning_curves.png'),
                               (5, 'Curvas da CNN treinada do zero', 'cnn_learning_curves.png')]:
        image = page(title, 'Históricos completos preservados em results/; não são curvas ilustrativas')
        fitted(image, ROOT / 'results' / filename, (70, 210, 1850, 955))
        frames.append(save(image, n))

    results = pd.read_csv(ROOT / 'results/comparison.csv')
    test = results[results.split == 'test']
    image = page('Comparação no teste: oito imagens', 'Acurácia por imagem e mAP de detecção medem aspectos distintos')
    d = ImageDraw.Draw(image)
    columns = [85, 760, 1170, 1500]
    for x, label in zip(columns, ['Modelo', 'Acertos', 'mAP50', 'mAP50–95']):
        d.text((x, 240), label, font=font(34, True), fill='#132d40')
    names = {'yolo_padrao':'YOLOv5n padrão', 'yolov5n_30ep':'YOLOv5n 30 épocas', 'yolov5n_60ep':'YOLOv5n 60 épocas', 'cnn_zero':'CNN do zero'}
    for y, (_, row) in zip([340, 450, 560, 670], test.iterrows()):
        vals = [names[row.model], f'{row.correct_images}/8 ({row.image_accuracy:.1%})',
                f'{row.map50:.3f}' if pd.notna(row.map50) else '—',
                f'{row.map50_95:.3f}' if pd.notna(row.map50_95) else '—']
        for x, value in zip(columns, vals):
            d.text((x, y), value, font=font(33), fill='#202c35')
    lines(image, 'YOLO: maior confiança ≥ 0,25 define a classe; abstenção conta como erro.\nCNN: classifica a imagem inteira; mAP de caixas não se aplica.', xy=(85, 835), size=29, width=106)
    frames.append(save(image, 6))

    image = page('Todas as imagens de teste: YOLO ajustado', '60 épocas • referência em verde • predições em laranja • limiar 0,25')
    fitted(image, ROOT / 'results/yolov5n_60ep_test_montage.jpg', (70, 180, 1850, 1000))
    frames.append(save(image, 7))

    image = page('Todas as imagens de teste: YOLO padrão', 'O pré-treinamento COCO já inclui as duas classes deste experimento')
    fitted(image, ROOT / 'results/yolo_padrao_test_montage.jpg', (70, 180, 1850, 1000))
    frames.append(save(image, 8))

    image = page('Tempo e facilidade de integração', 'Medições locais • RTX 3050 Laptop • batch 1 • FP32 na avaliação')
    lines(image, '\n'.join(f'{names[r.model]}: treino local {r.training_seconds:.1f} s; inferência {r.latency_ms_median:.2f} ms'
                            for _, r in test.iterrows()), size=33, width=96)
    lines(image, 'Padrão: sem ajuste local; custo do pré-treinamento não contabilizado.\nLatência: leitura + pré-processamento + rede + NMS no YOLO.\n5 aquecimentos + 5 repetições por imagem; sincronização CUDA.\nResoluções e tarefas diferentes limitam a comparação direta.\nYOLO fornece caixas; CNN exige definir a arquitetura e só classifica.', xy=(85, 580), size=31, width=99)
    frames.append(save(image, 9))

    image = page('Limitações e conclusão', 'Resultados exploratórios, com rastreabilidade dos dados e das medições')
    lines(image, 'Teste pequeno: quatro imagens por classe\n6/8 acertos: IC95% Wilson de aproximadamente 40,9% a 92,9%\nClasses conhecidas no COCO; não é benchmark independente\nSem negativos, vídeo ao vivo ou avaliação de domínio agrícola\n\nYOLO padrão: melhor mAP de localização no teste\nYOLO ajustado: 6/8 na regra de acurácia adotada\nCNN: 5/8; menor custo local, sem caixas\n\nRepositório: dados, notebook, pesos, logs e análise crítica', size=33, width=94)
    frames.append(save(image, 10))
    return frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=Path)
    parser.add_argument('--frames-only', action='store_true')
    args = parser.parse_args()
    frames = build_frames()
    if args.frames_only:
        return
    # Divisão inicial; ajustar à fala original após recebimento do áudio.
    durations = [24, 38, 25, 39, 12, 10, 47, 16, 9, 23, 27]
    if args.audio:
        info = json.loads(subprocess.check_output(['ffprobe', '-v', 'quiet', '-show_format', '-of', 'json', str(args.audio)]))
        duration = float(info['format']['duration'])
        if duration > 298:
            raise ValueError('Áudio excede 298 s; editar/gravar novamente para respeitar cinco minutos.')
        durations = [s * duration / sum(durations) for s in durations]
    concat = FRAMES / 'concat.txt'
    concat.write_text(''.join(f"file '{p}'\nduration {s:.6f}\n" for p, s in zip(frames, durations)) + f"file '{frames[-1]}'\n")
    target = OUT / ('GabrielMarcolinoRamos_rm565717_fase6.mp4' if args.audio else 'previa_sem_narracao.mp4')
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(concat)]
    if args.audio:
        command += ['-i', str(args.audio), '-c:a', 'aac', '-b:a', '192k', '-shortest']
    command += ['-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'stillimage', '-r', '24', '-pix_fmt', 'yuv420p',
                '-t', f'{sum(durations):.6f}', '-movflags', '+faststart', str(target)]
    subprocess.run(command, check=True)
    print(target)

if __name__ == '__main__':
    main()
