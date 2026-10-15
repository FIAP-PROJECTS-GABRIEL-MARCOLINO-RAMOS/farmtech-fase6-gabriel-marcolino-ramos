"""Gera o suporte de link para a entrega; não envia ao portal."""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

ROOT = Path(__file__).resolve().parents[1]
REPO = 'https://github.com/FIAP-PROJECTS-GABRIEL-MARCOLINO-RAMOS/farmtech-fase6-gabriel-marcolino-ramos'
OUT = ROOT / 'output/pdf/GabrielMarcolinoRamos_rm565717_pbl_fase6.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
styles = getSampleStyleSheet()
font_dir = Path('/usr/share/fonts/truetype/dejavu')
pdfmetrics.registerFont(TTFont('DejaVuSans', str(font_dir / 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', str(font_dir / 'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('DejaVuSans', normal='DejaVuSans', bold='DejaVuSans-Bold')
styles.add(ParagraphStyle('BodyPT', fontName='DejaVuSans', fontSize=11, leading=17, spaceAfter=13))
styles.add(ParagraphStyle('TitlePT', fontName='DejaVuSans-Bold', fontSize=23, leading=29, spaceAfter=20))
styles.add(ParagraphStyle('LinkPT', fontName='DejaVuSans', fontSize=9, leading=16, textColor=colors.HexColor('#15589b'), spaceAfter=14))
story = [Paragraph('FIAP | Inteligência Artificial', styles['BodyPT']),
         Paragraph('Fase 6<br/>O despertar da Rede Neural', styles['TitlePT']),
         Paragraph('<b>Gabriel Marcolino Ramos</b><br/>RM565717 | Integrante único', styles['BodyPT']),
         HRFlowable(width='100%', color=colors.HexColor('#bfc7d0'), thickness=1), Spacer(1, 20),
         Paragraph('<b>Repositório do projeto</b>', styles['BodyPT']),
         Paragraph(f'<link href="{REPO}">github.com/FIAP-PROJECTS-GABRIEL-MARCOLINO-RAMOS/<br/>farmtech-fase6-gabriel-marcolino-ramos</link>', styles['LinkPT']),
         Paragraph('O README centraliza o notebook executado, a base de imagens, os pesos treinados, os resultados e o acesso à demonstração em vídeo quando publicada.', styles['BodyPT']),
         Paragraph('<b>Escopo</b><br/>Duas classes, 80 imagens e divisão 32/4/4 por classe. Comparação entre YOLOv5n com 30 e 60 épocas, YOLOv5n padrão e CNN treinada do zero.', styles['BodyPT']),
         Paragraph('<b>Versão de preparação - 15/10/2026</b><br/>Vídeo narrado/publicado e revisão no MakeSense ainda pendentes. Este arquivo não é comprovante de envio. Concluir as pendências registradas no README antes da submissão formal.', styles['BodyPT'])]
SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=55, leftMargin=55, topMargin=58, bottomMargin=50,
                  title='FIAP Fase 6 - Gabriel Marcolino Ramos - RM565717', author='Gabriel Marcolino Ramos').build(story)
print(OUT)
