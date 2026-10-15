"""Confere exportação real do MakeSense contra VOC e coordenadas de treino."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / 'data/makesense/rotulos_makesense_yolo.zip'
maximum_normalized_difference = 0.0
maximum_pixel_difference = 0.0
box_count = 0
with zipfile.ZipFile(archive) as exported:
    files = exported.namelist()
    assert set(files) == {p.name for p in (ROOT / 'data/labels/train').glob('*.txt')}
    for name in files:
        xml = ET.parse(ROOT / 'data/annotations_voc/train' / Path(name).with_suffix('.xml'))
        width = int(xml.findtext('size/width'))
        height = int(xml.findtext('size/height'))
        expected = []
        for obj in xml.findall('object'):
            x1, y1, x2, y2 = [float(obj.findtext('bndbox/' + key))
                               for key in ('xmin', 'ymin', 'xmax', 'ymax')]
            expected.append([['gato', 'carro'].index(obj.findtext('name')),
                             (x1+x2)/2/width, (y1+y2)/2/height,
                             (x2-x1)/width, (y2-y1)/height])
        actual = np.loadtxt(exported.open(name), ndmin=2)
        expected = np.array(expected)
        original = np.loadtxt(ROOT / 'data/labels/train' / name, ndmin=2)
        assert actual.shape == expected.shape == original.shape
        assert np.all(actual[:, 0] == expected[:, 0])
        # Exportador usa seis casas, com truncamento/arredondamento de ponto flutuante.
        assert np.allclose(actual, expected, atol=0.0000011, rtol=0), name
        maximum_normalized_difference = max(maximum_normalized_difference,
                                            float(abs(actual-original).max()))
        def corners(values):
            return np.column_stack(((values[:,1]-values[:,3]/2)*width,
                                    (values[:,2]-values[:,4]/2)*height,
                                    (values[:,1]+values[:,3]/2)*width,
                                    (values[:,2]+values[:,4]/2)*height))
        maximum_pixel_difference = max(maximum_pixel_difference,
                                      float(abs(corners(actual)-corners(original)).max()))
        box_count += len(actual)
assert len(files) == 64 and maximum_pixel_difference < 0.501
result = {
    'date': '2026-10-15', 'tool': 'https://www.makesense.ai/',
    'images_imported': 64, 'format_imported': 'Pascal VOC',
    'format_exported': 'YOLO ZIP', 'class_order': ['gato', 'carro'],
    'exported_label_files': len(files), 'boxes': box_count,
    'comparison_with_imported_voc': '64/64 iguais dentro da precisão de exportação (6 casas)',
    'max_normalized_difference_to_training': maximum_normalized_difference,
    'max_corner_difference_pixels_to_training': maximum_pixel_difference,
    'training_labels_unchanged': True,
    'rounding': 'VOC arredonda as coordenadas COCO para pixels inteiros; treino preserva coordenadas COCO fracionárias. Não houve correção semântica das caixas.',
    'review': 'Conferência visual por amostragem na interface e comparação numérica de todos os 64 arquivos exportados. Caixas humanas COCO preservadas.',
    'drive_export_file_id': '1xZhWqPzxayZzHaLu8LjB5bJwDe7uKN3Z',
    'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
}
(ROOT / 'data/makesense/verificacao.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
