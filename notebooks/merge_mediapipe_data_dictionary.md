# MediaPipe combinado — esquema v2

frames.csv contém uma linha por frame e 1.641 colunas: 12 metadados canônicos e
1.629 coordenadas (543 landmarks x 3 eixos). Consulte data_dictionary.csv para
tipos, nulabilidade e significado de cada coluna. Os CSVs usam UTF-8, vírgula e
cabeçalho único. Ausências são campos vazios; booleanos são True/False.

## Chaves e metadados

Chave de frame: (sample_id, frame_id). classes.csv tem chave class_id;
samples.csv tem chave sample_id. Essas tabelas guardam rótulos e proveniência
uma única vez, sem repetir descrições em todos os frames.

IDs globais têm prefixo do dataset e partes separadas por ::; caracteres
especiais nas partes são escapados por urllib.parse.quote(safe='').
class_id usa sign_id da fonte, exceto MINDS, que usa category.
sample_id usa sequence_id no INCLUDE-50, (interpreter, video_name) no KSL,
(person, video_name) no MINDS e sample_id no UFOP.
sequence_id usa sequence_id da fonte no UFOP; nas demais fontes coincide
com sample_id. signer_id usa interpreter/person/participant_id e fica vazio
no INCLUDE-50. Pessoas e classes de fontes diferentes não são alinhadas.

frame_id preserva o índice local. source_frame_id preserva original_frame_id
do UFOP, sem mudar a base. source_start_frame/source_end_frame preservam os
limites inclusivos da fonte. Não some/subtraia 1 sem verificar a convenção.

## Transformações e valores ausentes

Landmarks anatômicos de mãos/pose do MINDS foram renomeados para índices pela
ordem MediaPipe documentada em src/preprocessing/landmark_subsets.py. A ordem é
face 0..467, hand_0 0..20, pose 0..32, hand_1 0..20; cada índice tem x, y, z.
Valores numéricos finitos não são arredondados, interpolados ou reescalados.
O esquema unifica nomes, sem certificar equivalência dos protocolos de extração
ou dos referenciais das coordenadas entre fontes ou grupos de landmarks.
Campo vazio e NaN (sem distinção de maiúsculas) viram campo vazio. Zero é válido.

missing_hand_0, missing_hand_1, missing_pose e missing_face são recalculados:
True indica QUALQUER coordenada x/y/z ausente no respectivo grupo, incluindo
ausência parcial. Esses campos não medem confiança nem visibilidade/oclusão.
Os flags originais são descartados porque suas definições diferem entre fontes.
Não há missing_hand agregado: ele pode ser derivado explicitamente com OR ou
AND dos flags individuais, conforme o uso, sem impor uma semântica ambígua.

Aliases frame/person/interpreter/participant_id e *_uid não são exportados no
CSV principal. IDs originais, nomes de arquivo e rótulos foram mapeados para
classes.csv e samples.csv. category do MINDS é classe; category do INCLUDE-50
é categoria temática. sign_key é mantido apenas como proveniência da classe.

## Splits e validação

O split original train/val/test do INCLUDE-50 é preservado; demais splits ficam
vazios. Não foi criado um protocolo conjunto. Ao criar splits, agrupe todos os
frames de sample_id e todos os segmentos de sequence_id. Não use sample_id como
proxy de sinalizador. signer_id ausente não constitui uma pessoa adicional.

A exportação rejeita cabeçalhos inesperados, coordenadas não numéricas/infinitas,
frames duplicados, inteiros inválidos, limites UFOP inconsistentes e mudanças de
metadados dentro de uma classe/amostra. Também verifica signer_id/split dentro
de sequence_id. Não descarta frames automaticamente. Duplicidade de chaves é
validada com SQLite temporário, sem guardar todas as coordenadas em memória.

manifest.json informa entradas, contagens e complete_export. Se false, trata-se
somente de um teste limitado por frames e os vídeos podem estar incompletos.
As contagens refletem o conteúdo processado; não comprovam a completude dos
datasets originais. As licenças e versões de extração não são inferidas dos CSVs.

## Leitura

Leia IDs como string, preservando zeros à esquerda. Para coordenadas, deixe o
pandas interpretar campos vazios como NaN. Leia arquivos grandes em chunks.
Associe frames a classes usando class_id e a samples usando sample_id com
validate='many_to_one'; não faça joins por nome do sinal ou IDs locais.

## Colunas de metadados

### frames.csv

| Coluna | Tipo | Aceita vazio | Significado |
| --- | --- | --- | --- |
| `dataset` | string | não | Origem: include50, ksl, minds ou ufop. |
| `class_id` | string | não | ID global da classe: dataset::ID original, com escape percentual. Referência a classes.csv. |
| `sample_id` | string | não | ID global do vídeo/segmento; referência a samples.csv. Não identifica o sinalizador. |
| `sequence_id` | string | não | Grupo de segmentos da mesma sequência no UFOP; nas demais fontes coincide com sample_id. Use para agrupar splits. |
| `signer_id` | string | sim | ID do sinalizador com prefixo de origem; vazio no INCLUDE-50. Não relaciona pessoas entre datasets. |
| `frame_id` | integer | não | Índice não negativo do frame na amostra, copiado de frame_id ou frame. Não é renumerado; lacunas são preservadas. |
| `source_frame_id` | integer | sim | original_frame_id do UFOP, na convenção da fonte. Vazio nas demais fontes; não se assume a mesma base de frame_id. |
| `split` | string | sim | train, val ou test no INCLUDE-50; vazio nos demais datasets. Nenhum split novo é criado. |
| `missing_hand_0` | boolean | não | True se qualquer coordenada x/y/z da mão 0 (esquerda na extração) estiver ausente; False caso contrário. |
| `missing_hand_1` | boolean | não | True se qualquer coordenada x/y/z da mão 1 (direita na extração) estiver ausente; False caso contrário. |
| `missing_pose` | boolean | não | True se qualquer coordenada x/y/z da pose estiver ausente; False caso contrário. |
| `missing_face` | boolean | não | True se qualquer coordenada x/y/z da face estiver ausente; False caso contrário. |

### classes.csv

| Coluna | Tipo | Aceita vazio | Significado |
| --- | --- | --- | --- |
| `class_id` | string | não | ID global da classe: dataset::ID original, com escape percentual. Referência a classes.csv. |
| `source_class_id` | string | não | sign_id da fonte, ou category no MINDS; zeros à esquerda preservados. |
| `label` | string | sim | Texto de sign, disponível no INCLUDE-50 e KSL. Não traduzido; vazio quando não fornecido. |
| `category_name` | string | sim | Categoria temática de category no INCLUDE-50. O category do MINDS não é copiado para este campo. |
| `source_category_id` | string | sim | category_id do UFOP; código original, não nome de categoria. |
| `source_class_key` | string | sim | sign_key original no INCLUDE-50/UFOP, preservado para proveniência. |
| `source_class_id_in_category` | string | sim | sign_id_in_category do UFOP, preservado para proveniência. |

### samples.csv

| Coluna | Tipo | Aceita vazio | Significado |
| --- | --- | --- | --- |
| `sample_id` | string | não | ID global do vídeo/segmento; referência a samples.csv. Não identifica o sinalizador. |
| `source_sample_id` | string | sim | sample_id original do INCLUDE-50/UFOP. Pode não ser globalmente único; não usar como chave. |
| `source_sequence_id` | string | sim | sequence_id original do INCLUDE-50/UFOP, sem prefixo. Vazio quando a fonte não possui esse campo. |
| `source_signer_id` | string | sim | interpreter do KSL, person do MINDS ou participant_id do UFOP; vazio no INCLUDE-50. |
| `source_video_name` | string | não | video_name original, sem inferir se representa o arquivo completo ou segmento. |
| `source_video_relpath` | string | sim | video_relpath original do INCLUDE-50; vazio nas demais fontes. |
| `source_start_frame` | integer | sim | start_frame original do UFOP, limite inicial do segmento na convenção da fonte. |
| `source_end_frame` | integer | sim | end_frame original do UFOP, limite final inclusivo na convenção da fonte. |
