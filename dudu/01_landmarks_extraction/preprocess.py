import os
import argparse
from datasets import BaseDataset
from base_extractor import BaseExtractor

def main():
    parser = argparse.ArgumentParser(description="Pipeline para extração de landmarks de datasets de LIBRAS")
    parser.add_argument("-d", "--dataset", required=True, choices=['ufop', 'vlibras', 'minds'], help="Nome do dataset a ser processado")
    parser.add_argument("-e", "--extractor", default='mediapipe', choices=['mediapipe', 'openpose'], help="Extrator de landmarks a ser utilizado.")
    parser.add_argument("-i", "--input_dir", default='../00_datasets', help="Diretório base onde os datasets brutos estão localizados.")
    parser.add_argument("-o", "--output_dir", default='../00_datasets/dataset_output/', help="Diretório base para salvar os arquivos CSV processados.")
    parser.add_argument("-c", "--chunk_size", type=int, default=10000, help="Número de frames por chunk a ser salvo no CSV.")

    args = parser.parse_args()

    dataset_instance = BaseDataset.create(dataset_name=args.dataset, base_path=args.input_dir)
    extractor = BaseExtractor.create(args.extractor)
    video_processor = dataset_instance.get_processor(extractor)

    videos = dataset_instance.prepare_data()

    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.dataset}_{args.extractor}.csv")
    
    if os.path.exists(output_path):
        raise FileExistsError(f"File {output_path} already exists. Please remove it before running the script again.")

    video_processor.process_all(
        videos,
        output_path,
        chunk_size=args.chunk_size
    )

if __name__ == "__main__":
    main()