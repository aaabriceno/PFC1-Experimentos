"""
Version paralelizada de cpg_GS.py: mismo algoritmo (graphGeneration, word2vec,
check de cpg_GS.py, sin tocar esas funciones), pero reparte los archivos de
cada split entre N workers con multiprocessing.Pool en vez de un for loop
secuencial de un solo core.

Motivo: timing test (100 train + 100 test + 20 valid) midio cpg_GS.py como
el cuello de botella del pipeline completo (~1s/archivo secuencial, ~9.4h
extrapoladas a escala completa de LineVul test.csv + train/valid propios),
con 32 cores disponibles y 0% de paralelismo en el script original.

Uso: identico a cpg_GS.py (mismos flags), agrega --workers (default: todos
los cores menos 2).
"""
import argparse
import json
import multiprocessing
import os

from cpg_GS import (
    edgeType_reduced,
    graphGeneration,
    read_csv,
    ver_edge_type_reduced,
    word2vec as word2vec_embed,
)
from gensim.models import Word2Vec
from tqdm import tqdm

_WV_MODEL = None
_CSV_DIR = None


def _init_worker(wv_path, csv_dir):
    global _WV_MODEL, _CSV_DIR
    _WV_MODEL = Word2Vec.load(wv_path)
    _CSV_DIR = csv_dir


def _process_entry(entry):
    file_name = entry["file_path"]
    label = int(entry["label"])
    nodes_path = os.path.join(_CSV_DIR, file_name, "nodes.csv")
    edges_path = os.path.join(_CSV_DIR, file_name, "edges.csv")
    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        return None

    nodes = read_csv(nodes_path)
    edges = read_csv(edges_path)
    index_map, graph, nodes_num, edges_num = graphGeneration(
        nodes, edges, edgeType_reduced, ver_edge_type_reduced
    )
    bad = edges_num is True
    if index_map is None or graph is None or nodes_num is None or edges_num is None:
        return {"bad": bad, "file_name": file_name if bad else None}

    gInput = word2vec_embed(nodes, index_map, graph, _WV_MODEL)
    if gInput is None:
        return {"bad": bad, "file_name": None}

    data_point = {"node_features": gInput, "graph": graph, "targets": [[label]]}
    num_point = {"file_name": file_name, "nodes_num": nodes_num, "edges_num": edges_num}
    return {"bad": False, "file_name": None, "data_point": data_point, "num_point": num_point, "src_file_name": file_name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../devign_dataset/devign_parsed/")
    parser.add_argument("--json_files", nargs=3, default=[
        "devign_dataset/devign_data_split/train_raw_code.json",
        "devign_dataset/devign_data_split/test_raw_code.json",
        "devign_dataset/devign_data_split/valid_raw_code.json",
    ])
    parser.add_argument("--wv", default="devign_dataset/devign_wv_models/devign_train_subtoken_data")
    parser.add_argument("--output_dir", default="devign_dataset/devign_cpg_c2_2")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    args = parser.parse_args()

    train_path, test_path, valid_path = args.json_files
    splits = {}
    for name, path in [("train", train_path), ("test", test_path), ("valid", valid_path)]:
        with open(path, "r") as f:
            splits[name] = json.load(f)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    bad_file = []
    with multiprocessing.Pool(
        processes=args.workers, initializer=_init_worker, initargs=(args.wv, args.csv)
    ) as pool:
        for name, entries in splits.items():
            final_data, final_num, files = [], [], []
            for result in tqdm(
                pool.imap(_process_entry, entries, chunksize=8),
                total=len(entries),
                desc=name,
            ):
                if result is None:
                    continue
                if result["bad"] and result["file_name"]:
                    bad_file.append(result["file_name"])
                if "data_point" not in result:
                    continue
                final_data.append(result["data_point"])
                final_num.append(result["num_point"])
                files.append(result["src_file_name"])

            with open(os.path.join(output_dir, f"devign-{name}-v2.json"), "w") as f:
                json.dump(final_data, f)
            with open(os.path.join(output_dir, f"devign-{name}-num-v2.json"), "w") as f:
                json.dump(final_num, f)
            with open(os.path.join(output_dir, f"devign-{name}-file.json"), "w") as f:
                json.dump(files, f)
            print(f"{name}: {len(final_data)}/{len(entries)} grafos generados")

    with open(os.path.join(output_dir, "bad_file.json"), "w") as f:
        json.dump(bad_file, f)


if __name__ == "__main__":
    main()
