import os
import requests
import tarfile
import zipfile
import shutil

def download_file(url, output_path):
    if "dropbox.com" in url and not url.endswith("?dl=1"):
        url = f"{url}?dl=1"  # Force Dropbox direct download
    response = requests.get(url, stream=True, allow_redirects=True)
    # print(f"Redirect history: {[resp.url for resp in response.history]}")
    response.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def extract_tar(file_path, extract_to):
    """Extract a tar.gz file."""
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)

def extract_zip(file_path, extract_to):
    """Extract a zip file."""
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(path=extract_to)

def main():
    # Define paths
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    detox_train_path = os.path.join(data_dir, "detox_train.tsv")
    detox_val_path = os.path.join(data_dir, "detox_dev.tsv")
    detox_test_path = os.path.join(data_dir, "detox_test.tsv")
    ch_path = os.path.join(data_dir, "2ch.csv")
    ok_path = os.path.join(data_dir, "ok.ft")
    persona_path = os.path.join(data_dir, "persona.tsv")
    koziev_path = os.path.join(data_dir, "koziev.txt")
    bad_vocab_path = os.path.join(data_dir, "bad_vocab.txt")
    train_gen_path = os.path.join(data_dir, "seq2seq_gen_train.jsonl")
    val_gen_path = os.path.join(data_dir, "seq2seq_gen_val.jsonl")
    gen_path = os.path.join(data_dir, "seq2seq_gen.jsonl")
    final_path = os.path.join(data_dir, "final")

    # Detox datasets
    download_file("https://raw.githubusercontent.com/skoltech-nlp/russe_detox_2022/main/data/input/train.tsv", detox_train_path)
    download_file("https://raw.githubusercontent.com/skoltech-nlp/russe_detox_2022/main/data/input/dev.tsv", detox_val_path)
    download_file("https://raw.githubusercontent.com/skoltech-nlp/russe_detox_2022/main/data/input/test.tsv", detox_test_path)

    # Backtranslation/marker seq2seq
    seq2seq_tar_path = os.path.join(data_dir, "seq2seq_gen.tar.gz")
    download_file("https://www.dropbox.com/s/97wgsd4pvx49eqq/seq2seq_gen.tar.gz", seq2seq_tar_path)
    extract_tar(seq2seq_tar_path, data_dir)
    os.rename(os.path.join(data_dir, "seq2seq_gen_train.jsonl"), train_gen_path)
    os.rename(os.path.join(data_dir, "seq2seq_gen_val.jsonl"), val_gen_path)
    os.rename(os.path.join(data_dir, "seq2seq_gen.jsonl"), gen_path)
    os.remove(seq2seq_tar_path)

    # Vocabulary with bad words
    bad_vocab_tar_path = os.path.join(data_dir, "bad_vocab.txt.tar.gz")
    download_file("https://www.dropbox.com/s/ou6lx03b10yhrfl/bad_vocab.txt.tar.gz", bad_vocab_tar_path)
    extract_tar(bad_vocab_tar_path, data_dir)
    os.remove(bad_vocab_tar_path)

    # Kaggle 2ch dataset
    ch_zip_path = os.path.join(data_dir, "2ch.zip")
    download_file("https://www.dropbox.com/s/ob5tox8w8uoat12/2ch.zip", ch_zip_path)
    extract_zip(ch_zip_path, data_dir)
    os.rename(os.path.join(data_dir, "labeled.csv"), ch_path)
    os.remove(ch_zip_path)

    # Kaggle OK dataset
    ok_zip_path = os.path.join(data_dir, "ok.zip")
    download_file("https://www.dropbox.com/s/udn2a70obakzpa2/ok.zip", ok_zip_path)
    extract_zip(ok_zip_path, data_dir)
    os.rename(os.path.join(data_dir, "dataset.txt"), ok_path)
    os.remove(ok_zip_path)

    # Toloka Persona Chat Rus
    persona_zip_path = os.path.join(data_dir, "TlkPersonaChatRus.zip")
    download_file("https://tlk.s3.yandex.net/dataset/TlkPersonaChatRus.zip", persona_zip_path)
    extract_zip(persona_zip_path, data_dir)
    os.rename(os.path.join(data_dir, "TlkPersonaChatRus", "dialogues.tsv"), persona_path)
    os.remove(persona_zip_path)
    shutil.rmtree(os.path.join(data_dir, "TlkPersonaChatRus"))

    # Koziev dialogues
    koziev_zip_path = os.path.join(data_dir, "dialogues.zip")
    download_file("https://raw.githubusercontent.com/Koziev/NLP_Datasets/master/Conversations/Data/dialogues.zip", koziev_zip_path)
    extract_zip(koziev_zip_path, data_dir)
    os.rename(os.path.join(data_dir, "dialogues.txt"), koziev_path)
    os.remove(koziev_zip_path)

    # Final dataset for classification
    os.makedirs(final_path, exist_ok=True)
    clf_tar_path = os.path.join(final_path, "clf_data.tar.gz")
    download_file("https://www.dropbox.com/s/a7mwe74w2a7rzm0/clf_data.tar.gz", clf_tar_path)
    extract_tar(clf_tar_path, final_path)

    # Output paths
    print(f"Detox train: {detox_train_path}")
    print(f"Detox val: {detox_val_path}")
    print(f"Detox test: {detox_test_path}")
    print(f"Vocab: {bad_vocab_path}")
    print(f"2ch/Pikabu: {ch_path}")
    print(f"Odnoklassniki: {ok_path}")
    print(f"Toloka Persona: {persona_path}")
    print(f"Final dataset: {final_path}")

if __name__ == "__main__":
    main()
