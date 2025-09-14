import time
import numpy as np
from typing import List

SUBMISSION_TIME = 180
MAX_QUERY_BATCH_SIZE = 512
MAX_BATCH_SIZE = 64
prev_submission_time = 0
timeout_disabled = False
prev_submission_result = {
    'average_cosine_similarity': -69,
    'exact_reconstruction_accuracy': -2137,
    'final_score': 0
}

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-mpnet-base-v2", device='cuda')

try:
    target_embeddings = np.load('embeddings.npy')
except FileNotFoundError:
    msg = "Nie odnaleziono pliku embeddings.npy. Powinien być umieszczony w tym samym folderze co notatnik w momencie importowania simulated_api!"
    print(msg)
    raise Exception(msg)

def compute_score(value, min_value, max_value):
    return max(0, min(1, (value - min_value) / (max_value - min_value)))

def ranking():
    print(f'Twój ostatni submit ({prev_submission_result["final_score"]:.1f} pkt):')
    print(prev_submission_result)

def query(strings: List[str]) -> np.ndarray:
    start_t = time.time()
    if len(strings) > MAX_QUERY_BATCH_SIZE:
        raise Exception({"error": f"Max query batch size is {MAX_QUERY_BATCH_SIZE}."})
    if len(strings) == 0:
        raise Exception({"error": "Strings list is empty."})
    for s in strings:
        if type(s) != str:
            raise Exception({"error": "At least one element of strings list is not a string."})
        if len(s) > 80:
            raise Exception({"error": "At least one element of strings list is longer than 80 characters."})

    embeddings = model.encode(strings, convert_to_tensor=False, batch_size=MAX_BATCH_SIZE)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    
    tar_time = 0
    if not timeout_disabled:
        if len(strings) <= 5:
            tar_time = 0.4 + len(strings) * 0.1
        else:
            tar_time = 0.9 + (len(strings) - 7) / (512 - 7) * 0.8
    dn_time = time.time() - start_t
    if dn_time < tar_time:
        time.sleep(tar_time - dn_time)
    return embeddings / norms

def submit(strings: List[str]) -> dict:
    global prev_submission_time, prev_submission_result

    if len(strings) != len(target_embeddings):
        return {"error": f"Expected exactly {len(target_embeddings)} strings in submission."}
    for s in strings:
        if type(s) != str:
            return {"error": "At least one element of strings list is not a string."}
        if len(s) > 64:
            return {"error": "At least one element of strings list is longer than 64 characters."}

    if not timeout_disabled:
        if (time.time() - prev_submission_time) < SUBMISSION_TIME:
            rem_time = round(SUBMISSION_TIME - (time.time() - prev_submission_time) + 2)
            if rem_time < 60:
                return {"error": f"You must wait {rem_time}s before next submission"}
            else:
                return {"error": f"You must wait {rem_time // 60}min {rem_time % 60}s before next submission"}
    
    predicted_embeddings = np.concatenate([query(strings[:100]), query(strings[100:])])
    cos_sim = (predicted_embeddings * target_embeddings).sum(1).mean().item()
    accur = (abs(predicted_embeddings - target_embeddings).max(axis=1) < 1e-6).astype(int).sum().item() / len(target_embeddings)
    points = 50 * compute_score(cos_sim, 0.2, 1.0) + 50 * compute_score(accur, 0.01, 1.0)
    
    result = {
        'average_cosine_similarity': cos_sim,
        'exact_reconstruction_accuracy': accur,
        'final_score': points
    }
    prev_submission_time = time.time()
    prev_submission_result = result
    return result

def disable_timeout():
    global timeout_disabled
    timeout_disabled = True
