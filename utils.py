import json
import numpy as np
import httpx
import os

from constants import MUBERT_TAGS, MUBERT_MODE, MUBERT_LICENCE, MUBERT_TOKEN

def get_mubert_tags_embeddings(w2v_model):
    return w2v_model.encode(MUBERT_TAGS)


def get_pat(email: str):
    r = httpx.post('https://api-b2b.mubert.com/v2/GetServiceAccess',
                   json={
                       "method": "GetServiceAccess",
                       "params": {
                           "email": email,
                           "license": MUBERT_LICENCE,
                           "token": MUBERT_TOKEN,
                           "mode": MUBERT_MODE,
                       }
                   })

    rdata = json.loads(r.text)
    assert rdata['status'] == 1, "probably incorrect e-mail"
    pat = rdata['data']['pat']
    return pat


def find_similar(em, embeddings, method='cosine'):
    scores = []
    for ref in embeddings:
        if method == 'cosine':
            scores.append(1 - np.dot(ref, em) / (np.linalg.norm(ref) * np.linalg.norm(em)))
        if method == 'norm':
            scores.append(np.linalg.norm(ref - em))
    return np.array(scores), np.argsort(scores)


def get_tags_for_prompts(w2v_model, mubert_tags_embeddings, prompts, top_n=3, debug=False):
    prompts_embeddings = w2v_model.encode(prompts)
    ret = []
    for i, pe in enumerate(prompts_embeddings):
        scores, idxs = find_similar(pe, mubert_tags_embeddings)
        top_tags = MUBERT_TAGS[idxs[:top_n]]
        top_prob = 1 - scores[idxs[:top_n]]
        if debug:
            print(f"Prompt: {prompts[i]}\nTags: {', '.join(top_tags)}\nScores: {top_prob}\n\n\n")
        ret.append((prompts[i], list(top_tags)))
    return ret