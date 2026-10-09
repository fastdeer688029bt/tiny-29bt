"""Tiny embedding similarity search utility.

Usage:
    python embed_search.py --file embeddings.txt --query 0.1 0.2 0.3 --top 3
"""

import argparse, math

def parse_embeddings(path):
    emb={}
    with open(path) as f:
        for line in f:
            parts=line.strip().split()
            if not parts: continue
            key, vals=parts[0], list(map(float, parts[1:]))
            emb[key]=vals
    return emb

def cosine_similarity(a,b):
    dot=sum(x*y for x,y in zip(a,b))
    na=math.sqrt(sum(x*x for x in a))
    nb=math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0

def search(query,data,k=5):
    sims=sorted(((cosine_similarity(query,vec),k2) for k2,vec in data.items()), reverse=True)
    return sims[:k]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--file', required=True, help='Embedding file')
    p.add_argument('--query', nargs='+', type=float, required=True, help='Query vector')
    p.add_argument('--top', type=int, default=5, help='Top k results')
    a=p.parse_args()
    data=parse_embeddings(a.file)
    for score,key in search(a.query,data,a.top):
        print(f'{key}\t{score:.4f}')

if __name__=='__main__':
    main()