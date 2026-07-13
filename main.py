
from gitsource import GithubRepositoryDataReader
data_gen_instructions = """
You emulate a student who is taking our LLM course.
You are given one lesson page from the course.
Formulate 5 questions this student might ask that are answered by this page.

Rules:
- The page should contain the answer to each question.
- Make the questions complete and not too short.
- Use as few words as possible from the page; don't copy its phrasing.
- The questions should resemble how people actually ask things online:
  not too formal, not too short, not too long.
- Ask about the content of the lesson, not about its formatting or filename.
""".strip()
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]



from pydantic import BaseModel

class Questions(BaseModel):
    questions: list[str]

from openai import OpenAI
from dotenv import load_dotenv 
import requests

import os                                                                                                                                                                                                          
from pathlib import Path
load_dotenv(Path("/workspaces/04-evaluation/.venv/.env"))
##print(os.getenv("OPENAI_API_KEY"))

#load_dotenv()
True
openai_client = OpenAI(


     api_key=os.getenv("OPENAI_API_KEY"),
      base_url="https://api.openai.com/v1",
)
import json






from evaluation_utils import llm_structured_retry
def generate_ground_truth(doc):
    user_prompt = json.dumps(doc)

    out, usage = llm_structured_retry(
        openai_client,
        data_gen_instructions,
        user_prompt,
        Questions
    )

    results = []

    for q in out.questions:
        results.append({
            "question": q,
            "document": doc["filename"]
        })

    return results, usage


#print ("######################")

from tqdm.auto import tqdm

ground_truth = []
usages = []
moyen=[]


def average (number) :
    average =0
    for doc in tqdm(documents[:3]): 
      records, usage  = generate_ground_truth(doc)
      ground_truth.extend(records)
      usages.append(usage)
     
      average =usage.input_tokens /number + average
    return average

average=average (3)
print("What's the average number of input tokens across these 3 calls?")
print("Response is {first}".format(first=average)) 
from concurrent.futures import ThreadPoolExecutor
from evaluation_utils import map_progress


with ThreadPoolExecutor(max_workers=6) as pool:
    results = map_progress(pool, documents, generate_ground_truth)

ground_truth = []
usages = []

for records, usage in results:
    ground_truth.extend(records)
    usages.append(usage)
len(ground_truth)
import pandas as pd

df_ground_truth = pd.DataFrame(ground_truth)
df_ground_truth.to_csv("data/ground_truth-new.csv", index=False)

from gitsource import chunk_documents

from minsearch import Index

from embedder import Embedder

chunks = chunk_documents(documents, size=2000, step=1000)
texts = [doc["filename"] + " " + doc["content"] for doc in chunks]
index_chunk = Index(
    text_fields=["content"],
    keyword_fields=["filename"]
)
index_chunk.fit(chunks)

def text_search(query: str,num_results ):

    """
    Search for entries matching the given query.
    """
    
    
    return index_chunk.search(
      
        query,
        num_results=num_results,
       # boost_dict = {'content': 3.0, 'filename': 0.5},
        
    )


model = Embedder()
from tqdm.auto import tqdm
import numpy as np
from minsearch import VectorSearch

def vector_search(query,num_results):

  X= []



  batch_size = 50
  vectors = []

  for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i + batch_size]
    batch_vectors = model.encode_batch(batch)
    batch_vectors.shape
    vectors.extend(batch_vectors)

  X = np.array(vectors)
  v_query= model.encode(query)
  vindex = VectorSearch(keyword_fields=["filename"])
  vindex.fit(X, chunks)
  vectors_results =vindex.search(v_query,num_results=num_results)
  return vectors_results

def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]
 
def hybrid_search(query, k):
    text_results = text_search(query, 10)
    vector_results = vector_search(query,10)
    return rrf([text_results, vector_results], k)


q = ground_truth[0]["question"]

filename=ground_truth[0]["document"]

results = text_search(q,num_results=5)
print (results)
First =results[0]
print("After running text_search for the first ground truth question, what is the filename of the first result? ")
print("Response is {first}".format(first=First['filename'])) 

result2=vector_search(q,num_results=5)
First =result2[0]
print("After running text_search for the first ground truth question, what is the filename of the first result? ")
print("Response is {first}".format(first=First['filename'])) 




def compute_relevance(q, search_function, num_results):
    filename= q["document"]
    query=q["question"]
    results = search_function(query, num_results)
    
    relevance = []
    for d in results:
        relevance.append(int(d["filename"] == filename))

    return relevance


from tqdm.auto import tqdm

def compute_relevance_total(ground_truth,search_function,*args, **kwargs):
    relevance_total = []

    for q in tqdm(ground_truth):
        relevance = compute_relevance(q,search_function,*args, **kwargs)
        relevance_total.append(relevance)

    return relevance_total


relevance = compute_relevance_total(ground_truth,text_search,5)
#print(relevance)


def hit_rate(relevance):
    cnt = 0

    for line in relevance:
        if 1 in line:
            cnt = cnt + 1

    return cnt / len(relevance)

hrate= hit_rate(relevance)
print (hrate)

relevance_vs = compute_relevance_total(ground_truth,vector_search,5)
def mrr(relevance_vs):
    total_score = 0.0

    for line in relevance_vs:
        for rank in range(len(line)):
            if line[rank] == 1:
                total_score = total_score + 1 / (rank + 1)
                break

    return total_score / len(relevance_vs)


#mrrvs=mrr(relevance_vs)
#print(mrrvs)

def evaluate(ground_truth, search_function, *args, **kwargs):
    relevance_total = compute_relevance_total(ground_truth, search_function,*args, **kwargs)

    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }
resu= evaluate(
    ground_truth,
    hybrid_search,
    1
   
)
print(resu)
results = []

for K in [0.1, 0.2, 0.5]:
    result = evaluate(
    ground_truth,
    hybrid_search,
    K
   
)
    results.append({
                
                "hit_rate": result["hit_rate"],
                "mrr": result["mrr"],
                })
    
df_results = pd.DataFrame(results)
best =df_results.sort_values("mrr", ascending=False).head(1)
print (best)
