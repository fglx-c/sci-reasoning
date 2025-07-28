import os
import json
import torch
from adapters import AutoAdapterModel
from sentence_transformers import util
from transformers import AutoTokenizer

# Load Specter v2 model and tokenizer using the adapter interface
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")

# Load and activate the adapter for Specter v2
adapter_name = model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

# Load tokenizer for Specter v2
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")

# Function to generate embeddings using Specter v2
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over tokens
    return embedding

# Function to retrieve top 5 similar background sentences
def get_top_5_similar_backgrounds(query_embedding, background_embeddings, background_sentences, result_sentences):
    hits = util.semantic_search(query_embedding, background_embeddings, top_k=5)[0]
    
    similar_backgrounds = []
    corresponding_results = []
    for hit in hits:
        idx = hit['corpus_id']
        similar_backgrounds.append(background_sentences[idx])
        corresponding_results.append(result_sentences[idx])
    
    return similar_backgrounds, corresponding_results

# Load the training data (background and results) - assuming multiple JSON objects are on separate lines in the file
with open("/Users/bill/Desktop/AI2/SN/abstract-annotations-small.json", "r") as f:
    training_data = [json.loads(line) for line in f]

background_sentences = []
result_sentences = []
background_embeddings = []

# Process and encode the training data
for entry in training_data:
    for sentence, category in entry.items():
        if category == "Background":
            background_sentences.append(sentence)
            background_embeddings.append(get_embedding(sentence))
        elif category == "Result":
            result_sentences.append(sentence)

# Convert list of embeddings to tensor
background_embeddings = torch.stack(background_embeddings)

# Function to retrieve results corresponding to similar background sentences
def retrieve_results_for_background(input_background):
    query_embedding = get_embedding(input_background)
    similar_backgrounds, results = get_top_5_similar_backgrounds(query_embedding, background_embeddings, background_sentences, result_sentences)
    
    print("Input Background Sentence:")
    print(input_background)
    print("\nTop 5 Similar Background Sentences:")
    for idx, similar_background in enumerate(similar_backgrounds):
        print(f"{idx+1}. {similar_background}")
    
    print("\nCorresponding Results:")
    for idx, result in enumerate(results):
        print(f"{idx+1}. {result}")
    
    return similar_backgrounds, results

input_background = "This work reports on the spectral dependence of both nonlinear refraction and absorption in lead-germanium oxide glasses (PbO-GeO\u2082) containing silver nanoparticles."
retrieve_results_for_background(input_background)
