import os
import json
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Path to the LLaMA-3.1-8B-Instruct  model
model_path = "/path/to/Llama-3.1-8B-Instruct"

# Load the model using vLLM
llm = LLM(model=model_path, tensor_parallel_size=1)
tokenizer = llm.get_tokenizer()

# Define the prompt template
prompt_template = (
    "Extract the main words or phrases that best summarize the"
    "following sentence. Return only the words from the sentence in"
    "a space-separated format, without extra explanations or examples.\n\n"
    "Example 1:\nSentence: \"The teacher explains a complex math problem.\"\nKeywords: teacher, explains, math problem\n"
    "Example 2:\nSentence: \"A boy is playing soccer in the park.\"\nKeywords: boy, playing, soccer, park\n\n"
    "Now, extract the keywords from the following sentence:\n\"{}\""
)

# Prepare the input batch for parallel processing
def prepare_input_batch(sentences):
    myinput = []
    for sentence in sentences:
        input_text = prompt_template.format(sentence)
        myinput.append([{'role': 'user', 'content': input_text}])
    return myinput

# Extract keywords in batches
def extract_keywords_batch(sentences):
    # Prepare input in batch format
    myinput = prepare_input_batch(sentences)

    # Convert input into a format compatible with vLLM
    conversations = tokenizer.apply_chat_template(myinput, tokenize=False)

    # Set sampling parameters for batch generation
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id]
    )
    
    # Generate results using vLLM model
    outputs = llm.generate(conversations, sampling_params)

    # Process the output and return the list of keywords
    keywords_list = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        cleaned_text = generated_text.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
        keywords_list.append(cleaned_text.split())  # Split the keywords into a list by spaces
    return keywords_list

# Process the list of candidate sentences in batches and extract keywords
def process_cand_list(cands, batch_size=256):
    cand_keywords = []
    for i in tqdm(range(0, len(cands), batch_size), desc="Processing cand"):
        batch = cands[i:i+batch_size]
        keywords_batch = extract_keywords_batch(batch)
        cand_keywords.extend(keywords_batch)
    return cand_keywords

if __name__ == '__main__':
    # Load candidate sentences from a pickle file
    samples_list = pickle.load(open('/path/to/vatex/candidates_list.pkl', 'rb'))

    cands = samples_list.tolist()

    # Extract keywords from the candidate sentences
    cand_keywords = process_cand_list(cands)

    # Save the extracted keywords to a JSON file, preserving the original structure
    with open('/path/to/result-keywords.json', 'w', encoding='utf-8') as f:
        json.dump(cand_keywords, f, ensure_ascii=False, indent=4)

    print("Keywords extraction completed and saved to JSON files.")
