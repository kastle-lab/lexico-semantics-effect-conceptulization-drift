import os
import json
import time
import pandas as pd
from datetime import datetime
from google import genai
import google.auth
from google.genai import types
from google.cloud import storage

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "gemini-2.5-pro"
# BUCKET_NAME = "sparql-gen-graphrag"
BUCKET_NAME = "saini-research"
DIRECTORY_NAME = "structural-lexico-semantic-effects-graphrag"
LOCATION = "us-central1"
TEMPERATURE = 1.0

# File names
# SCHEMA_FILE_PATH = "big_schema_trim.ttl"
SCHEMA_FILE_PATH = "small_schema_trim.ttl"
EXCEL_OUTPUT_FILE = f"{os.path.basename(SCHEMA_FILE_PATH).replace('.ttl', '')}_cq_Gemini_results.xlsx"

# GCS Prefixes (using timestamps to avoid overwriting previous runs)
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
# GCS_INPUT_DIR = f"gs://{BUCKET_NAME}/inputs_{run_id}/"
# GCS_OUTPUT_DIR = f"gs://{BUCKET_NAME}/batch_results_{run_id}/"
GCS_INPUT_DIR = f"gs://{BUCKET_NAME}/{DIRECTORY_NAME}/inputs_{run_id}/"
GCS_OUTPUT_DIR = f"gs://{BUCKET_NAME}/{DIRECTORY_NAME}/batch_results_{run_id}/"

# ==========================================
# AUTHENTICATION
# ==========================================
credentials, project_id = google.auth.default()
client = genai.Client(
    vertexai=True,
    project=project_id,
    location=LOCATION,
    credentials=credentials
)
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_file_to_string(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def fill_prompt_template(template_text, values_dict):
    for key, value in values_dict.items():
        template_text = template_text.replace(f"{{{key}}}", value)
    return template_text

def wait_for_job(job_name, client):
    """Polls the Vertex AI job until it succeeds or fails."""
    print(f"Polling job {job_name}...")
    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name
        if state == 'JOB_STATE_SUCCEEDED':
            print(f"\nJob {job_name} SUCCEEDED!")
            return True
        elif state in ['JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_PARTIALLY_SUCCEEDED']:
            print(f"\nJob {job_name} stopped with state: {state}")
            if job.error:
                 print(f"Error: {job.error}")
            return False
        print(".", end="", flush=True)
        time.sleep(30)

# ==========================================
# PROMPTS & DATA
# ==========================================
CQs = [
    # Simple
    "Who are all the characters available?",
    "What are all the movies (and/or TV shows) available?",
    "What are the real names and/or primary aliases for each of the characters?",
    "What are all the associated species or types (e.g., human, Asgardian, AI) for each of the characters?",
    "What are all the origin locations (e.g., homeworld, birthplace, base) for each of the characters?",
    "What are all the release dates (or years), if available, for each of the movies?",
    "Who are all the director(s) for each of the movies?",
    "What are all the listed powers or abilities for each of the characters?",
    "What are all the team or organization affiliations for each of the characters?",
    "What are all the locations (places) available?",
    "What are all the available movies and their directors?",
    "What are all the available characters and their primary alias/real name?",
    # Moderate
    "What are all the character–movie appearance pairs (which characters appear in which movies) available?",
    "What are all the actor–character pairs (which actors portray which characters) available?",
    "What are all the character–team–movie triples available where a character is a member of a team and appears in a movie?",
    "What are all the character–power–movie triples available with respect to characters with specific powers and the movies they appear in?",
    "What are all the pairs of characters that are linked through movies and co-appear in at least two movies?",
    "What are all the teams and the set of members for each team available?",
    "What are all the movies and the set of teams that have at least one member appearing in them?",
    "What are all locations that are associated with at least one character appearance (e.g., origin or major setting)?",
    "What are all the director–actor pairs linked through movies available?",
    "What are all the available distinct powers and the set of characters associated with each power?",
    "What are all the pairs of characters available that have co-appeared in a movie?",
    # Complex
    "How many movies does each character appear in (character appearance count)?",
    "What are all the movies and their counts for each pair of characters that co-appear in multiple movies?",
    "What are all the unions of movies for each of the teams in which any of their members appear (team-level filmography)?",
    "Who are the distinct characters that possess a power, their counts, and rank-ordered by popularity?",
    "Who are all the bridge characters that are members of more than one team, and what are those team combinations?",
    "What are all the sets of characters that have been portrayed across all movies by each of the actors?",
    "What are all the locations used as settings or associated contexts for multiple movies and/or characters and the counts of those associations?",
    "What is the number of unique teams, unique powers, and unique locations represented via their characters for each of the movies?",
    "Who are all the other characters connected via shared movie appearances for each of the characters (character co-appearance network)?",
    "What are all the distributions of powers among each team's members, grouped by teams, for comparing teams for computability (e.g., which powers are most characteristic of each team)?"
]

initial_system_message = """
You are an expert in knowledge graphs and SPARQL query generation. Your task is to generate SPARQL queries based on the provided competency questions and a given TTL schema and return only the SPARQL query.

Guidelines:
Use only the schema provided in the context block to determine appropriate classes, properties, and relationships.
 - Ensure queries follow SPARQL syntax and use prefixes correctly.
 - Generate queries that efficiently retrieve relevant data while optimizing performance but with priority on correctness and efficiency.
 - If multiple valid queries exist, choose the most concise and efficient one.
 - Preserve the intent of the competency question while ensuring syntactic correctness.
 - Give only one SPARQL query and nothing else.
 - Only use the defined relationships in the schema. Don't use external ones unless specified.
 - If the competency question cannot be answered with the provided schema, respond to a partial extent that it can be answered to or respond with "No valid query can be generated based on the provided schema."
 - Don't summarize or return an analysis of the given schema but return only the respective SPARQL query for the Competency Question.
"""

template_prompt = """
Task: Write a SPARQL query that answers the following competency question:
{Insert_CQ_here}

Requirements:
- Use the schema to determine correct URIs and relationships.
- Ensure the query retrieves the necessary information efficiently.
- Provide only one full SPARQL query without placeholders.
- Don't summarize or return an analysis of the given schema but return only the respective SPARQL query for the Competency Question.

Context:
Below is the TTL schema of the knowledge graph:
{Insert_schema_here}
"""

# ==========================================
# MAIN PIPELINE EXECUTION
# ==========================================
def run_pipeline():
    # 1. Load schema
    if not os.path.exists(SCHEMA_FILE_PATH):
        print(f"Error: Schema file '{SCHEMA_FILE_PATH}' not found.")
        return
        
    schema_string = load_file_to_string(SCHEMA_FILE_PATH)

    # 2. Build Batch JSONL File
    print(f"--- GENERATING BATCH FOR {len(CQs)} COMPETENCY QUESTIONS ---")
    json_requests = []
    
    # Store the actual filled prompts locally so we can map them back later for the Excel file
    prompt_mapping = {}

    for i, cq in enumerate(CQs):
        custom_id = f"cq-{i}"
        
        input_data = {
            "Insert_CQ_here": cq,
            "Insert_schema_here": schema_string
        }
        
        filled_prompt = fill_prompt_template(template_prompt, input_data)
        prompt_mapping[custom_id] = filled_prompt
        
        json_requests.append({
            "key": custom_id,
            "request": {
                "system_instruction": {"parts": [{"text": initial_system_message}]},
                "contents": [{"role": "user", "parts": [{"text": filled_prompt}]}],
                "generationConfig": {"temperature": TEMPERATURE}
            }
        })

    batch_input_file = "sparql_batch_input.jsonl"
    with open(batch_input_file, 'w', encoding='utf-8') as f:
        for req in json_requests:
            f.write(json.dumps(req) + '\n')

    # 3. Upload & Trigger Job
    print(f"Uploading {batch_input_file} to GCS...")
    # bucket.blob(f"inputs_{run_id}/{batch_input_file}").upload_from_filename(batch_input_file)
    bucket.blob(f"{DIRECTORY_NAME}/inputs_{run_id}/{batch_input_file}").upload_from_filename(batch_input_file)

    print("Triggering Vertex AI Batch Job...")
    batch_job = client.batches.create(
        model=MODEL_NAME,
        src=f"{GCS_INPUT_DIR}{batch_input_file}",
        config=types.CreateBatchJobConfig(
            dest=GCS_OUTPUT_DIR,
            display_name="sparql-generation-batch"
        )
    )
    
    if not wait_for_job(batch_job.name, client):
        print("Batch Job Failed. Exiting.")
        return

    # 4. Parse Results
    print("\n--- PROCESSING RESULTS ---")
    results_dict = {}
    
    # blobs = bucket.list_blobs(prefix=f"batch_results_{run_id}/")
    blobs = bucket.list_blobs(prefix=f"{DIRECTORY_NAME}/batch_results_{run_id}/")
    for blob in blobs:
        if blob.name.endswith('.jsonl'):
            content = blob.download_as_text()
            for line in content.strip().split('\n'):
                if not line: continue
                data = json.loads(line)
                
                custom_id = data.get("key", "")
                
                if "candidates" in data.get("response", {}):
                    response_text = data["response"]["candidates"][0]["content"]["parts"][0]["text"].strip()
                    results_dict[custom_id] = {
                        "result": response_text,
                        "raw": json.dumps(data, indent=4)
                    }
                else:
                    results_dict[custom_id] = {
                        "result": f"Error: {data.get('status', 'Unknown error')}",
                        "raw": json.dumps(data, indent=4)
                    }

    # 5. Map results back to original CQs and format for Excel
    cq_gemini_results = []
    for i, cq in enumerate(CQs):
        custom_id = f"cq-{i}"
        
        filled_prompt = prompt_mapping.get(custom_id, "Prompt not found")
        res = results_dict.get(custom_id, {"result": "No response returned", "raw": "{}"})
        
        cq_gemini_results.append((cq, filled_prompt, res["result"], res["raw"]))

    # 6. Save to Excel
    df = pd.DataFrame(cq_gemini_results, columns=["CQ", "Prompt", "Gemini_Result", "Gemini_Raw"])
    df.to_excel(EXCEL_OUTPUT_FILE, index=False)
    print(f"Success! Excel file saved to: {EXCEL_OUTPUT_FILE}")

if __name__ == "__main__":
    run_pipeline()