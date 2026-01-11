### Steps to start docker for neo4j
- docker compose up -d
- docker compose stop
- docker compose start

### Steps for setting up nodes and edges
- mkdir -p ~/neo4j-data ~/neo4j-import
- cp nodes_fixed.csv ~/neo4j-import/nodes.csv
- cp edges_fixed.csv ~/neo4j-import/edges.csv


### Sample backend hits / CURLs
- http://localhost:8000/search?q=policy&-speaker=Duisenberg&from_date=2018-01-01&-to_date=2022-12-31
- http://localhost:8000/search?q=Monetary%20policy
- http://localhost:8000/search?q=inflation&-from_date=2020-01-01
- http://localhost:8000/search?q=inflation&-from_date=2020-01-01&to_date=2024-03-20

## Steps to run application

neo4j :
- cd ./app
- docker compose start

server :
- cd ./app
- uvicorn api:app --reload

client :
- cd ./client
- streamlit run ./streamlit.py



### Developer's column

#### Existing pipeline flow for query search
normalizer.py → translator.py → kg_executor.py → retriever.py → formatter.py → verifier.py

#### Where each file fits
##### normalizer.py
- Normalize query text
- Token cleanup
- No KG logic

##### translator.py
Decides search mode
For now: intent = "TOPIC_SEARCH"

##### kg_executor.py
-  NO string matching
-  Only executes Cypher given:
-  topic_ids OR fulltext_query

##### retriever.py (key evolution)
Neo4j, full-text indexing on Topic.label

##### formatter.py
-  Highlights topic snippet
-  Attaches provenance
-  No retrieval logic

##### verifier.py
-  Ensures:     ≥1 Speech-[:COVERS]->Topic  else verified=false

##### pipeline.py
-  Orchestrates:   normalize → retrieve → ground → format → verify




### Headings answering 
#### How hallucination dies

#### Why word order doesn’t matter anymore
Because retrieval layers are order-agnostic:

Full-text index
Token-based
Order-independent
Partial match
Scored by relevance

So → same candidate set for :
- "monetary inflation policy"
- "policy inflation monetary"
- "impact of monetary policy on inflation"


#### Issue : multiple speeches have same content causing issue
Try for below speech entry which has 5+ dates attached and verified on original DB

- Speech_id = SPE000019
- Speech : Hearing of the Committee on Economic and Monetary Affairs of the European Parliament

