from langchain_google_vertexai import VertexAIEmbeddings
from google.cloud import bigquery
from langchain.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import BigQueryVectorSearch
#from google.colab import auth as google_auth

PROJECT_ID = "solid-sun-418711"
REGION = "US"
DATASET = "my_langchain_dataset"
TABLE = "docs_and_vectors"

#google_auth.authenticate_user


embedding = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest", project_id=PROJECT_ID
)

#print(embedding)

client = bigquery.Client(project=PROJECT_ID, location=REGION)
client.create_dataset(dataset=DATASET, exists_ok=True)


store = BigQueryVectorSearch(
    project_id=PROJECT_ID,
    dataset_name=DATASET,
    table_name=TABLE,
    location=REGION,
    embedding=embedding,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
)

query = "I'd like a fruit"

#docs = store.similarity_search(query)
#print(docs)

query_vector = embedding.embed_query(query)
#docs = store.similarity_search_by_vector(query_vector, k=2)
#print(docs)

#docs = store.similarity_search_by_vector(query_vector, filter={"len": 6}) 
#print(docs)

job_id = "d53e27ff-c848-4835-8f1a-859c544b3b64"
print(store.explore_job_stats(job_id))