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

all_texts = ["Apples and oranges", "Cars and airplanes", "Pineapple", "Train", "Banana"]
metadatas = [{"len": len(t)} for t in all_texts]

store.add_texts(all_texts, metadatas=metadatas)

