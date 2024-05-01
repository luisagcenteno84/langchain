from langchain_google_vertexai import VertexAIEmbeddings
from google.cloud import bigquery
from langchain.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import BigQueryVectorSearch
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ID = "solid-sun-418711"
REGION = "US"
DATASET = "chat_rag"
TABLE = "docs_and_vectors"


client = bigquery.Client(project=PROJECT_ID, location=REGION)
client.create_dataset(dataset=DATASET, exists_ok=True)

embedding = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest", project_id=PROJECT_ID
)

vectorstore = BigQueryVectorSearch(
    project_id=PROJECT_ID,
    dataset_name=DATASET,
    table_name=TABLE,
    location=REGION,
    embedding=embedding,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
)


loader = WebBaseLoader("https://plato.stanford.edu/entries/socrates/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)


vectorstore.add_documents(all_splits)