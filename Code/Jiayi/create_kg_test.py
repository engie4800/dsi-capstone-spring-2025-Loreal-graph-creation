import os
import time

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship

from dotenv import load_dotenv
load_dotenv()

DOCS_PATH = "./data/cosmetics_25_products.csv"

llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'), 
    model_name="gpt-4o-mini"
)

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="text-embedding-ada-002"
    )

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

# doc_transformer = LLMGraphTransformer(
#     llm=llm,
#     allowed_nodes=[
#     "Brand", "Category", "Product", "Price", 
#     "Rating", "Skin Type", "Size", "Description"
#     ],
#     allowed_relationships=[
#     "FROM_BRAND", "HAS_PRICE", "HAS_RATING", "FOR_SKIN_TYPE",
#     "IN_SIZE", "HAS_DESCRIPTION", "BELONGS_TO"
#     ],
#     node_properties=["productName", "ingredients", "size", "rating", "price",
#                      "skinType", "description", "brandName", "categoryName"],
#     strict_mode=True
#     )

doc_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes = ["Product"],
    node_properties=["productName", "ingredients", "size", "rating", "price",
                     "forSkinType", "description", "brand", "category"],
    strict_mode=True
    )

# Load and split the documents
loader = CSVLoader(file_path=DOCS_PATH)

text_splitter = CharacterTextSplitter(
    separator="\n\n"
)

docs = loader.load()
chunks = text_splitter.split_documents(docs)

# construction_start_time = time.time()
# for chunk in chunks:

#     filename = os.path.basename(chunk.metadata["source"])
#     chunk_id = f"{filename}.{chunk.metadata.get('row', '')}"
#     print("Processing -", chunk_id)

#     # Embed the chunk
#     chunk_embedding = embedding_provider.embed_query(chunk.page_content)

#     # Add the Document and Chunk nodes to the graph
#     properties = {
#         "filename": filename,
#         "chunk_id": chunk_id,
#         "text": chunk.page_content,
#         "embedding": chunk_embedding
#     }
    
#     graph.query("""
#         MERGE (d:Document {id: $filename})
#         MERGE (c:Chunk {id: $chunk_id})
#         SET c.text = $text
#         MERGE (d)<-[:PART_OF]-(c)
#         WITH c
#         CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
#         """, 
#         properties
#     )

#     # Generate the entities and relationships from the chunk
#     graph_docs = doc_transformer.convert_to_graph_documents([chunk])

#     # Map the entities in the graph documents to the chunk node
#     for graph_doc in graph_docs:
#         chunk_node = Node(
#             id=chunk_id,
#             type="Chunk"
#         )

#         for node in graph_doc.nodes:
#             if node.type == "Product":
#                 graph_doc.relationships.append(
#                     Relationship(
#                         source=chunk_node,
#                         target=node, 
#                         type="HAS_PRODUCT"
#                     )
#                 )

#     # add the graph documents to the graph
#     graph.add_graph_documents(graph_docs)
# construction_end_time = time.time()
construction_start_time = time.time()
for doc in docs:

    filename = os.path.basename(doc.metadata["source"])
    chunk_id = f"{filename}.{doc.metadata.get('row', '')}"
    print("Processing -", chunk_id)

    # Embed the chunk
    chunk_embedding = embedding_provider.embed_query(doc.page_content)

    # Add the Document and Chunk nodes to the graph
    properties = {
        "filename": filename,
        "chunk_id": chunk_id,
        "text": doc.page_content,
        "embedding": chunk_embedding
    }
    
    graph.query("""
        MERGE (d:Document {id: $filename})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text
        MERGE (d)<-[:PART_OF]-(c)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
        """, 
        properties
    )

    # Generate the entities and relationships from the chunk
    graph_docs = doc_transformer.convert_to_graph_documents([doc])

    # Map the entities in the graph documents to the chunk node
    for graph_doc in graph_docs:
        chunk_node = Node(
            id=chunk_id,
            type="Chunk"
        )

        for node in graph_doc.nodes:
            if node.type == "Product":
                graph_doc.relationships.append(
                    Relationship(
                        source=chunk_node,
                        target=node, 
                        type="HAS_PRODUCT"
                    )
                )

    # add the graph documents to the graph
    graph.add_graph_documents(graph_docs)
construction_end_time = time.time()
print(f"Total time use to construction knowledge graph: {construction_end_time - construction_start_time} seconds")

# Create the vector index
graph.query("""
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
    }};""")

# Print total counts of nodes and relationships
node_count = graph.query("MATCH (n) RETURN count(n) as count")[0]["count"]
rel_count = graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]["count"]

print(f"\nTotal nodes created: {node_count}")
print(f"Total relationships created: {rel_count}")

# Print counts by node types
node_types = graph.query("""
    MATCH (n)
    RETURN labels(n) as type, count(*) as count
    ORDER BY count DESC
""")
print("\nNode counts by type:")
for record in node_types:
    print(f"  {record['type']}: {record['count']}")

# Print counts by relationship types
rel_types = graph.query("""
    MATCH ()-[r]->()
    RETURN type(r) as type, count(*) as count
    ORDER BY count DESC
""")
print("\nRelationship counts by type:")
for record in rel_types:
    print(f"  {record['type']}: {record['count']}")