import os
import pandas as pd

from tqdm import tqdm
from typing import Union

from langchain_core.embeddings import Embedding
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jVector

from ragas.dataset_schema import EvaluationResult
from ragas import EvaluationDataset
from ragas import evaluate


async def generate_qr_data(
        df: pd.DataFrame, 
        llm: ChatOpenAI,
        metadata_fields: list = ["description", "title"],
        num_questions_per_chunk: int = 1,
    ) -> pd.DataFrame:
    """
    Generate QR data for the given DataFrame.
    """
    from llama_index.core import Document
    from llama_index.core.evaluation import DatasetGenerator
    cols = df.columns

    docs = []
    for i, row in df.iterrows():
        doctext = [f"{col}: {row[col]}" for col in cols]
        doctext = "\n".join(doctext)
        doc = Document(
            text=doctext,
            metadata={field: row[field] for field in metadata_fields},
        )
        docs.append(doc)

    dataset_generator = DatasetGenerator.from_documents(
        docs,
        num_questions_per_chunk=num_questions_per_chunk,
        show_progress=True,
    )
    qr_ds = await dataset_generator.agenerate_dataset_from_nodes()
    qr_ds.save_json(os.environ["QR_DATA"])
    qr_ds = {
        "queries": qr_ds.queries,
        "responses": qr_ds.responses,
    }
    return qr_ds


def get_graphstore(
        url: str, 
        username: str,
        password: str,
        embed: Embedding,
    ) -> Neo4jVector:
    """
    Create a Neo4j vector store.
    """
    graph_vecstore = Neo4jVector.from_existing_graph(
        embedding=embed,
        url=url,
        username=username,
        password=password,
        index_name="product_index",
        node_label="Product",
        text_node_properties=["description"],
        embedding_node_property="embedding",
    )
    return graph_vecstore


def eval_qr_data(
        questions: list[str], 
        responses: list[str], 
        retriever: Union[VectorStoreRetriever, Neo4jVector], 
        llm: ChatOpenAI,
    ) -> EvaluationDataset:
    """
    Wraps the evaluation data into a list of dictionaries.
    """
    dataset = []

    for query, reference in tqdm(zip(questions, responses), total=len(questions)):

        if isinstance(retriever, VectorStoreRetriever):
            relevant_docs = retriever.invoke(query)
        elif isinstance(retriever, Neo4jVector):
            relevant_docs = retriever.similarity_search(query, k=TOP_K)
        else:
            raise ValueError("Unsupported retriever type")
        
        relevant_context = "\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"""
        Based on the following context, answer the question:\n
        {relevant_context}\n
        Question: {query}\n
        Answer:
        """
        response = llm.invoke(prompt).content
        dataset.append(
            {
                "user_input":query,
                "retrieved_contexts":[d.page_content for d in relevant_docs],
                "response":response,
                "reference":reference
            }
        )

    return EvaluationDataset.from_list(dataset)


def parse_eval_results(
        eval_result: Union[EvaluationResult, pd.DataFrame], 
        method_name: str,
    ) -> pd.DataFrame:
    """
    Parses the evaluation results into a DataFrame.
    """
    if isinstance(eval_result, EvaluationResult):
        # Convert the evaluation result to a DataFrame
        eval_df = eval_result.to_pandas()
        metrics_names = [i for i in eval_result.scores[0].keys()]
    else:
        # If it's already a DataFrame, use it directly
        eval_df = eval_result
        metrics_names = [col for col in eval_result.columns if col not in ["user_input", "retrieved_contexts", "response", "reference"]]
    eval_df_long = pd.melt(
        eval_df,
        value_vars=metrics_names,  # Columns to unpivot
        var_name='metric',             # Name for the variable column
        value_name='score'              # Name for the value column
    )
    eval_df_long['method'] = method_name
    return eval_df_long