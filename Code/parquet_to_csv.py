import pandas as pd

GRAPH_OUTPUT_DIRECTORY="Code/beautyragtest"
embeddings_on=True

df = pd.read_parquet(f'{GRAPH_OUTPUT_DIRECTORY}/output/communities.parquet')
df.to_csv(f'{GRAPH_OUTPUT_DIRECTORY}/communities.csv')

df = pd.read_parquet(f'{GRAPH_OUTPUT_DIRECTORY}/output/community_reports.parquet')
df.to_csv(f'{GRAPH_OUTPUT_DIRECTORY}/community_reports.csv')

df = pd.read_parquet(f'{GRAPH_OUTPUT_DIRECTORY}/output/documents.parquet')
df.to_csv(f'{GRAPH_OUTPUT_DIRECTORY}/documents.csv')

df = pd.read_parquet(f'{GRAPH_OUTPUT_DIRECTORY}/output/entities.parquet')
df.to_csv(f'{GRAPH_OUTPUT_DIRECTORY}/entities.csv')

df = pd.read_parquet(f'{GRAPH_OUTPUT_DIRECTORY}/output/relationships.parquet')
df.to_csv(f'{GRAPH_OUTPUT_DIRECTORY}/relationships.csv')

df = pd.read_parquet(f'{GRAPH_OUTPUT_DIRECTORY}/output/text_units.parquet')
df.to_csv(f'{GRAPH_OUTPUT_DIRECTORY}/text_units.csv')

if embeddings_on:
    df = pd.read_parquet(f'{GRAPH_OUTPUT_DIRECTORY}/output/embeddings.entity.description.parquet')
    df.to_csv(f'{GRAPH_OUTPUT_DIRECTORY}/embeddings.entity.description.csv')

    df = pd.read_parquet(f'{GRAPH_OUTPUT_DIRECTORY}/output/embeddings.community.full_content.parquet')
    df.to_csv(f'{GRAPH_OUTPUT_DIRECTORY}/embeddings.community.full_content.csv')

    df = pd.read_parquet(f'{GRAPH_OUTPUT_DIRECTORY}/output/embeddings.text_unit.text.parquet')
    df.to_csv(f'{GRAPH_OUTPUT_DIRECTORY}/embeddings.text_unit.text.csv')