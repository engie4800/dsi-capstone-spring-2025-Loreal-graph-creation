import xml.etree.ElementTree as ET

def parse_graphml(file_path):

    with open(input_file, "r", encoding="utf-8") as f:
        graphml_content = f.read()

    graphml_content = graphml_content.replace("'", "")

    with open(input_file, "w", encoding="utf-8") as f:
        f.write(graphml_content)

    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Define the namespace
    ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
    
    # Extract nodes
    nodes = set()
    for node in root.findall(".//graphml:node", ns):
        node_id = node.get("id")
        nodes.add(node_id)
    
    # Extract edges
    edges = []
    for edge in root.findall(".//graphml:edge", ns):
        source = edge.get("source")
        target = edge.get("target")
        edges.append((source, target))
    
    return nodes, edges

def generate_cypher_queries(nodes, edges, output_file):
    with open(output_file, "w") as f:
        # Create nodes
        for node in nodes:
            f.write(f"CREATE (:Item {{name: '{node}'}});\n")
        
        # Create relationships
        for source, target in edges:
            f.write(f"""MATCH (a:Item {{name: "{source}"}}), (b:Item {{name: "{target}"}}) CREATE (a)-[:RELATED_TO]->(b);\n"""
        )

if __name__ == "__main__":
    input_file = "Code/beautyragtest2/output/graph.graphml"
    output_file = "Code/beautyragtest2/cypher_queries.txt"
    nodes, edges = parse_graphml(input_file)
    generate_cypher_queries(nodes, edges, output_file)
    print(f"Cypher queries saved to {output_file}")
