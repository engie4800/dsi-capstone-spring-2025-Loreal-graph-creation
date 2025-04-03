import os
from neo4j import GraphDatabase
from dotenv import load_dotenv


load_dotenv('Code/.env')

neo4j_url = os.getenv("NEO4J_CONNECTION_URL")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

gds = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))

def execute_cypher_from_file(filepath):
    """
    Reads Cypher queries from a text file and executes them sequentially.
    
    Args:
        filepath (str): Path to the text file containing Cypher queries (one per line)
    """
    try:
        # Read the cypher statements from file
        with open(filepath, 'r') as file:
            cypher_statements = [line.strip() for line in file if line.strip()]
        
        print(f"Found {len(cypher_statements)} Cypher statements to execute")
        
        # Execute each statement
        failed_statements = []
        for i, stmt in enumerate(cypher_statements, 1):
            try:
                print(f"Executing statement {i} of {len(cypher_statements)}")
                gds.execute_query(stmt)
            except Exception as e:
                print(f"Error executing statement {i}: {e}")
                failed_statements.append((stmt, str(e)))
        
        # Report results
        success_count = len(cypher_statements) - len(failed_statements)
        print(f"\nExecution complete:")
        print(f"Successfully executed: {success_count}/{len(cypher_statements)} statements")
        
        # If there were any failures, write them to a file
        if failed_statements:
            print(f"Failed statements: {len(failed_statements)}")
            with open("failed_statements.txt", "w") as f:
                for stmt, error in failed_statements:
                    f.write(f"Statement: {stmt}\nError: {error}\n\n")
            print("Failed statements have been written to 'failed_statements.txt'")
            
    except FileNotFoundError:
        print(f"Error: Could not find file '{filepath}'")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Ensure the database connection is closed properly
        gds.close()


if __name__ == "__main__":
    execute_cypher_from_file("Code/beautyragtest/cypher_queries.txt")