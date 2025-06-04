from neo4j import GraphDatabase
import csv

URI = "neo4j+s://c0651eec.databases.neo4j.io"
USERNAME = "neo4j"
PASSWORD = "eWRgS3xons7xBhxaoZM0fr1SJZeANZoS6d_334ykH1k"  

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

def load_celltype_markers(csv_path):
    with driver.session() as session:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                celltype = row["celltype"]
                gene = row["markergene"]
                session.run("""
                    MERGE (ct:CellType {name: $celltype})
                    MERGE (mg:MarkerGene {name: $gene})
                    MERGE (ct)-[:MARKERED_BY]->(mg)
                """, celltype=celltype, gene=gene)
    print("Data import completed.")


load_celltype_markers("/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/data/cell_marker_all.csv")
driver.close()


