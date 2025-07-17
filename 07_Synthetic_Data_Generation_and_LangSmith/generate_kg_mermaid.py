import json
from collections import Counter, deque

# Parameters
KG_FILE = 'loan_data_kg.json'
OUTPUT_FILE = 'loan_kg_mermaid.md'
MAX_NODES = 10  # Number of nodes to visualize

# Load the knowledge graph
with open(KG_FILE, 'r') as f:
    kg = json.load(f)

nodes = kg.get('nodes', [])
rels = kg.get('relationships', [])

# Improved get_id function

def get_id(x):
    if isinstance(x, dict) and 'id' in x:
        return x['id']
    return str(x)

# Build a mapping from id to node for fast lookup
id_to_node = {get_id(node.get('id')): node for node in nodes}

# Build adjacency list for the graph
adj = {}
for rel in rels:
    src = get_id(rel.get('source'))
    tgt = get_id(rel.get('target'))
    adj.setdefault(src, set()).add(tgt)
    adj.setdefault(tgt, set()).add(src)

# Count degree for each node
node_degrees = {node_id: len(neighbors) for node_id, neighbors in adj.items()}

# Select the top MAX_NODES nodes by degree
top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:MAX_NODES]
selected_ids = set(node_id for node_id, _ in top_nodes)
selected_nodes = [id_to_node[nid] for nid in selected_ids if nid in id_to_node]

# Filter relationships to only those between selected nodes
selected_rels = [rel for rel in rels if get_id(rel.get('source')) in selected_ids and get_id(rel.get('target')) in selected_ids]

# Print debug info
print(f"Total nodes in file: {len(nodes)}")
print(f"Node degrees:")
for node_id, degree in top_nodes:
    content = id_to_node[node_id].get('properties', {}).get('page_content', '')[:40].replace('\n', ' ') if node_id in id_to_node else ''
    print(f"Node {node_id}: degree {degree}, content: {content}")
print(f"Relationships among selected nodes: {len(selected_rels)}")

# Build Mermaid diagram
mermaid = ["```mermaid", "graph TD"]
for node in selected_nodes:
    node_id = get_id(node.get('id'))
    label = node.get('properties', {}).get('page_content', '')[:40].replace('\n', ' ').replace('"', "'")
    mermaid.append(f'{node_id}(["{label}"])')
for rel in selected_rels:
    src = get_id(rel.get('source'))
    tgt = get_id(rel.get('target'))
    rel_type = rel.get('type', '')
    mermaid.append(f'{src} --|{rel_type}|--> {tgt}')
mermaid.append("```\n")

with open(OUTPUT_FILE, 'w') as f:
    f.write('\n'.join(mermaid))
