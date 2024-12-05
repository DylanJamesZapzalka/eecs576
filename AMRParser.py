import re

def parse_amr_to_json(amr_text):
    # Extract nodes and edges from AMR
    nodes = []
    edges = []
    node_map = {}  # Map variable names to node indices
    
    # Helper function to clean node names
    def clean_node_name(name):
        return name.strip('/"')
    
    # Extract the sentence text from the AMR header
    snt_match = re.search(r'# ::snt (.*?)\n', amr_text)
    text = snt_match.group(1) if snt_match else ""
    
    # Parse nodes and edges
    lines = amr_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Parse variable definitions and relationships
        matches = re.findall(r'\(([\w\d]+)\s*/\s*([\w\-]+)|\:([\w\-]+)\s*([\w\d]+|\".+?\"|\(.+?\))', line)
        
        for match in matches:
            var, node_name, edge_type, target = match
            
            if var and node_name:  # This is a node definition
                if node_name not in nodes:
                    nodes.append(clean_node_name(node_name))
                    node_map[var] = len(nodes) - 1
                    
            if edge_type and target:  # This is an edge
                if edge_type not in ['op1', 'op2', 'op3', 'op4']:  # Skip operational edges
                    source_idx = node_map.get(var, -1)
                    
                    # Handle quoted targets
                    if target.startswith('"'):
                        if target not in nodes:
                            nodes.append(clean_node_name(target))
                        target_idx = nodes.index(clean_node_name(target))
                    else:
                        target_idx = node_map.get(target, -1)
                    
                    if source_idx != -1 and target_idx != -1:
                        edges.append([source_idx, edge_type, target_idx])
    
    return {
        "text": text,
        "nodes": nodes,
        "edges": edges
    }

if __name__ == "__main__":
    # Example usage
    sample_amr = '''# ::snt The field of automated reasoning is an outgrowth of the field of automated theorem proving.
    (g / grow-01
        :ARG1 (f / field
                :topic (r / reason-01
                    :manner (a / automate-01)))
        :ARG3 (f2 / field
                :topic (p / prove-01
                    :ARG1 (t / theorem
                            :ARG1-of (a2 / automate-01)))))'''

    result = parse_amr_to_json(sample_amr)