from typing import List, Dict, Tuple, Any
def build_prompt(query: str, context_chunks: List[Tuple[str, float]]) -> str:
    context_texts = "\n\n".join([c[0] for c in context_chunks])
    prompt = f"""You are a helpful assistant. 
Here is some context from the knowledge base:

{context_texts}

Based on the above, answer this question:

{query}
"""
    return prompt