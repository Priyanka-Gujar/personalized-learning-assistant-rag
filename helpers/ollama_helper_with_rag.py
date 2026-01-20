"""
Enhanced Ollama Helper with RAG Integration
Combines retrieval-augmented generation with Ollama LLMs
"""

import ollama
from helpers.rag_helper import RAGSystem, create_rag_enhanced_prompt
from typing import Optional


class RAGEnhancedOllamaHelper:
    """
    Wrapper for Ollama that incorporates RAG for improved context-aware generation.
    """
    
    def __init__(self, rag_system: Optional[RAGSystem] = None):
        """
        Initialize the RAG-enhanced Ollama helper.
        
        Args:
            rag_system: Optional pre-initialized RAG system. If None, creates a new one.
        """
        self.rag_system = rag_system if rag_system else RAGSystem()
        
    def generate_response_with_rag(self, 
                                   prompt: str, 
                                   document: str,
                                   model: str = "llama3.2",
                                   use_rag: bool = True,
                                   top_k: int = 3) -> str:
        """
        Generate a response using RAG to retrieve relevant context.
        
        Args:
            prompt: The input prompt/question
            document: The full document text
            model: Ollama model to use
            use_rag: Whether to use RAG enhancement
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Generated response from the model
        """
        if use_rag:
            # Index the document if not already indexed
            index_result = self.rag_system.index_document(document)
            
            # Retrieve relevant context based on the prompt
            rag_context = self.rag_system.get_rag_context(
                query=prompt, 
                top_k=top_k
            )
            
            # Create enhanced prompt with RAG context
            full_prompt = create_rag_enhanced_prompt(prompt, rag_context, document)
        else:
            # Fallback to original behavior
            full_prompt = f"Document: {document}\n\nPrompt: {prompt}"
        
        # Call Ollama
        response = ollama.chat(
            model=model, 
            messages=[{"role": "user", "content": full_prompt}]
        )
        
        return response['message']['content']
    
    def generate_summary_with_rag(self,
                                  prompt: str,
                                  document: str,
                                  model: str = "gemma2:2b",
                                  use_rag: bool = True,
                                  top_k: int = 5) -> str:
        """
        Generate a summary using RAG to focus on key content.
        
        Args:
            prompt: The summarization prompt
            document: The full document text
            model: Ollama model to use
            use_rag: Whether to use RAG enhancement
            top_k: Number of relevant chunks for summary
            
        Returns:
            Generated summary
        """
        if use_rag:
            # Index the document
            self.rag_system.index_document(document)
            
            # For summarization, retrieve key sections
            # Use a generic query to get representative chunks
            summary_query = "main points key concepts important information"
            rag_context = self.rag_system.get_rag_context(
                query=summary_query,
                top_k=top_k,
                max_context_length=3000
            )
            
            full_prompt = create_rag_enhanced_prompt(prompt, rag_context, document)
        else:
            full_prompt = f"Document: {document}\n\nPrompt: {prompt}"
        
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": full_prompt}]
        )
        
        return response['message']['content']


# Backward-compatible wrapper functions
def generate_response(prompt: str, 
                     document: str, 
                     model: str = "llama3.2",
                     use_rag: bool = False) -> str:
    """
    Backward-compatible function for generating responses.
    Set use_rag=True to enable RAG enhancement.
    """
    if use_rag:
        helper = RAGEnhancedOllamaHelper()
        return helper.generate_response_with_rag(prompt, document, model)
    else:
        # Original implementation
        full_prompt = f"Document: {document}\n\nPrompt: {prompt}"
        response = ollama.chat(model=model, messages=[{"role": "user", "content": full_prompt}])
        return response['message']['content']


def generate_summary(prompt: str, 
                    document: str, 
                    model: str = "gemma2:2b",
                    use_rag: bool = False) -> str:
    """
    Backward-compatible function for generating summaries.
    Set use_rag=True to enable RAG enhancement.
    """
    if use_rag:
        helper = RAGEnhancedOllamaHelper()
        return helper.generate_summary_with_rag(prompt, document, model)
    else:
        # Original implementation
        full_prompt = f"Document: {document}\n\nPrompt: {prompt}"
        response = ollama.chat(model=model, messages=[{"role": "user", "content": full_prompt}])
        return response['message']['content']