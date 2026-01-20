"""
RAG (Retrieval-Augmented Generation) Helper Module
Provides semantic search and context retrieval for the Personalized Learning Assistant
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime
import hashlib
import re


class RAGSystem:
    """
    Retrieval-Augmented Generation System for enhanced question generation and summarization.
    Uses ChromaDB for vector storage and sentence-transformers for embeddings.
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 collection_name: str = "learning_documents",
                 persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model: HuggingFace model for embeddings
            collection_name: Name for the ChromaDB collection
            persist_directory: Directory to persist the vector database
        """
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        self.chunk_size = 500  # words per chunk
        self.chunk_overlap = 50  # words overlap between chunks
        
    def _create_document_id(self, text: str) -> str:
        """Create a unique document ID based on content hash."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _intelligent_chunk_text(self, text: str) -> List[Dict[str, str]]:
        """
        Split text into semantically meaningful chunks.
        Uses sentence boundaries and paragraph structure.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_word_count + sentence_word_count > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'word_count': current_word_count,
                    'sentence_count': len(current_chunk)
                })
                
                # Keep last few sentences for overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_word_count = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'word_count': current_word_count,
                'sentence_count': len(current_chunk)
            })
        
        return chunks
    
    def index_document(self, document_text: str, metadata: Dict = None) -> Dict:
        """
        Index a document into the vector store.
        
        Args:
            document_text: Full text of the document
            metadata: Optional metadata (title, source, etc.)
            
        Returns:
            Dictionary with indexing statistics
        """
        if not document_text or len(document_text.strip()) < 10:
            return {"error": "Document too short to index"}
        
        # Create document ID
        doc_id = self._create_document_id(document_text)
        
        # Chunk the document
        chunks = self._intelligent_chunk_text(document_text)
        
        if not chunks:
            return {"error": "Failed to chunk document"}
        
        # Prepare data for ChromaDB
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunk_texts).tolist()
        
        # Prepare metadata
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            meta = {
                'doc_id': doc_id,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'word_count': chunk['word_count'],
                'indexed_at': datetime.now().isoformat()
            }
            if metadata:
                meta.update(metadata)
            chunk_metadata.append(meta)
        
        # Add to ChromaDB
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=chunk_metadata
        )
        
        return {
            'doc_id': doc_id,
            'chunks_created': len(chunks),
            'total_words': sum(c['word_count'] for c in chunks),
            'indexed_at': datetime.now().isoformat()
        }
    
    def retrieve_context(self, 
                        query: str, 
                        top_k: int = 5,
                        relevance_threshold: float = 0.3) -> List[Dict]:
        """
        Retrieve relevant context chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            relevance_threshold: Minimum cosine similarity score
            
        Returns:
            List of relevant chunks with scores and metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Process results
        retrieved_chunks = []
        
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                # Calculate similarity score (ChromaDB returns distances, convert to similarity)
                distance = results['distances'][0][i] if 'distances' in results else 1.0
                similarity = 1 - distance  # Convert distance to similarity
                
                if similarity >= relevance_threshold:
                    retrieved_chunks.append({
                        'chunk_id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'similarity_score': similarity,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                    })
        
        # Sort by similarity score
        retrieved_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return retrieved_chunks
    
    def get_rag_context(self, 
                       query: str, 
                       top_k: int = 3,
                       max_context_length: int = 2000) -> str:
        """
        Get formatted context string for RAG-enhanced generation.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            max_context_length: Maximum words in context
            
        Returns:
            Formatted context string
        """
        chunks = self.retrieve_context(query, top_k=top_k)
        
        if not chunks:
            return ""
        
        # Build context string
        context_parts = []
        total_words = 0
        
        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk['text']
            chunk_words = len(chunk_text.split())
            
            if total_words + chunk_words <= max_context_length:
                context_parts.append(f"[Context {i} - Relevance: {chunk['similarity_score']:.2f}]\n{chunk_text}")
                total_words += chunk_words
            else:
                break
        
        return "\n\n".join(context_parts)
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def get_statistics(self) -> Dict:
        """Get statistics about the indexed documents."""
        count = self.collection.count()
        return {
            'total_chunks': count,
            'embedding_model': self.embedding_model.get_sentence_embedding_dimension(),
            'collection_name': self.collection.name
        }


def create_rag_enhanced_prompt(base_prompt: str, 
                               rag_context: str, 
                               original_document: str) -> str:
    """
    Create an enhanced prompt that combines RAG context with the original document.
    
    Args:
        base_prompt: Original prompt (e.g., question generation prompt)
        rag_context: Retrieved relevant context from RAG
        original_document: Full original document
        
    Returns:
        Enhanced prompt string
    """
    if rag_context:
        enhanced_prompt = f"""You are provided with relevant context retrieved from the document along with the full document text.

RETRIEVED RELEVANT CONTEXT:
{rag_context}

FULL DOCUMENT:
{original_document}

{base_prompt}

Note: Use the retrieved context to focus on the most relevant parts, but you can reference the full document if needed."""
    else:
        enhanced_prompt = f"Document: {original_document}\n\n{base_prompt}"
    
    return enhanced_prompt


# Evaluation functions for RAG performance

def calculate_retrieval_metrics(retrieved_chunks: List[Dict], 
                               ground_truth_relevant: List[str] = None) -> Dict:
    """
    Calculate retrieval performance metrics.
    
    Args:
        retrieved_chunks: List of retrieved chunks from RAG
        ground_truth_relevant: List of ground truth relevant chunk IDs (if available)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'num_retrieved': len(retrieved_chunks),
        'avg_similarity': np.mean([c['similarity_score'] for c in retrieved_chunks]) if retrieved_chunks else 0,
        'min_similarity': min([c['similarity_score'] for c in retrieved_chunks]) if retrieved_chunks else 0,
        'max_similarity': max([c['similarity_score'] for c in retrieved_chunks]) if retrieved_chunks else 0
    }
    
    # Calculate precision and recall if ground truth is provided
    if ground_truth_relevant and retrieved_chunks:
        retrieved_ids = set([c['chunk_id'] for c in retrieved_chunks])
        relevant_ids = set(ground_truth_relevant)
        
        true_positives = len(retrieved_ids & relevant_ids)
        metrics['precision'] = true_positives / len(retrieved_ids) if retrieved_ids else 0
        metrics['recall'] = true_positives / len(relevant_ids) if relevant_ids else 0
        metrics['f1_score'] = (2 * metrics['precision'] * metrics['recall'] / 
                              (metrics['precision'] + metrics['recall'])) if (metrics['precision'] + metrics['recall']) > 0 else 0
    
    return metrics