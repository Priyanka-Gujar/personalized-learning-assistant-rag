"""
RAG System Evaluation Script
Demonstrates RAG performance improvements with metrics
"""

import time
from helpers.rag_helper import RAGSystem, calculate_retrieval_metrics
from helpers.ollama_helper_with_rag import RAGEnhancedOllamaHelper, generate_response, generate_summary
import json


class RAGEvaluator:
    """
    Comprehensive evaluation of RAG system performance.
    """
    
    def __init__(self):
        self.rag_system = RAGSystem()
        self.rag_helper = RAGEnhancedOllamaHelper(self.rag_system)
        
    def evaluate_retrieval_performance(self, document: str, test_queries: list) -> dict:
        """
        Evaluate retrieval performance with various queries.
        
        Args:
            document: Document to index
            test_queries: List of test queries
            
        Returns:
            Dictionary with performance metrics
        """
        # Index document
        index_start = time.time()
        index_result = self.rag_system.index_document(document)
        index_time = time.time() - index_start
        
        results = {
            'indexing': {
                'time_seconds': index_time,
                'chunks_created': index_result['chunks_created'],
                'total_words': index_result['total_words']
            },
            'queries': []
        }
        
        # Test each query
        for query in test_queries:
            query_start = time.time()
            chunks = self.rag_system.retrieve_context(query, top_k=5)
            query_time = time.time() - query_start
            
            metrics = calculate_retrieval_metrics(chunks)
            
            results['queries'].append({
                'query': query,
                'retrieval_time_ms': query_time * 1000,
                'chunks_retrieved': len(chunks),
                'metrics': metrics
            })
        
        return results
    
    def compare_with_without_rag(self, 
                                 document: str, 
                                 prompt: str,
                                 model: str = "llama3.2") -> dict:
        """
        Compare generation quality with and without RAG.
        
        Args:
            document: Document text
            prompt: Generation prompt
            model: Ollama model to use
            
        Returns:
            Comparison results
        """
        # Without RAG
        start = time.time()
        response_no_rag = generate_response(prompt, document, model, use_rag=False)
        time_no_rag = time.time() - start
        
        # With RAG
        start = time.time()
        response_with_rag = self.rag_helper.generate_response_with_rag(
            prompt, document, model, top_k=3
        )
        time_with_rag = time.time() - start
        
        # Get retrieval context for analysis
        rag_context = self.rag_system.get_rag_context(prompt, top_k=3)
        
        return {
            'without_rag': {
                'response': response_no_rag,
                'time_seconds': time_no_rag,
                'response_length': len(response_no_rag.split())
            },
            'with_rag': {
                'response': response_with_rag,
                'time_seconds': time_with_rag,
                'response_length': len(response_with_rag.split()),
                'context_used': len(rag_context.split())
            }
        }
    
    def evaluate_summary_quality(self, document: str) -> dict:
        """
        Compare summary quality with and without RAG.
        """
        summary_prompt = "Provide a comprehensive summary of the key concepts and main ideas."
        
        # Without RAG
        start = time.time()
        summary_no_rag = generate_summary(summary_prompt, document, use_rag=False)
        time_no_rag = time.time() - start
        
        # With RAG
        start = time.time()
        summary_with_rag = self.rag_helper.generate_summary_with_rag(
            summary_prompt, document, top_k=5
        )
        time_with_rag = time.time() - start
        
        return {
            'without_rag': {
                'summary': summary_no_rag,
                'time_seconds': time_no_rag,
                'length_words': len(summary_no_rag.split())
            },
            'with_rag': {
                'summary': summary_with_rag,
                'time_seconds': time_with_rag,
                'length_words': len(summary_with_rag.split())
            }
        }


def run_comprehensive_evaluation():
    """
    Run a complete evaluation suite for the RAG system.
    """
    evaluator = RAGEvaluator()
    
    # Sample document (you can replace with your own)
    sample_document = """
    Machine learning is a subset of artificial intelligence that focuses on developing systems 
    that can learn from and make decisions based on data. There are three main types of machine 
    learning: supervised learning, unsupervised learning, and reinforcement learning.
    
    Supervised learning involves training a model on labeled data, where the correct output is 
    known. Common algorithms include linear regression, logistic regression, decision trees, 
    and neural networks. This approach is widely used for classification and regression tasks.
    
    Unsupervised learning deals with unlabeled data and tries to find hidden patterns or 
    structures. Clustering algorithms like K-means and hierarchical clustering are examples. 
    Dimensionality reduction techniques like PCA also fall under this category.
    
    Reinforcement learning is about training agents to make sequences of decisions by rewarding 
    desired behaviors and punishing undesired ones. It's commonly used in robotics, game playing, 
    and autonomous systems. Q-learning and policy gradient methods are popular approaches.
    
    Deep learning, a subset of machine learning, uses artificial neural networks with multiple 
    layers. Convolutional neural networks (CNNs) excel at image processing, while recurrent 
    neural networks (RNNs) and transformers are powerful for sequence data and natural language 
    processing.
    
    Model evaluation is crucial in machine learning. Common metrics include accuracy, precision, 
    recall, F1-score for classification, and mean squared error (MSE) for regression. 
    Cross-validation helps ensure models generalize well to unseen data.
    
    Feature engineering, the process of creating new features from raw data, significantly 
    impacts model performance. Techniques include normalization, encoding categorical variables, 
    and creating interaction features.
    
    Overfitting occurs when a model learns the training data too well, including noise, 
    resulting in poor performance on new data. Regularization techniques like L1 and L2 
    regularization help prevent overfitting by penalizing complex models.
    """
    
    print("=" * 80)
    print("RAG SYSTEM COMPREHENSIVE EVALUATION")
    print("=" * 80)
    
    # 1. Retrieval Performance Evaluation
    print("\n1. RETRIEVAL PERFORMANCE EVALUATION")
    print("-" * 80)
    
    test_queries = [
        "What are the types of machine learning?",
        "Explain supervised learning algorithms",
        "How does deep learning differ from traditional ML?",
        "What is overfitting and how to prevent it?",
        "Describe model evaluation metrics"
    ]
    
    retrieval_results = evaluator.evaluate_retrieval_performance(sample_document, test_queries)
    
    print(f"\nIndexing Performance:")
    print(f"  - Time: {retrieval_results['indexing']['time_seconds']:.3f}s")
    print(f"  - Chunks Created: {retrieval_results['indexing']['chunks_created']}")
    print(f"  - Total Words: {retrieval_results['indexing']['total_words']}")
    
    print(f"\nQuery Performance:")
    for query_result in retrieval_results['queries']:
        print(f"\n  Query: '{query_result['query']}'")
        print(f"    - Retrieval Time: {query_result['retrieval_time_ms']:.2f}ms")
        print(f"    - Chunks Retrieved: {query_result['chunks_retrieved']}")
        print(f"    - Avg Similarity: {query_result['metrics']['avg_similarity']:.3f}")
    
    # 2. Question Generation Comparison
    print("\n\n2. QUESTION GENERATION COMPARISON (With vs Without RAG)")
    print("-" * 80)
    
    question_prompt = "Generate 3 questions about machine learning types."
    comparison = evaluator.compare_with_without_rag(sample_document, question_prompt)
    
    print("\nWithout RAG:")
    print(f"  - Time: {comparison['without_rag']['time_seconds']:.2f}s")
    print(f"  - Response Length: {comparison['without_rag']['response_length']} words")
    print(f"  - Response: {comparison['without_rag']['response'][:200]}...")
    
    print("\nWith RAG:")
    print(f"  - Time: {comparison['with_rag']['time_seconds']:.2f}s")
    print(f"  - Response Length: {comparison['with_rag']['response_length']} words")
    print(f"  - Context Used: {comparison['with_rag']['context_used']} words")
    print(f"  - Response: {comparison['with_rag']['response'][:200]}...")
    
    # 3. Summary Quality Comparison
    print("\n\n3. SUMMARY QUALITY COMPARISON (With vs Without RAG)")
    print("-" * 80)
    
    summary_results = evaluator.evaluate_summary_quality(sample_document)
    
    print("\nWithout RAG:")
    print(f"  - Time: {summary_results['without_rag']['time_seconds']:.2f}s")
    print(f"  - Length: {summary_results['without_rag']['length_words']} words")
    print(f"  - Summary: {summary_results['without_rag']['summary'][:200]}...")
    
    print("\nWith RAG:")
    print(f"  - Time: {summary_results['with_rag']['time_seconds']:.2f}s")
    print(f"  - Length: {summary_results['with_rag']['length_words']} words")
    print(f"  - Summary: {summary_results['with_rag']['summary'][:200]}...")
    
    # 4. Save results to JSON
    print("\n\n4. SAVING RESULTS")
    print("-" * 80)
    
    all_results = {
        'retrieval_performance': retrieval_results,
        'question_generation': comparison,
        'summary_quality': summary_results
    }
    
    with open('rag_evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("Results saved to: rag_evaluation_results.json")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_evaluation()