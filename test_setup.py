"""
Quick verification script for RAG setup
"""

print("=" * 60)
print("TESTING RAG SETUP")
print("=" * 60)

# Test 1: Import core libraries
print("\n1. Testing core imports...")
try:
    import chromadb
    print("   ✅ chromadb imported")
except ImportError as e:
    print(f"   ❌ chromadb import failed: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("   ✅ sentence_transformers imported")
except ImportError as e:
    print(f"   ❌ sentence_transformers import failed: {e}")

try:
    import streamlit
    print("   ✅ streamlit imported")
except ImportError as e:
    print(f"   ❌ streamlit import failed: {e}")

try:
    import ollama
    print("   ✅ ollama imported")
except ImportError as e:
    print(f"   ❌ ollama import failed: {e}")

# Test 2: Import your RAG helper
print("\n2. Testing RAG helper imports...")
try:
    from helpers.rag_helper import RAGSystem
    print("   ✅ RAGSystem imported successfully")
except ImportError as e:
    print(f"   ❌ RAGSystem import failed: {e}")
    print(f"      Make sure helpers/rag_helper.py exists")

try:
    from helpers.ollama_helper_with_rag import RAGEnhancedOllamaHelper
    print("   ✅ RAGEnhancedOllamaHelper imported successfully")
except ImportError as e:
    print(f"   ❌ RAGEnhancedOllamaHelper import failed: {e}")

# Test 3: Initialize RAG system
print("\n3. Testing RAG system initialization...")
try:
    from helpers.rag_helper import RAGSystem
    rag = RAGSystem()
    print("   ✅ RAG system initialized")
    
    # Test document indexing
    test_doc = """
    Machine learning is a subset of artificial intelligence. 
    It involves training models on data to make predictions.
    Deep learning uses neural networks with multiple layers.
    """
    
    print("\n4. Testing document indexing...")
    result = rag.index_document(test_doc)
    print(f"   ✅ Document indexed successfully")
    print(f"      - Chunks created: {result['chunks_created']}")
    print(f"      - Total words: {result['total_words']}")
    
    # Test retrieval
    print("\n5. Testing semantic search...")
    chunks = rag.retrieve_context("What is machine learning?", top_k=2)
    print(f"   ✅ Retrieved {len(chunks)} chunks")
    if chunks:
        print(f"      - Top similarity score: {chunks[0]['similarity_score']:.3f}")
    
    # Test statistics
    print("\n6. Testing RAG statistics...")
    stats = rag.get_statistics()
    print(f"   ✅ Statistics retrieved")
    print(f"      - Total chunks: {stats['total_chunks']}")
    
except Exception as e:
    print(f"   ❌ RAG system test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
print("\nIf all tests passed ✅, you're ready to commit!")
print("If any failed ❌, check the error messages above.\n")