import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import numpy as np

def load_squad_dataset(filepath: str) -> List[Dict]:
    """
    Load SQuAD dataset from JSON file.
    
    Args:
        filepath (str): Path to the SQuAD JSON file
    
    Returns:
        List[Dict]: Loaded dataset
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data['data']
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in the dataset file")

def extract_squad_info(dataset: List[Dict]) -> pd.DataFrame:
    """
    Extract relevant information from SQuAD dataset into a DataFrame.
    
    Args:
        dataset (List[Dict]): Raw SQuAD dataset
    
    Returns:
        pd.DataFrame: Processed dataset with questions, answers, and metadata
    """
    records = []
    
    for article in dataset:
        title = article['title']
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                answer_texts = [ans['text'] for ans in qa.get('answers', [])]
                question = qa['question']
                
                # Extract question type (first word of the question)
                question_type = question.strip().split()[0].lower()
                
                records.append({
                    'title': title,
                    'question': question,
                    'question_type': question_type,
                    'answers': answer_texts if answer_texts else ['Unanswerable'],
                    'context_length': len(context.split()),
                    'question_length': len(question.split()),
                    'answer_count': len(answer_texts),
                    'context': context  # Adding context for potential future analysis
                })
    
    df = pd.DataFrame(records)
    df['combined_length'] = df['context_length'] + df['question_length']
    df['answer_length'] = df['answers'].apply(lambda x: len(x[0].split()))
    return df

def analyze_question_types(df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """
    Analyze question types and provide examples.
    
    Args:
        df (pd.DataFrame): Processed SQuAD dataset
    
    Returns:
        Tuple[Dict, pd.DataFrame]: Question type counts and examples
    """
    # Get question type counts
    question_types = dict(df['question_type'].value_counts())
    
    # Get examples for top 10 question types
    examples = defaultdict(list)
    for qt in list(question_types.keys())[:10]:
        examples[qt] = df[df['question_type'] == qt]['question'].sample(
            min(3, len(df[df['question_type'] == qt])), 
            random_state=42
        ).tolist()
    
    return question_types, pd.DataFrame.from_dict(examples, orient='index')

def plot_length_distributions(df: pd.DataFrame) -> None:
    """
    Plot various length distributions.
    
    Args:
        df (pd.DataFrame): Processed SQuAD dataset
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Question length distribution
    sns.histplot(data=df, x='question_length', bins=30, kde=True, ax=axes[0,0])
    axes[0,0].set_title('Question Length Distribution')
    axes[0,0].set_xlabel('Number of Words')
    axes[0,0].axvline(df['question_length'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["question_length"].mean():.1f}')
    axes[0,0].legend()

    # Context length distribution
    sns.histplot(data=df, x='context_length', bins=30, kde=True, ax=axes[0,1])
    axes[0,1].set_title('Context Length Distribution')
    axes[0,1].set_xlabel('Number of Words')
    axes[0,1].axvline(df['context_length'].mean(), color='red', linestyle='--',
                      label=f'Mean: {df["context_length"].mean():.1f}')
    axes[0,1].legend()

    # Combined length distribution
    sns.histplot(data=df, x='combined_length', bins=30, kde=True, ax=axes[1,0])
    axes[1,0].set_title('Combined (Context + Question) Length Distribution')
    axes[1,0].set_xlabel('Number of Words')
    axes[1,0].axvline(df['combined_length'].mean(), color='red', linestyle='--',
                      label=f'Mean: {df["combined_length"].mean():.1f}')
    axes[1,0].legend()

    # Answer length distribution
    sns.histplot(data=df, x='answer_length', bins=30, kde=True, ax=axes[1,1])
    axes[1,1].set_title('Answer Length Distribution')
    axes[1,1].set_xlabel('Number of Words')
    axes[1,1].axvline(df['answer_length'].mean(), color='red', linestyle='--',
                      label=f'Mean: {df["answer_length"].mean():.1f}')
    axes[1,1].legend()

    plt.tight_layout()
    plt.show()

def analyze_squad_dataset(df: pd.DataFrame) -> None:
    """
    Perform comprehensive analysis of the SQuAD dataset.
    
    Args:
        df (pd.DataFrame): Processed SQuAD dataset
    """
    plt.style.use('seaborn')
    
    # 1. Dataset Overview
    print("=== SQuAD Dataset Analysis ===")
    print(f"\nTotal number of questions: {len(df):,}")
    print(f"Number of unique articles: {df['title'].nunique():,}")
    print("\nDataset Summary:")
    print(df.describe())
    
    # 2. Question Types Analysis
    question_types, examples_df = analyze_question_types(df)
    print("\n=== Question Types Analysis ===")
    print(f"Total number of unique question types: {len(question_types)}")
    print("\nTop 10 question types with counts:")
    for qtype, count in list(question_types.items())[:10]:
        print(f"{qtype}: {count}")
        
    print("\nExample questions for each type:")
    print(examples_df.head(10))
    
    # 3. Plot question type distribution
    plt.figure(figsize=(12, 6))
    top_types = pd.Series(question_types).head(10)
    sns.barplot(x=top_types.values, y=top_types.index)
    plt.title('Top 10 Question Types')
    plt.xlabel('Count')
    plt.ylabel('Question Word')
    plt.show()
    
    # 4. Articles Analysis
    plt.figure(figsize=(12, 6))
    top_articles = df['title'].value_counts().head(10)
    sns.barplot(x=top_articles.values, y=top_articles.index)
    plt.title('Top 10 Articles by Number of Questions')
    plt.xlabel('Number of Questions')
    plt.ylabel('Article Title')
    plt.tight_layout()
    plt.show()
    
    # 5. Length Statistics and Distributions
    print("\n=== Length Statistics ===")
    length_stats = pd.DataFrame({
        'Question Length': df['question_length'].describe(),
        'Context Length': df['context_length'].describe(),
        'Combined Length': df['combined_length'].describe(),
        'Answer Length': df['answer_length'].describe()
    })
    print(length_stats)
    
    # 6. Plot all length distributions
    plot_length_distributions(df)

def main():
    """
    Main function to run the SQuAD dataset analysis.
    """
    try:
        # Load and process dataset
        filepath = '/kaggle/input/stanford-question-answering-dataset/train-v1.1.json'  # Update with actual path
        print("Loading SQuAD dataset...")
        data = load_squad_dataset(filepath)
        
        print("Processing dataset...")
        squad_df = extract_squad_info(data)
        
        print("Analyzing dataset...")
        analyze_squad_dataset(squad_df)
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please ensure the SQuAD dataset file exists and update the filepath.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
