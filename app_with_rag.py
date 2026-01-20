import streamlit as st
from helpers.ollama_helper_with_rag import generate_response, generate_summary, RAGEnhancedOllamaHelper
from helpers.rag_helper import RAGSystem
from helpers.pdf_reader import extract_text
from helpers.exa_search import perform_web_search, extract_text_from_exa
from helpers.elevenlabs_helper import text_to_speech_file


# Initialize RAG system in session state
if 'rag_system' not in st.session_state:
    st.session_state['rag_system'] = RAGSystem()

if 'rag_helper' not in st.session_state:
    st.session_state['rag_helper'] = RAGEnhancedOllamaHelper(st.session_state['rag_system'])


# Streamlit UI setup
st.set_page_config(page_title="Personalized Learning Assistant with RAG", layout="centered")

st.title("🎓 Personalized Learning Assistant with RAG")
st.write("Upload documents or enter text to generate AI-powered summaries and questions using **Retrieval-Augmented Generation**!")

# Sidebar for RAG settings
with st.sidebar:
    st.header("⚙️ RAG Settings")
    
    use_rag = st.checkbox("Enable RAG Enhancement", value=True, 
                         help="Use semantic search to retrieve relevant context before generation")
    
    if use_rag:
        top_k_questions = st.slider("Top-K chunks for Questions", 1, 10, 3,
                                   help="Number of relevant chunks to retrieve for question generation")
        top_k_summary = st.slider("Top-K chunks for Summary", 1, 10, 5,
                                 help="Number of relevant chunks to retrieve for summarization")
        
        st.markdown("---")
        st.subheader("📊 RAG Statistics")
        
        if st.button("View RAG Stats"):
            stats = st.session_state['rag_system'].get_statistics()
            st.json(stats)
        
        if st.button("Clear Vector Database"):
            st.session_state['rag_system'].clear_collection()
            st.success("Vector database cleared!")
    else:
        top_k_questions = 3
        top_k_summary = 5

# Main content
# File uploader and text box input
uploaded_file = st.file_uploader("Upload a DOCX, PPTX, or PDF file", type=['docx', 'pptx', 'pdf'])

text_disabled = uploaded_file is not None
user_text_input = st.text_area("Or enter text manually:", "", height=200, disabled=text_disabled)

user_web_urls = st.text_input("Enter URLs to extract text from (optional, comma-separated):", "", 
                              help="Paste URLs separated by commas to extract content from web pages")

url_extracted_text = ""

if user_web_urls:
    url_list = [url.strip() for url in user_web_urls.split(',')]    
    url_extracted_text = extract_text_from_exa(url_list)
    
    if url_extracted_text:
        st.success('✅ Successfully extracted text from URLs')

# Check if any input is provided
if uploaded_file or user_text_input.strip() or url_extracted_text:
    if uploaded_file:
        extracted_text = "Extracted text from PDF"
    elif url_extracted_text:
        extracted_text = url_extracted_text
    else:
        extracted_text = user_text_input

# Settings
level = st.selectbox("Select your education level:", ['High School', 'Bachelors', 'Masters', 'PhD'])
severity = st.selectbox("Select the severity of the questions:", ["Easy", "Medium", "Tough"])
num_questions = st.number_input("Number of questions to generate:", min_value=1, max_value=20, value=5)

# Load prompts
prompt_file = f"prompts/{severity.lower()}_questions.txt"
with open(prompt_file, 'r') as file:
    question_prompt = file.read()

refined_prompt_file = f"prompts/refined_questions.txt"
with open(refined_prompt_file, 'r') as file:
    refined_prompt = file.read()

def get_input_text():
    if uploaded_file is not None:
        with st.spinner("Extracting text from document..."):
            return extract_text(uploaded_file)
    elif user_text_input.strip():
        return user_text_input.strip()
    elif url_extracted_text:
        return url_extracted_text
    else:
        return None


# ========== QUESTIONS GENERATION WITH RAG ==========
st.markdown("---")
st.subheader("❓ Generate Questions")

col1, col2 = st.columns([3, 1])

with col1:
    if st.button("🤖 Generate Questions", use_container_width=True):
        extracted_text = get_input_text()
        
        if extracted_text is None:
            st.error("Please upload a document or enter text manually.")
        else:
            with st.spinner("Generating questions with RAG..."):
                formatted_prompt = question_prompt.format(num_questions=num_questions, level=level)
                
                # Use RAG-enhanced generation
                if use_rag:
                    response = st.session_state['rag_helper'].generate_response_with_rag(
                        formatted_prompt, 
                        extracted_text,
                        top_k=top_k_questions
                    )
                else:
                    response = generate_response(formatted_prompt, extracted_text, use_rag=False)
                
                # Refine the response
                refined_response = generate_response(refined_prompt, response, use_rag=False)
                st.session_state['questions'] = refined_response

                st.success("✅ Questions generated successfully!")
                
                # Show retrieved context if RAG is enabled
                if use_rag:
                    with st.expander("🔍 View Retrieved Context"):
                        chunks = st.session_state['rag_system'].retrieve_context(
                            formatted_prompt, 
                            top_k=top_k_questions
                        )
                        for i, chunk in enumerate(chunks, 1):
                            st.markdown(f"**Chunk {i}** (Relevance: {chunk['similarity_score']:.2f})")
                            st.text(chunk['text'][:200] + "...")
                            st.markdown("---")

with col2:
    if use_rag:
        st.info(f"RAG: ON\nTop-K: {top_k_questions}")
    else:
        st.warning("RAG: OFF")

if 'questions' in st.session_state and st.session_state['questions']:
    st.markdown("### Generated Questions:")
    st.write(st.session_state['questions'])


# ========== ANSWER GENERATION WITH RAG ==========
answers_file = "prompts/answers.txt"
with open(answers_file, 'r') as file:
    answer_prompt = file.read()

if 'questions' in st.session_state and st.session_state['questions']:
    st.markdown("---")
    if st.button("💡 Generate Answers", use_container_width=True):
        extracted_text = get_input_text()
        st.subheader("Generated Answers:")

        questions_list = [q.strip() for q in st.session_state['questions'].split("\n") if q.strip()]

        for idx, question in enumerate(questions_list):
            with st.spinner(f"Generating answer for question {idx + 1}..."):
                prompt_for_question = f"{answer_prompt}\n{question}"
                try:
                    if use_rag:
                        answer = st.session_state['rag_helper'].generate_response_with_rag(
                            prompt_for_question,
                            extracted_text,
                            top_k=top_k_questions
                        )
                    else:
                        answer = generate_response(prompt_for_question, extracted_text, use_rag=False)
                except Exception as e:
                    answer = f"Error generating answer: {str(e)}"

            st.markdown(f"**Q{idx+1}: {question}**")
            st.write(f"**Answer:** {answer}")
            st.markdown("---")


# ========== SUMMARY GENERATION WITH RAG ==========
st.markdown("---")
st.subheader("📝 Generate Summary")

summary_prompt_file = "prompts/summary.txt"
with open(summary_prompt_file, 'r') as file:
    summary_prompt = file.read()

col1, col2 = st.columns([3, 1])

with col1:
    if st.button("📄 Generate Summary", use_container_width=True):
        extracted_text = get_input_text()
        
        if extracted_text is None:
            st.error("Please upload a document or enter text manually.")
        else:
            with st.spinner("Generating summary with RAG..."):
                if use_rag:
                    summary = st.session_state['rag_helper'].generate_summary_with_rag(
                        summary_prompt,
                        extracted_text,
                        top_k=top_k_summary
                    )
                else:
                    summary = generate_summary(summary_prompt, extracted_text, use_rag=False)
                    
                st.session_state["summary"] = summary 
                st.success("✅ Summary generated successfully!")
                
                # Show retrieved context if RAG is enabled
                if use_rag:
                    with st.expander("🔍 View Retrieved Key Sections"):
                        summary_query = "main points key concepts important information"
                        chunks = st.session_state['rag_system'].retrieve_context(
                            summary_query,
                            top_k=top_k_summary
                        )
                        for i, chunk in enumerate(chunks, 1):
                            st.markdown(f"**Section {i}** (Relevance: {chunk['similarity_score']:.2f})")
                            st.text(chunk['text'][:200] + "...")
                            st.markdown("---")

with col2:
    if use_rag:
        st.info(f"RAG: ON\nTop-K: {top_k_summary}")
    else:
        st.warning("RAG: OFF")

if 'summary' in st.session_state and st.session_state["summary"]:
    st.markdown("### Document Summary:")
    st.write(st.session_state["summary"])


# ========== AUDIO SUMMARY ==========
if 'summary' in st.session_state and st.session_state["summary"]: 
    if st.button("🔊 Generate Audio Summary"):
        with st.spinner("Generating audio summary..."):
            audio_file_path = text_to_speech_file(st.session_state["summary"])
            st.audio(audio_file_path, format="audio/mp3")
            st.markdown(f"[Download Audio Summary]({audio_file_path})", unsafe_allow_html=True)
            st.success("Audio summary generated successfully!")


# ========== WEB SEARCH SECTION ==========
st.markdown("---")
st.subheader("🌐 Need assistance with a topic?")
user_topic = st.text_input("Enter a topic or question you'd like help with:")

if user_topic:
    search_option = st.selectbox("Where would you like to search?", ["Articles", "YouTube"])

    if st.button("🔍 Generate Web Search"):
        try:
            prompt = f"{search_option} about {user_topic}"
            parsed_results = perform_web_search(prompt)
            
            if parsed_results:
                st.subheader(f"Results for {search_option} on '{user_topic}':")
                for result in parsed_results:
                    st.write(f"**Title**: {result['title']}")
                    st.write(f"**URL**: [{result['url']}]({result['url']})")
                    st.write(f"**Score**: {result['score']:.2f}")
                    st.write(f"**Published Date**: {result['published_date']}")
                    st.markdown("---")
            else:
                st.write(f"No results found for {user_topic} on {search_option}.")
        except Exception as e:
            st.error(f"An error occurred while performing the web search: {str(e)}")