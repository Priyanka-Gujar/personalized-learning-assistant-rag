import streamlit as st
from helpers.ollama_helper import generate_response, generate_summary
from helpers.pdf_reader import extract_text
from helpers.exa_search import perform_web_search
from helpers.exa_search import extract_text_from_exa
from helpers.elevenlabs_helper import text_to_speech_file


# Streamlit UI setup
st.set_page_config(page_title="Personalized Learning Assistant", layout="centered")

st.title("Personalized Learning Assistant")
st.write("Upload a DOCX, PPTX, or PDF **or** enter text manually to generate a rich summary or AI-generated questions and answers!")

# File uploader and text box input
uploaded_file = st.file_uploader("Upload a DOCX, PPTX, or PDF file", type=['docx', 'pptx', 'pdf'])

# Disable text input if a file is uploaded
text_disabled = uploaded_file is not None

# Text input box (Disabled if file is uploaded)
user_text_input = st.text_area("Or enter text manually:", "",height=200, disabled=text_disabled)

# Create a text input for URLs (comma-separated)
user_web_urls = st.text_input("Enter URLs to extract text from (optional, comma-separated):", "", 
                              help="Paste URLs separated by commas to extract content from web pages")

# Initialize variable to store extracted text from URLs
url_extracted_text = ""

# If URLs are provided, extract text from them
if user_web_urls:
    # Split the input by commas and strip whitespace to get a list of URLs
    url_list = [url.strip() for url in user_web_urls.split(',')]    
    
    url_extracted_text = extract_text_from_exa(url_list)
    
    if url_extracted_text:
        st.write('Success')

# Check if any input is provided (file, text, or URLs)
if uploaded_file or user_text_input.strip() or url_extracted_text:
    # Extract text from PDF, URL, or use manual input
    if uploaded_file:
        extracted_text = "Extracted text from PDF"
    elif url_extracted_text:
        extracted_text = url_extracted_text
    else:
        extracted_text = user_text_input


# Dropdown to select education level
level = st.selectbox("Select your education level:", ['High School', 'Bachelors', 'Masters', 'PhD'])

# Dropdown to select question severity
severity = st.selectbox("Select the severity of the questions:", ["Easy", "Medium", "Tough"])

# Number of questions to generate
num_questions = st.number_input("Number of questions to generate:", min_value=1, max_value=20, value=5)


# Determine which prompt file to use
prompt_file = f"prompts/{severity.lower()}_questions.txt"
with open(prompt_file, 'r') as file:
    question_prompt = file.read()

refined_prompt_file = f"prompts/refined_questions.txt"
with open(refined_prompt_file, 'r') as file:
    refined_prompt = file.read()

# Function to get text from either PDF or text box
def get_input_text():
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            return extract_text(uploaded_file)
    elif user_text_input.strip():
        return user_text_input.strip()
    elif url_extracted_text:
        return url_extracted_text
    else:
        return None


# Questions generation section

if st.button("Generate Questions"):
    extracted_text = get_input_text()
    
    if extracted_text is None:
        st.error("Please upload a PDF or enter text manually.")
    else:
        with st.spinner("Generating questions..."):
            formatted_prompt = question_prompt.format(num_questions=num_questions, level=level)
            response = generate_response(formatted_prompt, extracted_text)
            # refine the response
            refined_response = generate_response(refined_prompt, response)
            st.session_state['questions'] = refined_response

            st.subheader("Generated Questions:")
            st.write(refined_response)


# Answer Generation Section

answers_file = "prompts/answers.txt"
with open(answers_file, 'r') as file:
    answer_prompt = file.read()

if 'questions' in st.session_state and st.session_state['questions']:
    if st.button("Generate Answers"):
        extracted_text = get_input_text()
        st.subheader("Generated Answers:")

        questions_list = [q.strip() for q in st.session_state['questions'].split("\n") if q.strip()]

        for idx, question in enumerate(questions_list):
            with st.spinner(f"Generating answer for question {idx + 1}..."):
                prompt_for_question = f"{answer_prompt}\n{question}"
                try:
                    answer = generate_response(prompt_for_question, extracted_text)
                except Exception as e:
                    answer = f"Error generating answer: {str(e)}"

            st.write(f"**Q: {question}**")
            st.write(f"**Answer: {answer}**")
            st.write("")  # Spacing for readability


# Summary generation section

summary_prompt_file = "prompts/summary.txt"
with open(summary_prompt_file, 'r') as file:
    summary_prompt = file.read()

if st.button("Generate Summary"):
    extracted_text = get_input_text()
    
    if extracted_text is None:
        st.error("Please upload a PDF or enter text manually.")
    else:
        with st.spinner("Generating summary..."):
            summary = generate_summary(summary_prompt, extracted_text)
            st.session_state["summary"] = summary 
            st.subheader("Document Summary:")
            st.write(summary)

if 'summary' in st.session_state and st.session_state["summary"]: 
    if st.button("Generate Audio Summary"):
        extracted_text = get_input_text()
    
        if extracted_text is None:
            st.error("Please upload a PDF or enter text manually.")
        else:
            with st.spinner("Generating audio summary..."):
                st.subheader("Document Summary:")
                st.write(st.session_state["summary"])
            # Generate audio file
            audio_file_path = text_to_speech_file(st.session_state["summary"])
            st.audio(audio_file_path, format="audio/mp3")
            st.markdown(f"[Download Audio Summary]({audio_file_path})", unsafe_allow_html=True)
            st.success("Audio summary generated successfully!")


# Web Search Section

st.subheader("Need assistance with a topic?")
user_topic = st.text_input("Enter a topic or question you'd like help with:")

if user_topic:
    search_option = st.selectbox("Where would you like to search?", ["Articles", "YouTube"])

    if st.button("Generate Web Search"):
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
                    st.write("---")
            else:
                st.write(f"No results found for {user_topic} on {search_option}.")
        except Exception as e:
            st.error(f"An error occurred while performing the web search: {str(e)}")
