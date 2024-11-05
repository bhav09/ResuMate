import fitz
import os
import shutil
import time
from datetime import datetime, timedelta
import pandas as pd
from pdf2image import convert_from_path
import json
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import google.generativeai as genai
import requests
import hashlib
from google.cloud import bigquery
from google.oauth2 import service_account
from deep_translator import GoogleTranslator
from streamlit_feedback import streamlit_feedback
from langdetect import detect
import streamlit as st

global session_id, session_time, session_data_df

new_data = None
def upload_to_bq(df, table_name):
    # Define the destination table reference
    destination_table = client.dataset("resumate_pro").table(f'{table_name}')

    # Set write disposition to append
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")

    # Load the DataFrame to the table
    load_job = client.load_table_from_dataframe(df, destination_table, job_config=job_config)
    load_job.result()
    

@st.dialog("❄️ Welcome to ResuMate Pro")
def welcome_message():
    st.snow()
    st.write(f"""
🔍 **Features at a Glance**
1. **Instant Analysis**: Get a full resume review in just 15 seconds!
2. **Global Language Support**: Accessible in 133 languages, making it easy for everyone, everywhere.
3. **Free and Comprehensive**: Enjoy a complete resume check at no cost.
4. **AI-Driven Insights**: Leverages cutting-edge LLMs for precise, personalized feedback.
5. **Versatile for All Roles**: Perfect for both tech and non-tech job applications.

🏁 Ready to transform your resume? Let’s get started!

###### Collects feedback to improve — no personal data 🔒
###### Powered by Google Cloud 🌥️
""")

@st.dialog("Share Your Exclusive Find 🕵️‍♂️")
def share_app():
    # if st.session_state.share_button:
    if 'copy_button_clicked' not in st.session_state:
        st.session_state.copy_button_clicked = False
    app_url = 'https://prompt-debugger-lbgzisv3qa-uc.a.run.app'
    text = f'''Did you know you can get your resume reviewed for free? 🤔
Here's an app - "ResuMate" that does it for you.

Visit this free-to-use tool and get your resume reviewed now 🚀
Link to the app: {app_url}
    '''
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        url = 'https://www.linkedin.com/sharing/share-offsite/?url={app_url}'
        st.link_button('💼 LinkedIn', url)
    with col2:
        url = f'https://x.com/intent/post?original_referer=http%3A%2F%2Flocalhost%3A8502%2F&ref_src=twsrc%5Etfw%7Ctwcamp%5Ebuttonembed%7Ctwterm%5Eshare%7Ctwgr%5E&text={text}+%F0%9F%8E%88&url=%7B{app_url}%7D'
        st.link_button('𝕏 Twitter', url)
    with col3:
        placeholder = st.empty()
        # print(st.session_state)
        if st.session_state.copy_button_clicked:
            placeholder.button("Copied", disabled=True)
        else:
            placeholder.button('📄 Copy Link', on_click=app_url)
    st.text_area("Sample Text", text, height=350)

def _submit_feedback(user_response, emoji=None):
    # Retrieve session_id from session_state
    session_id = st.session_state.get("session_id")
    
    feedback_value = 1 if user_response['score'] == '👍' else 0
    user_feedback = user_response['text']
    
    # Append feedback to the DataFrame and save it
    new_feedback = pd.DataFrame([[session_id, feedback_value, user_feedback]], columns=["session_id", "vote", "comment"])
    upload_to_bq(new_feedback, 'feedback_data')
    st.success("Your feedback has been submitted!")

def store_session_data(session_data_df, session_id, session_time):
    new_row = pd.DataFrame.from_dict({'session_id': [session_id], 'session_creation_time': [session_time]})
    
    # Append the new row to the DataFrame
    session_data_df = pd.concat([session_data_df, new_row], ignore_index=True)

def get_client_ip():
    try:
        ip = requests.get('https://api.ipify.org').text
    except Exception as e:
        ip = "127.0.0.1"  # Fallback to localhost if IP fetch fails
    return ip

def generate_session_id(ip):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    session_string = f"{ip}_{now}"
    session_id = hashlib.sha256(session_string.encode()).hexdigest()
    # print(session_id)
    return session_id, now

def make_new_session(session_data_df):
    ip = get_client_ip()
    session_id, session_time = generate_session_id(ip)
    
    # Store session data in session_state
    st.session_state.session_id = session_id
    st.session_state.session_time = session_time
    st.session_state.created_at = datetime.now()
    st.session_state.welcome_shown = False
    st.session_state.share_button = False
    
    # Store session data in the BQ
    print(session_id, session_time)
    session_data_df['session_id']=[session_id]
    session_data_df['session_creation_time']=[str(session_time)]
    print(session_data_df.shape)
    upload_to_bq(session_data_df, 'session_data')

def detect_and_translate(text):
    # Detect language of the text
    detected_language = detect(text)
    print(detected_language)
    # If the detected language is not English, translate to English
    if detected_language != 'en':
        try:
            translated_text = GoogleTranslator(source=detected_language, target='en').translate(text)
        except:
            translated_text = 'Language not supported'
        return detected_language, translated_text
    else:
        # If already in English, return the original text
        return detected_language, text

def extract_personal_info(role, resume_text, generation_config):
    with open('prompts/extract_information.txt', 'r') as f:
        file_content = f.read().strip()

    output_index = file_content.find("Example format JSON:")
    output_index += len("Example format JSON:")
    extraction_prompt = file_content[:output_index].format(resume_text = resume_text)
    prompt = role + '\n' + extraction_prompt + '\n' + file_content[output_index:]
 
    response = generate_response(prompt, generation_config=generation_config)
    return response

def parse_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    links, font_data, text = [], [], ''
    metadata = {}
    if doc.page_count<=3:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
            text = text.replace('\n', '').replace('  ',' ')
            
            text_instances = page.get_text("dict")["blocks"]
            for block in text_instances:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_info = {
                                "page": page_num + 1,
                                "text": span["text"],
                                "font_name": span["font"],
                                "font_size": span["size"],
                                "color": span["color"],
                            }
                            font_data.append(font_info)
            unique_font_names = {entry['font_name'] for entry in font_data}
            unique_font_sizes = {entry['font_size'] for entry in font_data}

            metadata['font_names'] = list(unique_font_names)
            metadata['font_sizes'] = list(unique_font_sizes)
            try:
                for link in page.get_links():
                    if link["uri"]:
                        links.append(link["uri"])
            except:
                pass
                    
            metadata['urls'] = links
            metadata['text'] = text
            metadata['num_pages'] = doc.page_count
        
        doc.close()
    else:
        metadata['num_pages'] = doc.page_count
    return metadata

def pdf_to_images(pdf_path):
    # Create a folder name with the current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f'images_{current_datetime}'

    # Create the folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert PDF pages to images
    pages = convert_from_path(pdf_path)
    for i, page in enumerate(pages):
        image_path = os.path.join(output_folder, f'page_{i + 1}.png')
        page.save(image_path, 'PNG')
    print(f"All pages have been saved in the '{output_folder}' folder.")
    return output_folder

def generate_response(prompt, generation_config):
    model = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config)
    response = model.generate_content(prompt)
    return response.text.replace('*','') if hasattr(response, 'text') else str(response)

async def resume_analysis(file_path):
    pdf_path = file_path
    metadata = parse_pdf(pdf_path)
    
    if metadata['num_pages'] > 3:
        st.warning('Uploaded file consists more than 3 pages.')
        return None

    with st.expander('Extracted details'):
        # Display Font Names
        st.write("##### Font Details")
        for font_name, font_size in zip(metadata['font_names'], metadata['font_sizes']):
            st.write(f"**Font Name:** {font_name} | **Font Size:** {font_size}")

        url_mapping = {
            'linkedin.com': 'LinkedIn',
            'github.com': 'GitHub',
            'medium.com': 'Medium',
            'devpost.com': 'Devpost',
            'mailto:': 'Email'
        }

        # Display URLs (if any)
        st.write("##### URLs")
        if len(metadata['urls'])!=0:
            platform_names = []
            for url in metadata['urls']:
                try:
                    if 'mailto:' in url:
                        platform_names.append(url_mapping['mailto:'])
                    else:
                        domain = url.split('/')[2]
                        platform_name = url_mapping.get(domain, url)
                        platform_names.append(platform_name)
                except:
                    pass
            st.write(", ".join(platform_names))
        else:
            st.write("No URLs found.")

        # Display Number of Pages
        st.write(f"##### Number of Pages: {metadata['num_pages']}")

    progress = st.progress(0)
    total_steps = 9

    # Step 1: Detect and translate language
    detected_language, text = detect_and_translate(metadata['text'])
    if detected_language == 'Language not supported':
        st.warning('Analysis terminated because language is not available')
        return None

    metadata['text'] = text
    progress.progress(1 / total_steps, text='Language Detected')
    output_folder = pdf_to_images(pdf_path)

    # Define async tasks for parallel processing
    async def extract_info_task():
        with ThreadPoolExecutor() as executor:
            extracted_info = await asyncio.get_event_loop().run_in_executor(
                executor,
                extract_personal_info,
                role,
                metadata['text'],
                generation_config
            )
            extracted_text = extracted_info.replace('```json', '').replace('```', '').strip()
            print(extracted_text)
            return json.loads(extracted_text)

    async def review1_task():
        with open('prompts/guidelines0.txt', 'r') as file:
            guidelines0 = file.read()
        check0 = role + guidelines0.format(content=metadata['text'], num_pages=metadata['num_pages'])
        return await asyncio.get_event_loop().run_in_executor(
            None,
            generate_response,
            [check0],
            generation_config
        )

    async def review2_task():
        with open('prompts/guidelines1.txt', 'r') as file:
            guidelines1 = file.read()
        check1 = role + guidelines1
        return await asyncio.get_event_loop().run_in_executor(
            None,
            generate_response,
            [check1],
            generation_config
        )

    # Execute tasks in parallel
    info_task = asyncio.create_task(extract_info_task())
    review1_task = asyncio.create_task(review1_task())
    review2_task = asyncio.create_task(review2_task())
    
    # Wait for all tasks to complete
    dictionary, review0, review1 = await asyncio.gather(info_task, review1_task, review2_task)
    print(dictionary)
    new_data = pd.DataFrame([dictionary])
    progress.progress(4 / total_steps, text='3 Sanity Checks Completed')

    # Continue with the rest of the sequential processing
    with open('prompts/guidelines2.txt', 'r') as file:
        guidelines2 = file.read()
    check2 = role + guidelines2
    images = [Image.open(f'{output_folder}/{image}') for image in os.listdir(f'{output_folder}/')]
    review2 = generate_response([check2] + images, generation_config)
    progress.progress(5 / total_steps, text='Sanitary Check 3 Completed')

    # Check for action words
    action_words = open('prompts/action_words.txt').read().split(',')
    content = set(metadata['text'].lower().split())
    action_words = set(word.lower() for word in action_words)
    common_words = content.intersection(action_words)
    review3 = f"Action words found: {', '.join(common_words)}.\n" if common_words else "No action words found in resume. You can choose from action words provided."
    progress.progress(6 / total_steps, text='Action words analysed')

    # Check for fonts
    font_guidelines = {'Times New Roman', 'Arial', 'Helvetica', 'Verdana', 'Calibri'}
    common_fonts = set(metadata['font_names']).intersection(font_guidelines)
    review4 = f"Fonts found: {', '.join(common_fonts)}. Font Sizes Detected: {metadata['font_sizes']}. Keep it in the range of 10-12." if common_fonts else f"Suggested fonts not found. Use fonts like {font_guidelines}."
    progress.progress(7 / total_steps, text='Font details analysed')

    # Generate final review
    reviews = [review0, review1, review2, review3, review4]
    final_review_prompt = open('prompts/final_review.txt', 'r').read()
    prompt = final_review_prompt.format(
        role=role,
        resume_text=metadata['text'],
        detected_language=detected_language,
        common_fonts=common_fonts,
        resume_font_size=metadata['font_sizes'],
        reviews=reviews
    )
    final_review = generate_response([prompt] + images, generation_config)
    with st.expander("Show Final Review", expanded=True):
        st.write(final_review)
    progress.progress(8 / total_steps, text='Final Review Completed')

    # Cleanup
    shutil.rmtree(output_folder)
    progress.progress(1.0, text='Resume Reviewed Successfully')
    return new_data    

def footer():
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #0E1117;
            color: #FAFAFA;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
        }
        .footer a {
            color: #318ce7;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        </style>
        <div class="footer">
            Built with 💙 by <a href="https://www.linkedin.com/in/bhavishya-pandit/" target="_blank">Bhavishya Pandit</a>.  
            Let's connect on <a href="https://x.com/BhavishyaP9" target="_blank">Twitter</a> and 
            <a href="https://www.linkedin.com/in/bhavishya-pandit/" target="_blank">LinkedIn</a>.
        </div>
        """,
        unsafe_allow_html=True
    )


session_data_df = pd.DataFrame(columns=["session_id", "session_creation_time"])
json_key_path = 'service-account-key.json'  # Update with your service account key path
credentials = service_account.Credentials.from_service_account_file(json_key_path)

# Create a BigQuery client using the service account credentials
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# Load session data
if 'session_id' not in st.session_state:
    make_new_session(session_data_df)
else:
    # Check if session is older than 5 minutes
    session_age = datetime.now() - st.session_state.created_at
    if session_age > timedelta(minutes=5):
        make_new_session(session_data_df)

st.set_page_config(page_title="ResuMate Pro", page_icon="🚀")

if 'copy_button_clicked' not in st.session_state:
    st.session_state.copy_button_clicked = False

if 'feedback_key' not in st.session_state:
    st.session_state.feedback_key = str(uuid.uuid4())

if not st.session_state.welcome_shown:
    welcome_message()
    st.session_state.welcome_shown = True
    st.session_state.share_button = True

with open('credentials.json') as config_file:
    creds = json.load(config_file)
genai.configure(api_key=creds['token'])

generation_config = {
"temperature": 0.5,
"top_p": 0.98,
"top_k": 64,
"max_output_tokens": 1024,
"response_mime_type": "text/plain",
}

role = 'You are an expert Resume Reviewer - who has reviewed resumes for both tech and non tech roles. Your resume suggestions have helped people land new opportunities.'
# Set page configuration
st.title('❄️ ResuMate Pro')
_, col_share_button = st.columns([0.7, 0.15])
col_share_button.button("Share app 🔄", key="share", on_click=share_app)

st.markdown(
    """
    <style>
    .element-container {
            position: relative;
            top: -10px;
        }

    .st-at.st-ai.st-au.st-av.st-aw.st-ax.st-ay.st-az.st-b0.st-b1.st-b2.st-b3.st-b4{
        position: relative;
        top: -80px;
    }
    .st-av.st-bv.st-co.st-bg.st-bh.st-be {
        position: relative;
        top: -65px;  /* Adjust as needed to move the elements up */
    }
    .MuiStack-root.css-16ogmd7
    {
        position: relative;
        top: -65px;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload your resume", accept_multiple_files=False, type=['pdf'])
submit_button = st.button("Submit")
footer()

if uploaded_file is not None and submit_button:
    temp_file_path = os.path.join("/tmp", uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
    
    start = time.time()
    new_data = asyncio.run(resume_analysis(temp_file_path))
    end = time.time()
    total_time = round((end - start), 2) - 1.25
    if total_time==-1.25:
        total_time=0
    st.success(f'Total time taken: {total_time} seconds')

streamlit_feedback(
                    feedback_type="thumbs",
                    optional_text_label="Please provide extra information",
                    on_submit=_submit_feedback,
                    key=st.session_state.feedback_key,
                )
