import streamlit as st
import modal
import json
import os
import base64
import pandas as pd

# Custom CSS for a dark theme with refined typography and layout adjustments.
def set_custom_css():
    custom_css = """
    <style>
    /* Import Google Fonts: Montserrat for headers and Roboto for body text */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&family=Roboto:wght@400;500&display=swap');

    /* Overall app styling */
    .stApp {
        background-color: #000000;
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
    }
    /* Title banner style */
    .title-banner {
        font-family: 'Montserrat', sans-serif;
        font-size: 3.8rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        padding: 20px 0;
        background: #000000;
        border-bottom: 2px solid #ffffff;
        margin-bottom: 20px;
    }
    /* Input container style for RSS feed (compact) */
    .input-container {
        background-color: #1e1e1e;
        padding: 15px;
        border: 1px solid #ffffff;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    /* Input title style */
    .input-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 10px;
    }
    /* Section headers for main content */
    .section-header {
        border-bottom: 2px solid #ffffff;
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 15px;
        color: #ffffff;
    }
    /* Card style for podcast details */
    .podcast-card {
        background: #1e1e1e;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    a {
        color: #ffffff;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Function to set a background image from a file
def set_png_as_page_bg(png_file):
    @st.cache_data(show_spinner=False)
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        opacity: 0.85;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

def main():
    # Apply background image and custom CSS
    set_png_as_page_bg('content/black-brick-wall-textured-background.jpg')
    set_custom_css()
    
    # Title banner at the top center
    st.markdown('<div class="title-banner">Podcast Summarizer</div>', unsafe_allow_html=True)
    
    # RSS Feed Input Container (placed at the top, compact)
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown('<div class="input-title">Enter RSS Feed URL</div>', unsafe_allow_html=True)
    rss_url = st.text_input("", placeholder="e.g., https://feeds.npr.org/510318/podcast.xml")
    process_button = st.button(":heavy_check_mark: Process Feed")
    st.markdown(
        "<small><em>Note: Processing takes about a minute. Once complete, your podcast will appear below.</em></small>",
        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content: Podcast selection from existing ones (displayed below the RSS input)
    available_podcast_info = create_dict_from_json_files('content')
    if available_podcast_info:
        keys = list(available_podcast_info.keys())
        if 'selected_podcast' in st.session_state and st.session_state['selected_podcast'] in keys:
            default_index = keys.index(st.session_state['selected_podcast'])
        else:
            default_index = len(keys) - 1
        selected_podcast = st.selectbox("Select Podcast", options=keys, index=default_index)
        if selected_podcast:
            podcast_info = available_podcast_info[selected_podcast]
            st.markdown(f"<h2 class='section-header'>{podcast_info['podcast_details']['podcast_title']}</h2>", unsafe_allow_html=True)
            st.subheader("Episode Title")
            st.write(podcast_info['podcast_details']['episode_title'])
            st.subheader("Episode Summary")
            st.write(podcast_info['podcast_summary'])
            st.image(podcast_info['podcast_details']['episode_image'], caption="Podcast Cover", width=300)
            
            st.markdown("<h3 class='section-header'>Guest Information</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns([3, 7])
            with col1:
                st.subheader("Guest")
                if podcast_info['podcast_guest']['wiki_img']:
                    st.image(podcast_info['podcast_guest']['wiki_img'],
                             caption=f"{podcast_info['podcast_guest']['name']}, {podcast_info['podcast_guest']['job']}",
                             width=300)
                else:
                    st.write(f"{podcast_info['podcast_guest']['name']}, {podcast_info['podcast_guest']['job']}")
            with col2:
                st.subheader("Details")
                guest_details_text = (
                    podcast_info['podcast_guest']['wiki_title'] + "\n\n" +
                    podcast_info['podcast_guest']['wiki_summary'] + "\n\n" +
                    "More info: " + podcast_info['podcast_guest']['wiki_url'] + "\n" +
                    "Google: " + podcast_info['podcast_guest']['google_URL']
                )
                st.write(guest_details_text)
    
            st.markdown("<h3 class='section-header'>Key Moments</h3>", unsafe_allow_html=True)
            for moment in podcast_info['podcast_highlights'].split('\n'):
                st.markdown(f"<p style='margin-bottom: 5px;'>{moment}</p>", unsafe_allow_html=True)
            
            # Display Sentiment Analysis in a table format
            st.markdown("<h3 class='section-header'><strong>Sentiment Analysis</strong></h3>", unsafe_allow_html=True)
            sentiment = podcast_info.get("podcast_sentiment", {})
            if sentiment:
                overall = sentiment.get("overall", {})
                interpretation = sentiment.get("interpretation", "")
                explanation = sentiment.get("metrics_explanation", {})
                metrics_data = []
                for key in ['compound', 'pos', 'neu', 'neg']:
                    metrics_data.append({
                        "Metric": key.upper(),
                        "Value": f"{overall.get(key):.2f}",
                        "Description": explanation.get(key, "")
                    })
                st.table(pd.DataFrame(metrics_data))
                st.markdown(f"**Interpretation:** {interpretation}")
                speakers = sentiment.get("speakers", {})
                if speakers:
                    st.write("Speaker Sentiments:")
                    for speaker, scores in speakers.items():
                        st.write(f"{speaker}: Compound: {scores.get('compound'):.2f}, Pos: {scores.get('pos'):.2f}, Neu: {scores.get('neu'):.2f}, Neg: {scores.get('neg'):.2f}")
    else:
        st.info("No podcasts processed yet. Please add a new RSS feed above.")
    
    if process_button and rss_url:
        podcast_info = process_podcast_info(rss_url)
        if podcast_info["podcast_summary"] == "" or podcast_info["podcast_highlights"] == "":
            st.error(
                "Error processing RSS URL. The episode might be too long. Please choose an episode shorter than 1 hour."
            )
        else:
            new_podcast_name = get_next_available_name(available_podcast_info)
            available_podcast_info[new_podcast_name] = podcast_info
            save_path = os.path.join('content', new_podcast_name)
            with open(save_path, 'w') as json_file:
                json.dump(podcast_info, json_file, indent=4)
            st.session_state['selected_podcast'] = new_podcast_name
            st.rerun()

def create_dict_from_json_files(folder_path):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    json_files = sorted(json_files)
    data_dict = {}
    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            podcast_info = json.load(file)
            podcast_name = podcast_info['podcast_details']['podcast_title']
            data_dict[podcast_name] = podcast_info
    return data_dict

def process_podcast_info(url):
    f = modal.Function.lookup("podcast-project", "process_podcast")
    output = f.remote(url, '/tmp/podcast/')
    return output

def get_next_available_name(existing_podcasts):
    idx = len(existing_podcasts.keys()) + 1
    return f"podcast-{idx}.json"

if __name__ == '__main__':
    main()
