# Podcast Summarizer

Welcome to the Podcast Summarizer project – a personal tool designed to automatically extract and condense the key points from podcast episodes using AI. Simply provide the RSS feed URL, and the project downloads, transcribes, and summarizes the episode for you.

## Project Demo GIF

<div align="center">
   <img src="content/podcast/group-8-mini-sem-6.gif" width="100%" max-width="800"/>
</div>
![Podcast Summarizer Demo](content/podcast/your_gif_name.gif)

## Demo Overview

Watch the demo GIF to see the process in action:

1. The app opens and displays the main page.
2. You obtain a podcast RSS feed URL from sources like [Listen Notes](https://www.listennotes.com) or [Castos](https://castos.com/tools/find-podcast-rss-feed/).
3. You paste the RSS feed URL into the input field.
4. You click the "Process a Podcast Feed" button.
5. The app downloads the episode in MP3 format.
6. The WhisperX model transcribes the audio into text.
7. The ChatGPT 3.5 Turbo model then generates a concise summary.
8. Steps 5, 6, and 7 run on a GPU-powered backend deployed on [Modal](https://modal.com), while the summary and related details are displayed via a [Streamlit](https://streamlit.io) frontend.

## Key Features

- **Automated Summarization:**  
  Automatically downloads podcast episodes, transcribes the speech, and generates concise summaries using AI.

- **WhisperX Transcription:**  
  Uses the WhisperX model to accurately convert podcast audio into text.

- **ChatGPT 3.5 Turbo Summaries:**  
  Leverages ChatGPT 3.5 Turbo to create informative and coherent summaries from the transcript.

- **Sentiment Analysis:**  
  Provides sentiment scores and interpretations to help you gauge the overall tone of the episode.

- **Docker & Modal Integration:**  
  The project includes a custom Dockerfile for containerization and is deployed on Modal for scalable GPU-powered processing.

- **Streamlit Frontend:**  
  A clean, interactive interface displays episode details, guest info, key highlights, and sentiment analysis.

## Installation & Setup

To run this project locally:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/podcast-summarizer.git
   cd podcast-summarizer

Install Dependencies:

pip install streamlit modal

Deploy the Backend on Modal:

modal deploy /content/podcast/podcast_backend.py

Run the Streamlit Frontend:

    streamlit run podcast_frontend.py

Local Usage

    Open your browser and navigate to http://localhost:8501.

    Paste a podcast RSS feed URL into the input field.

    Click the "Process a Podcast Feed" button to begin the process.

    The app downloads the episode, transcribes it, generates a summary, and displays all relevant details along with sentiment analysis on-screen.

Contributing

We welcome contributions! To get started:

    Fork the repository.

    Create a new branch (git checkout -b feature-name).

    Implement your feature or improvement.

    Commit your changes (git commit -m "Add feature"), then push your branch (git push origin feature-name).

    Open a pull request detailing your changes.

License

This project is licensed under the MIT License.