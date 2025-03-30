import modal

# Define the Modal app
app = modal.App("podcast-project")

def download_whisperX():
    # Pre-download the WhisperX model to speed up subsequent calls.
    import whisperx
    print("Downloading the WhisperX model...")
    device = "cpu"
    compute_type = "float32"
    _ = whisperx.load_model("medium", device, compute_type=compute_type)

# Build the podcast image using our Dockerfile.
podcast_image = modal.Image.from_dockerfile("Dockerfile")\
    .pip_install(
        "feedparser",
        "requests",
        "ffmpeg",
        "openai==0.28",  # Pin openai to version 0.28 to use the legacy API.
        "tiktoken",
        "wikipedia",
        "ffmpeg-python",
        "googlesearch-python",
        "transformers",
        "nltk",
        "numpy<2.0",         # Force numpy 1.x for compatibility
        "ctranslate2",
        "faster-whisper"
    )

# Install PyTorch 1.10.0 with CUDA 10.2 (cu102) support.
podcast_image = podcast_image.pip_install(
    "torch==1.10.0+cu102",
    "torchvision==0.11.1+cu102",
    "torchaudio==0.10.0+cu102",
    extra_index_url="https://download.pytorch.org/whl/cu102"
)

# Install a compatible version of pyannote.audio (3.3.0)
podcast_image = podcast_image.pip_install("pyannote.audio==3.3.0")

# Instead of installing from GitHub, install the official release of whisperx from PyPI.
podcast_image = podcast_image.pip_install("whisperx==3.3.1")

# Run download_whisperX to pre-download the model.
podcast_image = podcast_image.run_function(download_whisperX)

@app.function(image=podcast_image, gpu="any", timeout=600)
def get_transcribe_podcast(rss_url: str, local_path: str):
    print("Starting Podcast Transcription Function")
    print("Feed URL:", rss_url)
    print("Local Path:", local_path)
    
    import feedparser
    intelligence_feed = feedparser.parse(rss_url)
    podcast_title = intelligence_feed['feed'].get('title', 'Unknown Podcast')
    episode = intelligence_feed.entries[0]
    episode_title = episode.get('title', 'Unknown Episode')
    episode_image = intelligence_feed['feed']['image'].get('href', '') if 'image' in intelligence_feed['feed'] else ''
    
    # Get the audio URL from the first entry's links.
    episode_url = None
    for item in episode.links:
        if item.get('type') == 'audio/mpeg':
            episode_url = item.get('href')
            break
    if not episode_url:
        raise ValueError("No audio URL found in RSS feed.")
    
    episode_name = "podcast_episode.mp3"
    print("RSS read successfully; episode URL:", episode_url)
    
    from pathlib import Path
    p = Path(local_path)
    p.mkdir(parents=True, exist_ok=True)
    print("Downloading the podcast episode from:", episode_url)
    import requests
    with requests.get(episode_url, stream=True) as r:
        r.raise_for_status()
        episode_path = p.joinpath(episode_name)
        with open(episode_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Podcast episode downloaded to:", episode_path)
    
    import whisperx
    device = "cuda"
    batch_size = 32  # Adjust based on GPU memory availability.
    compute_type = "float16"  # or "int8" if needed.
    print("Loading the Whisper model...")
    model = whisperx.load_model("medium", device=device, compute_type=compute_type)
    
    audio = whisperx.load_audio(local_path + episode_name)
    print("Starting podcast transcription...")
    result = model.transcribe(audio, batch_size=batch_size)
    
    combined_text = "".join(segment["text"] for segment in result.get("segments", []))
    print("Podcast transcription completed.")
    return {
        "podcast_title": podcast_title,
        "episode_title": episode_title,
        "episode_image": episode_image,
        "episode_transcript": combined_text,
    }

@app.function(image=podcast_image, secrets=[modal.Secret.from_name("my-openai-secret")])
def get_podcast_summary(podcast_transcript: str) -> str:
    import openai, tiktoken
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
    if len(enc.encode(podcast_transcript)) >= 16385:
        print("Transcript too long for summary generation.")
        return ""
    instructPrompt = (
        "You are an expert copywriter who summarizes podcasts for newsletters. "
        "Please write a concise summary covering the key points discussed in the transcript below, "
        "omitting introductions and sponsorship details.\n\n"
    )
    request = instructPrompt + podcast_transcript
    print("Generating podcast summary...")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request},
        ],
        max_tokens=256,
    )
    summary = response.choices[0].message.content
    print("Podcast summary generated.")
    return summary

@app.function(image=podcast_image, secrets=[modal.Secret.from_name("my-openai-secret")])
def get_podcast_guest(podcast_transcript: str) -> dict:
    import openai, wikipedia, json
    from googlesearch import search
    snippet = podcast_transcript[:15000]
    print("Generating podcast guest information...")
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role": "user", "content": snippet}],
        max_tokens=256,
        functions=[
            {
                "name": "get_podcast_guest_information",
                "description": "Get guest information using their name and job",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "guest_name": {"type": "string", "description": "Guest name"},
                        "guest_job": {"type": "string", "description": "Guest job"},
                    },
                    "required": ["guest_name", "guest_job"],
                },
            }
        ],
        function_call={"name": "get_podcast_guest_information"},
    )
    msg = completion["choices"][0]["message"]
    if msg.get("function_call"):
        args = json.loads(msg["function_call"]["arguments"])
        guest_name = args.get("guest_name", "")
        guest_job = args.get("guest_job", "")
    else:
        guest_name, guest_job = "", ""
    
    query = f"{guest_name} {guest_job}"
    print("Searching for guest info using query:", query)
    def get_wiki_info(search_term):
        try:
            result = wikipedia.search(search_term, results=1)
            if not result:
                return "", "", "", ""
            page = wikipedia.WikipediaPage(title=result[0])
            return page.title, page.summary, page.url, ""
        except Exception as e:
            print("Wikipedia lookup error:", e)
            return "", "", "", ""
    wiki_title, wiki_summary, wiki_url, wiki_img = get_wiki_info(query)
    
    search_results = []
    try:
        for res in search(query):
            search_results.append(res)
    except Exception:
        search_results = [""]
    
    guest_info = {
        "name": guest_name,
        "job": guest_job,
        "wiki_title": wiki_title,
        "wiki_summary": wiki_summary,
        "wiki_url": wiki_url,
        "wiki_img": wiki_img,
        "google_URL": search_results[0] if search_results else "",
    }
    print("Podcast guest information generated.")
    return guest_info

@app.function(image=podcast_image, secrets=[modal.Secret.from_name("my-openai-secret")])
def get_podcast_highlights(podcast_transcript: str) -> str:
    import openai, tiktoken
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
    if len(enc.encode(podcast_transcript)) >= 16385:
        print("Transcript too long for highlights generation.")
        return ""
    instructPrompt = (
        "You are a podcast editor and producer. Using the transcript below, identify the 5 most significant highlights:\n"
        "- Each highlight must be a statement by one of the podcast guests.\n"
        "- Each must be impactful and a key takeaway.\n"
        "- Each must be concise and entice the listener.\n"
        "- The highlights should be spread throughout the episode.\n\n"
        "Format each as:\n"
        "- Highlight 1 of the podcast\n"
        "- Highlight 2 of the podcast\n"
        "- Highlight 3 of the podcast\n"
        "- Highlight 4 of the podcast\n"
        "- Highlight 5 of the podcast\n\n"
        "Use only the transcript below to extract these highlights and provide only the highlights.\n\n"
    )
    request = instructPrompt + podcast_transcript
    print("Generating podcast highlights...")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request},
        ],
        max_tokens=256,
    )
    highlights = response.choices[0].message.content
    print("Podcast highlights generated.")
    return highlights

# New Sentiment Analysis Function using VADER with custom adjustments and interpretation
@app.function(image=podcast_image)
def get_podcast_sentiment(podcast_transcript: str) -> dict:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
    analyzer = SentimentIntensityAnalyzer()
    
    # Overall sentiment
    overall_sentiment = analyzer.polarity_scores(podcast_transcript)
    
    # Speaker-level sentiment (if transcript is formatted as "Speaker: text")
    speaker_texts = {}
    for line in podcast_transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            speaker, text = line.split(":", 1)
            speaker = speaker.strip()
            text = text.strip()
            if speaker in speaker_texts:
                speaker_texts[speaker] += " " + text
            else:
                speaker_texts[speaker] = text
    speakers_sentiment = {speaker: analyzer.polarity_scores(text) for speaker, text in speaker_texts.items()}
    
    # Custom adjustment: modify overall compound score based on keywords
    positive_keywords = ["exciting", "amazing", "fantastic", "inspiring", "excellent"]
    negative_keywords = ["terrible", "awful", "bad", "poor", "disappointing"]
    adjustment = 0.0
    transcript_lower = podcast_transcript.lower()
    for word in positive_keywords:
        if word in transcript_lower:
            adjustment += 0.02
    for word in negative_keywords:
        if word in transcript_lower:
            adjustment -= 0.02
    overall_sentiment["compound"] = max(min(overall_sentiment["compound"] + adjustment, 1.0), -1.0)
    
    # Interpretation based on compound score
    def interpret_sentiment(overall):
        compound = overall.get("compound", 0)
        if compound >= 0.7:
            return "The podcast expresses a highly positive and enthusiastic tone."
        elif compound >= 0.3:
            return "The podcast has a moderately positive tone."
        elif compound > -0.3:
            return "The podcast tone is neutral."
        elif compound > -0.7:
            return "The podcast has a moderately negative tone."
        else:
            return "The podcast expresses a highly negative tone."
    
    interpretation = interpret_sentiment(overall_sentiment)
    
    # Explanation of metrics for the user
    metrics_explanation = {
        "compound": "Overall sentiment score from -1 (extremely negative) to +1 (extremely positive).",
        "pos": "Proportion of text considered positive.",
        "neu": "Proportion of text considered neutral.",
        "neg": "Proportion of text considered negative."
    }
    
    return {
        "overall": overall_sentiment,
        "speakers": speakers_sentiment,
        "interpretation": interpretation,
        "metrics_explanation": metrics_explanation
    }

@app.function(image=podcast_image, secrets=[modal.Secret.from_name("my-openai-secret")], timeout=1800)
def process_podcast(url: str, path: str) -> dict:
    print("Starting process_podcast with URL:", url)
    output = {}
    print("Calling get_transcribe_podcast...")
    details = get_transcribe_podcast.remote(url, path)
    print("Transcription job submitted.")
    print("Calling get_podcast_summary...")
    summary = get_podcast_summary.remote(details["episode_transcript"])
    print("Summary job submitted.")
    print("Calling get_podcast_guest...")
    guest = get_podcast_guest.remote(details["episode_transcript"])
    print("Guest info job submitted.")
    print("Calling get_podcast_highlights...")
    highlights = get_podcast_highlights.remote(details["episode_transcript"])
    print("Highlights job submitted.")
    print("Calling get_podcast_sentiment...")
    sentiment = get_podcast_sentiment.remote(details["episode_transcript"])
    print("Sentiment analysis job submitted.")
    output["podcast_details"] = details
    output["podcast_summary"] = summary
    output["podcast_guest"] = guest
    output["podcast_highlights"] = highlights
    output["podcast_sentiment"] = sentiment
    print("process_podcast completed.")
    return output

def get_wiki_info(search_term: str):
    import wikipedia, requests, json
    try:
        print("Searching Wikipedia for guest...")
        result = wikipedia.search(search_term, results=1)
        wikipedia.set_lang('en')
        page = wikipedia.WikipediaPage(title=result[0])
        title = page.title
        url = page.url
        summary = page.summary
        response = requests.get(
            'http://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&piprop=original&titles=' + title
        )
        data = json.loads(response.text)
        try:
            img_link = list(data['query']['pages'].values())[0]['original']['source']
        except Exception:
            img_link = ""
        return title, summary, url, img_link
    except wikipedia.exceptions.PageError:
        print("The page for guest does not exist on Wikipedia.")
        return "", "", "", ""
    except wikipedia.exceptions.DisambiguationError as e:
        print("The page for guest is ambiguous. Options:", e.options)
        return "", "", "", ""
    except Exception:
        return "", "", "", ""

@app.local_entrypoint()
def test_method(url: str, path: str):
    result = process_podcast(url, path)
    print("Test method result:", result)
