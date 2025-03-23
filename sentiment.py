import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon', quiet=True)

def analyze_sentiment(transcript: str) -> dict:
    """
    Analyzes sentiment for a podcast transcript.
    The transcript is assumed to be formatted so that each line is in the form:
        Speaker Name: spoken text...
    
    Returns a dictionary with overall sentiment and sentiment per speaker.
    Uses VADER for initial scores, then applies simple custom rules.
    """
    # Initialize VADER
    analyzer = SentimentIntensityAnalyzer()
    
    # Group text by speaker and overall text
    speaker_texts = {}
    overall_text = ""
    
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        # Assume speaker text lines are of the format "Speaker: text"
        if ":" in line:
            speaker, text = line.split(":", 1)
            speaker = speaker.strip()
            text = text.strip()
            overall_text += " " + text
            if speaker in speaker_texts:
                speaker_texts[speaker] += " " + text
            else:
                speaker_texts[speaker] = text
        else:
            # If no speaker indicated, append to overall text
            overall_text += " " + line
    
    # Compute sentiment scores using VADER
    overall_sentiment = analyzer.polarity_scores(overall_text)
    speakers_sentiment = {speaker: analyzer.polarity_scores(text) for speaker, text in speaker_texts.items()}
    
    # Custom rules to adjust the overall compound score:
    # Increase positive sentiment if these keywords appear; decrease if negative keywords are found.
    positive_keywords = ["exciting", "amazing", "inspiring", "great", "fantastic"]
    negative_keywords = ["terrible", "disappointed", "awful", "bad", "poor"]
    
    adjustment = 0.0
    text_lower = overall_text.lower()
    for word in positive_keywords:
        if word in text_lower:
            adjustment += 0.05
    for word in negative_keywords:
        if word in text_lower:
            adjustment -= 0.05
    
    # Adjust the overall compound score (capping it between -1 and 1)
    overall_sentiment["compound"] = max(min(overall_sentiment["compound"] + adjustment, 1.0), -1.0)
    
    return {
        "overall": overall_sentiment,
        "speakers": speakers_sentiment
    }

# Example usage:
if __name__ == "__main__":
    sample_transcript = """
    Alice: I think this podcast was absolutely amazing and very inspiring.
    Bob: I disagree. I found it quite boring and a bit disappointing.
    Alice: But overall, the ideas were great!
    """
    sentiment_results = analyze_sentiment(sample_transcript)
    print("Overall sentiment:", sentiment_results["overall"])
    print("Speaker sentiments:", sentiment_results["speakers"])
