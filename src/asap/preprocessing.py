import re 

def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    text= re.sub(r"[إأآا]", "ا", text)  # Alef variants → Alef
    text= text.replace("ى", "ي")        # Alef Maksura → Ya
    text= text.replace("ة", "ه")        # Ta Marbuta → Ha
    text= re.sub(r"گ", "ك", text)        # Gaf → Kaf
    text= re.sub(r"[ؗ-ًؚ-ْ]", "", text)  # harakat, shadda, sukun
    text= re.sub(r'ـ', "", text) # Remove Tatweel

    text= re.sub(r"http\S+|www\.\S+", " ", text)   # URLs
    text= re.sub(r"\S+@\S+\.\S+", " ", text) # Emails
    text= re.sub(r"[@#]\w+", " ", text)    # Mentions & Hashtags
    text= re.sub(r"\s+", " ", text).strip()  # Collapse whitespace
    text= re.sub(r'(.)\1{2,}', r'\1', text)   # Collapse 3+ repeated chars

    return text.strip()