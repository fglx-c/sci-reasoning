def get_content_between_tags(start_tag, end_tag, text):
    """
    Extract content between XML-like tags in text.
    
    Args:
        start_tag (str): Opening tag (e.g., "<novelty>")
        end_tag (str): Closing tag (e.g., "</novelty>")
        text (str): Input text to search
    
    Returns:
        str: Extracted content, stripped of whitespace
    """
    extracted_text = ""
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag) : end_index] + " "
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break
    return extracted_text.strip()


def extract_tagged_content(text, tag_name, fallback_to_full_text=True):
    """
    Extract content from XML-like tags with fallback options.
    
    Args:
        text (str): Input text
        tag_name (str): Name of tag to extract (without brackets)
        fallback_to_full_text (bool): If True, return full text when tag not found
    
    Returns:
        str: Extracted content or fallback
    """
    if not text:
        return ""
        
    target_str = get_content_between_tags(f"<{tag_name}>", f"</{tag_name}>", text)
    if target_str:
        return target_str
    elif fallback_to_full_text:
        return text
    else:
        return ""


def extract_json_content(text):
    """
    Extract JSON content from markdown code blocks.
    
    Args:
        text (str): Input text potentially containing ```json blocks
    
    Returns:
        str: Extracted JSON content
    """
    if "```json" in text:
        return get_content_between_tags("```json", "```", text)
    else:
        return text


def clean_and_normalize_text(text):
    """
    Clean and normalize text for evaluation.
    
    Args:
        text (str): Input text
    
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Remove markdown artifacts
    text = text.replace("**", "").replace("*", "")
    
    return text.strip()


def truncate_text(text, max_length=2000, preserve_sentences=True):
    """
    Truncate text to a maximum length while preserving readability.
    
    Args:
        text (str): Input text
        max_length (int): Maximum character length
        preserve_sentences (bool): If True, try to break at sentence boundaries
    
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    if preserve_sentences:
        # Try to break at sentence boundaries
        sentences = text.split('. ')
        truncated = ""
        for sentence in sentences:
            if len(truncated + sentence + '. ') <= max_length:
                truncated += sentence + '. '
            else:
                break
        if truncated:
            return truncated.strip()
    
    # Fallback to character truncation
    return text[:max_length].strip() + "..." 