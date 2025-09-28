# rag_components/utils.py
import re
from thefuzz import process, fuzz

def check_and_correct_intent(intent: str, all_fund_names: list, threshold: int = 85):
    """
    Finds and corrects fund names using a non-overlapping n-gram strategy,
    and flags for user disambiguation if a name is ambiguous.
    """
    stop_words = {'fund', 'funds', 'and', 'the', 'is', 'of', 'in', 'for', 'with', 'or', 'vs', 'compare', 'show', 'tell', 'me', 'better'}
    
    original_intent = intent
    words = [word for word in re.split(r'\s+', intent.lower()) if word not in stop_words and len(word) > 2]
    
    # --- Step 1: Find all possible matches for n-grams ---
    all_matches = []
    # Check for 4-grams down to 1-grams
    for n in range(min(len(words), 4), 0, -1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n])
            # Use process.extract to find top matches for this phrase
            top_matches = process.extract(phrase, all_fund_names, scorer=fuzz.token_set_ratio, limit=5)
            good_matches = [(match[0], match[1]) for match in top_matches if match[1] >= threshold]
            
            if good_matches:
                all_matches.append({'phrase': phrase, 'matches': good_matches, 'len': n})

    # --- Step 2: Select the best, non-overlapping corrections ---
    final_corrections = []
    corrected_indices = set()
    
    for match_info in sorted(all_matches, key=lambda x: x['len'], reverse=True):
        phrase = match_info['phrase']
        
        # Find start index of the phrase
        start_index = -1
        try:
            phrase_words = phrase.split()
            for i in range(len(words) - len(phrase_words) + 1):
                if words[i:i+len(phrase_words)] == phrase_words:
                    start_index = i
                    break
        except Exception: continue
        if start_index == -1: continue
        
        # Check if this part has already been corrected by a longer phrase
        if any(i in corrected_indices for i in range(start_index, start_index + match_info['len'])):
            continue

        # --- DISAMBIGUATION LOGIC ---
        if len(match_info['matches']) > 1:
            # If multiple strong matches are found, ask the user to clarify.
            print(f"Disambiguation needed for '{phrase}': Found multiple funds.")
            return {
                "status": "disambiguation_needed",
                "question_text": f"Which '{phrase}' fund did you mean?",
                "original_name": phrase,
                "options": [m[0] for m in match_info['matches']]
            }
        
        # If we have exactly one good match, it's an unambiguous correction.
        if len(match_info['matches']) == 1:
            correction = match_info['matches'][0][0]
            final_corrections.append({'original': phrase, 'corrected': correction})
            # Mark these word indices as corrected
            for i in range(start_index, start_index + match_info['len']):
                corrected_indices.add(i)

    # --- Step 3: Rebuild the intent string ---
    if not final_corrections:
        return {"status": "unambiguous", "intent": original_intent}

    corrected_intent = original_intent
    for correction in final_corrections:
        try:
            # Use regex for robust, case-insensitive replacement
            corrected_intent = re.sub(r'\b' + re.escape(correction['original']) + r'\b', correction['corrected'], corrected_intent, flags=re.IGNORECASE)
            print(f"Final Correction Applied: '{correction['original']}' -> '{correction['corrected']}'")
        except re.error:
            corrected_intent = corrected_intent.replace(correction['original'], correction['corrected'])
            
    return {"status": "unambiguous", "intent": corrected_intent}