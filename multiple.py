
from flask import Flask, render_template, request
import os
import pytesseract
import cv2
import string
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import re
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import PyPDF2
from docx import Document
import re
from spellchecker import SpellChecker
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

nltk.download('stopwords')
nltk.download("punkt")

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def calculate_jaccard_similarity(text1, text2, n=1):
    words1 = text1.split()
    words2 = text2.split()

    ngrams1 = set(ngrams(words1, n))
    ngrams2 = set(ngrams(words2, n))

    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)

    jaccard_similarity = len(intersection) / len(union)
    return jaccard_similarity
def evaluate_answer_with_keywords(answer_text, keywords):
    # Calculate the number of overlapping keywords
    answer_words = answer_text.split()
    overlapping_keywords = [word for word in answer_words if word in keywords]
    
    # Calculate the score based on the number of overlapping keywords
    score = (len(overlapping_keywords) / len(keywords)) * 100
    return score
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r') as txt_file:
        text = txt_file.read()
    return text
def extract_text_from_file(file_path):
    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.lower().endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file format")
@app.route('/')
def index():
    return render_template('image1.html')

@app.route('/upload', methods=['POST'])
def upload():
    # if 'imageInput' not in request.files:
    #     return "Please select an image or pdf"
    
    # file = request.files['imageInput']
    
    # if file.filename == '':
    #     return "Please select an image or pdf"
    if 'imageInput1' not in request.files or 'imageInput2' not in request.files:
        return "Both images are required"
    
    file1 = request.files['imageInput1']
    file2 = request.files['imageInput2']
    
    if file1.filename == '' or file2.filename == '':
        return "Both files are required"
    
    # if file:
    #     filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    #     file.save(filename)
        
    #     if file.filename.endswith('.pdf') or file.filename.endswith('.docx') or file.filename.endswith('.txt'):
    #         extracted_text = extract_text_from_file(filename)
    #     else:
    #         extracted_text = pytesseract.image_to_string(file)
        
    #     text1 = extracted_text.translate(str.maketrans('', '', string.punctuation))
    #     text2 = extracted_text.translate(str.maketrans('', '', string.punctuation))
    #     question_key_keywords = set(text2.split())
    #     answer_score = evaluate_answer_with_keywords(text1, question_key_keywords)
    if file1 and file2:
        filename1 = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        filename2 = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename )
        
        file1.save(filename1)
        file2.save(filename2)
        
        # image1 = cv2.imread(filename1)
        # image2 = cv2.imread(filename2)
        
        # text1 = pytesseract.image_to_string(image1)
        # text2 = pytesseract.image_to_string(image2)
        if (file1.filename.endswith('.pdf') or file1.filename.endswith('.docx') or file1.filename.endswith('.txt')) and (file2.filename.endswith('.pdf') or file2.filename.endswith('.docx') or file2.filename.endswith('.txt')):
            text1 = extract_text_from_file(filename1)
            text2 = extract_text_from_file(filename2)
        elif(file1.filename.endswith('.jpg') or file1.filename.endswith('.jpeg') or file1.filename.endswith('.png')) and (file2.filename.endswith('.jpg') or file2.filename.endswith('.jpeg') or file2.filename.endswith('.png')):
            text1 = pytesseract.image_to_string(filename1)
            text2 = pytesseract.image_to_string(filename2)
        else:
            return "Unsupported file format"
            
        
        #text1 = text1.translate(str.maketrans('', '', string.punctuation))
        #text2 = text2.translate(str.maketrans('', '', string.punctuation))
        question_key_keywords = set(text2.split())
        answer_score = evaluate_answer_with_keywords(text1, question_key_keywords)


        
        def spellchecker(text2): 
            spell = SpellChecker()           
            words = text2.split()
            corrected_words = [spell.correction(word) for word in words]
            swords=set(words)
            scorrected_words=set(corrected_words)
            mistakes=(len(list(swords-scorrected_words)))
            deduced_marks=mistakes/5
            corrected_text = ' '.join(corrected_words)
            return corrected_text
            
       #answer sheeet
        def preprocessing(text1):
            text = text1            
            # tokenization
            words = word_tokenize(text)
            #removing stop words
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word.lower() not in stop_words]
            # Join the filtered words back into a sentence
            filtered_text = ' '.join(filtered_words)
            #stemming
            stemmer = PorterStemmer()
            stemmed_words = [stemmer.stem(word) for word in filtered_text]
            stemmed_text = ''.join(stemmed_words)

            #question key sheet
            text = text2
            # tokenization
            words = word_tokenize(text)
            #removing stop words
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word.lower() not in stop_words]
            # Join the filtered words back into a sentence
            filtered_text1 = ' '.join(filtered_words)
            #stemming
            stemmer = PorterStemmer()
            stemmed_words = [stemmer.stem(word) for word in filtered_text1]
            stemmed_text1 = ''.join(stemmed_words)
            return stemmed_text1

      # You can also use a weighted average if desired
        #ngram_similarity = calculate_jaccard_similarity(stemmed_text, stemmed_text1, n=1)
        def split_to_dict(text):
    
            pattern = r'\(\d+\)'  # Match patterns like (1), (2), etc.
            matches = re.findall(pattern, text)            
            question_dict = {}
            for i in range(len(matches)):
                key = matches[i]
                if key in text:
                    if i < len(matches) - 1:
                        value = text.split(matches[i], 1)[1].split('**')[0].strip()
                    else:
                        value = text.split(matches[i], 1)[1].strip()
                    question_dict[key] = value
            return question_dict          
         
        dict1 = split_to_dict(text1)
        dict2 = split_to_dict(text2)
        def calculate_bleu_score(reference, candidate):
            weights = (1.0,)  # Use unigram (1-gram) weights
            smoothing_function = SmoothingFunction().method1
            return sentence_bleu([reference.split()], candidate.split(), weights=weights, smoothing_function=smoothing_function)
        similarity_sco = []
        for key in dict1.keys():
            if key in dict2:
                #spell_answer = spellchecker(str(dict2[key]))
                stemmed_key=preprocessing(dict1[key])
                stemmed_answer=preprocessing(dict2[key])
                similarity_score = calculate_bleu_score(stemmed_key, stemmed_answer)
                similarity_sco.append(similarity_score*100)
        similarity_sco_str = ""
        for key, score in enumerate(similarity_sco):
            similarity_sco_str += f"{key + 1}que: {score}, "

        text11=preprocessing(text1)
        text21=preprocessing(text2)
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        inputs = tokenizer(text11, text21, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        similarity_scores = torch.softmax(logits, dim=1)
        score_for_text1 = similarity_scores[0][1].item()

        similarity_sco_str = similarity_sco_str[:-2]

   
                
        ngram_similarity = calculate_jaccard_similarity(text1, text2, n=1)

        
        return render_template('res.html', text1=text1, text2=text2,d1=dict1,d2=dict2,stemmed_answer=stemmed_answer,stemmed_key=stemmed_key , n_gram=ngram_similarity , similarity_sco=similarity_sco_str ,bert= score_for_text1, keywords=answer_score)

if __name__ == '__main__':
    app.run(debug=True)
