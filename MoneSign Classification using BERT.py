from flask import Flask, request, jsonify
import torch
import random
import numpy as np
import logging
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import re

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Configuration
SEED_VAL = 42
MODEL_PATH = "C:\\Users\\Chirag Sharma\\Desktop\\BERT MODEL MONEY_SIGN\\Bert Configurations"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the seed value for reproducibility.
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)

# Load Model and Tokenizer
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# Fit Label Encoder with training data labels
train_file_path = "C:\\Users\\Chirag Sharma\\Desktop\\BERT MODEL MONEY_SIGN\\Train & Test Data for BERT Model\\predictions _test  - Sheet1.csv"
trainData = pd.read_csv(train_file_path)
labelEncoder = LabelEncoder()
labelEncoder.fit(trainData['MoneySign'])

# Define MoneySign to ID mapping
moneysign_to_id = {
    "Persistent Horse": 1,
    "Tactical Tiger": 2,
    "Opportunistic Lion": 3,
    "Virtuous Elephant": 4,
    "Far-sighted Eagle": 5,
    "Vigilant Turtle": 6,
    "Enlightened Whale": 7,
    "Stealthy Shark": 8
}

# Reordering Function
def reorder_options(input_text, predefined_list):
    # Create a mapping from the predefined list to an index for sorting
    predefined_map = {item: index for index, item in enumerate(predefined_list)} #each option in predefined_list is paired with its index. This mapping helps us know the correct order to sort the options.

    # Replace long options with placeholders to avoid issues during reordering
    input_text = input_text.replace("I invest money in the stock market for at least 5 years", "INVEST_STOCK_MARKET")
    input_text = input_text.replace("I invest in FDs, RDs, debt funds only for 3 years or less", "INVEST_FD_RD_DEBT_FUNDS")

    def reorder_section(section):
        # Split options by commas and strip any extra whitespace
        options = [opt.strip() for opt in section.split(',')] #Splits the section into individual options.
        # Sort options according to the predefined map order
        #Sorts these options based on their position in predefined_list, using the predefined_map to determine the correct order.
        ordered_options = sorted(options, key=lambda x: predefined_map.get(x, float('inf')))
        return ', '.join(ordered_options)

    # Define a pattern to extract sections and markers
    pattern = re.compile(r'(\d+\.?\d*\s+[^\d]+?(?=\s*\d|$))', re.DOTALL)
    sections = pattern.findall(input_text)
    
    reordered_sections = []
    for section in sections:
        # Match the marker and the text
        #(\d+\.?\d*\s+): Captures the number and the whitespace after it as the marker.
        #(.*): Captures the remaining part of the section as the text.
        marker_match = re.match(r'(\d+\.?\d*\s+)(.*)', section.strip(), re.DOTALL)  #Holds the match object if the section matches the regex. Otherwise, it's None
        if marker_match:
            marker = marker_match.group(1)  # The marker (e.g., "123.45 ")
            text = marker_match.group(2).strip() # The remaining text (e.g., "apples, oranges")

            if ',' in text:   #Checks if the text contains a comma (,), indicating there are multiple items.
                # Reorder only if there are multiple options
                text = reorder_section(text)

            #Combines the marker and the (possibly reordered) text into a single string and appends it to the reordered_sections list.    
            reordered_sections.append(f"{marker}{text}")
        else:
            #If marker_match fails (i.e., the section doesn’t match the marker-text pattern), the original section (stripped of whitespace) is added to the reordered_sections list as-is.
            reordered_sections.append(section.strip())
    
    # Join all reordered sections together
    final_text = ' '.join(reordered_sections)

    # Clean up formatting issues
    final_text = re.sub(r'\s+', ' ', final_text) # Replace multiple spaces with one space
    final_text = re.sub(r' ,', ',', final_text) # Remove spaces before commas
    final_text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', final_text) # Fix number formatting

    # Replace placeholders with the original text
    final_text = final_text.replace("INVEST_STOCK_MARKET", "I invest money in the stock market for at least 5 years")
    final_text = final_text.replace("INVEST_FD_RD_DEBT_FUNDS", "I invest in FDs, RDs, debt funds only for 3 years or less")

    return final_text.strip()

# Function to convert floats to integers in the input text
def Convert2Int(text):
    words = text.split()
    for i in range(len(words)):
        try:
            num = float(words[i])  #this convert number in to float
            if num == int(num):  #Checks whether the floating-point number (num) is equivalent to its integer version (int(num)).
                words[i] = str(int(num)) #it converts 5.0 into 5 and 3.2 as it is means 3.2 and Converts the integer back into a string (str(int(num))) so it can replace the original word in the words list.
        except ValueError:
            pass
    return " ".join(words)

# Prediction Function
def predict_money_sign(model, tokenizer, label_encoder, input_text, device):
    encoded_dict = tokenizer.encode_plus(
        input_text,
        ##Adds special tokens used by BERT:
        #[CLS] (classification token): Marks the beginning of the input.
        #[SEP] (separator token): Marks the end of the input.
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,  #If it’s shorter, it will be padded (fill). Ensures that all inputs are padded to the same length
        return_attention_mask=True, #Returns an attention mask, which tells the model which tokens are actual input and which are padding.
        return_tensors='pt',  #Converts the output into PyTorch tensors, which are needed for the BERT model.
        truncation=True, #If the tokenized input exceeds max_length, it will be truncated to fit. Prevents errors when input text is too long.

    )
    
    #These lines move the encoded input data (input_ids and attention_mask) to the correct device (GPU or CPU) for computation.
    input_ids = encoded_dict['input_ids'].to(device)   #[101, 1045, 7592, 1999, 3899, 102, 0, 0]
    attention_mask = encoded_dict['attention_mask'].to(device)  #[1, 1, 1, 1, 1, 1, 0, 0] 
    
    #This disables gradient calculations because we’re just making predictions (not training the model).
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits  #Extracts the raw model output (logits).
    pred = torch.argmax(logits, dim=1).cpu().numpy() #selects the class with the highest score (the predicted class). ensures the result is moved to the CPU (if it was on GPU).
    decoded_pred = label_encoder.inverse_transform(pred)[0] #Converts the numerical label back to its original text form (e.g., "Persistent Horse", "Tactical Tiger", etc.).
    predicted_id = moneysign_to_id.get(decoded_pred, -1) #Maps the decoded prediction (e.g., "Persistent Horse") to its corresponding ID (e.g., 1) using the moneysign_to_id dictionary.
                                                         #If the label is not found, it returns -1 as a default value.
    return decoded_pred, predicted_id


#This line is a route decorator. It tells Flask (the web framework) that whenever a POST request is sent to the URL path /predict, the predict() function will be executed.
#POST means the client is sending data to the server (usually in the body of the request).
@app.route('/predict', methods=['POST'])
def predict(): #This defines the function predict(), which is the code that will be executed when a POST request is received at the /predict route.
    data = request.json   #the request.json is a Flask object that automatically parses the JSON data from the body of the request, making it accessible in the data variable as a Python dictionary.
    input_text = data.get('input_text')

    if not input_text:
        return jsonify({"error": "Input text is required"}), 400

    # Apply Reordering
    predefined_list = [
        "Crypto",
        "Gold Exchange-Traded Funds (ETFs)",
        "Real Estate Investment Trusts (REITs)",
        "Non-Fungible Tokens (NFTs)",
        "Asset leasing",
        
        "International / Overseas Funds",
        "Revenue-based financing",
        "Art, paintings or antiques",
        "Invoice discounting",
        "Revenue-based financing",
        "International / Overseas Funds",
        
        "Asset leasing",
        "Non-Fungible Tokens (NFTs)",
        "Invoice discounting",
        "Art, paintings or antiques",
        "Salary / business income",
        "Equity dividends",
        "Stock / derivative trading",
        "Equity capital appreciation",
        "Income from secondary business(es)",
        "Real estate rental",
        "Others",
        "Debt interest",
        "Real estate price appreciation",
        "I prefer to avoid personal loans",
        "If I spend more in a particular month, I spend less the next month",
        "I add nominees wherever applicable",
        "I invest money in the stock market for at least 5 years",
        "I don't take loans for buying electronics or vehicles",
        "I bought term insurance to cover my dependents life expenses",
        "I invest in FDs, RDs, debt funds only for 3 years or less",
        "I use apps to track all my investments in one place",
        "The credit card benefits I earn are more than the fees I pay",
        "I prefer a home loan or a gold loan",
        "I Invest a fixed amount every month",
        "I increase my savings when my income increases",
        "I track my savings",
        "I always add nominees",
        "I avoid EMIs while making purchases",
        "I keep monthly expenses below a certain limit",
        "I get my vehicle serviced annually",
        "I buy health insurance every year",
        "I track my loan rate(s)",
        "I plan my taxes at the start of the financial year"
    ]
    input_text = reorder_options(input_text, predefined_list)
    
    # Convert numbers where needed
    input_text = Convert2Int(input_text)

    logging.debug(f"Received input text for prediction: {input_text}")

    # Predict MoneySign
    predicted_ms, predicted_id = predict_money_sign(model, tokenizer, labelEncoder, input_text, DEVICE)

    logging.debug(f"Predicted MoneySign: {predicted_ms}, Predicted ID: {predicted_id}")

    return jsonify({
        "predicted_money_sign": predicted_ms,
        "predicted_money_sign_id": int(predicted_id)
    })

if __name__ == '__main__':
    app.run(debug=True)
