import pandas as pd
from string import punctuation

emails = pd.read_csv('/Users/Bobev/Desktop/Python/Assignment2_contacts(1).csv')

emails.Email.notnull().sum()

#print(emails.Email.notnull().sum())

##  6 total email patterns
#First initial, lastname :ebobev
#Last initial, first name :bemil
#last name,first imitial: bobeve
#first name, last name: emil.bobev
#first name
#last name

def get_punct(email_prefix):
    punct_return = None
    
    for punct in punctuation:
        if punct in email_prefix:
            punct_return = punct
            break
        
    return punct_return

email_prefix = 'emil.bobev'

get_punct(email_prefix)

def pattern_match(first, last, email):
    first = 'Emil'.lower()
    last = 'Bobev'.lower()
    email_prefix = email.split.lower('@')[0]
    return email_prefix

    if punct:
        if first == email_prefix[:len(first)] and last == email_prefix[-len(last):]:
            return {'pattern_match': 'first_name_last_name','punct': punct}
pattern_match ('Emil','Bobev', 'emil.bobev@stockton.edu')

