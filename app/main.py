from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.linear_model import LinearRegression
from word2number import w2n

app = Flask(__name__)
with open('models/cv_best_nb.pkl', 'rb') as f:
    vect, clf = pickle.load(f)
    
with open('models/prediction_model.pkl', 'rb') as f1:
    prediction_clf = pickle.load(f1)
    
def str_to_num(s):
    s_return = []
    for w in s:
        try:
            w = w2n.word_to_num(w)
        except:
            pass
        s_return.append(str(w))
    return s_return

def clean_sentence(val):
    val = val[0].replace('\n', ' ').replace('--', ' ').replace('\r', ' ')
    sentence = val.split(" ")
    sentence = str_to_num(sentence)
    sent = []
    for i, s in enumerate(sentence):
        t_sent = []
        if '$' in s:
            dollar_split = s.split('$')[1]
            t_sent.append(str(dollar_split))
            t_sent.append('dollars')
        else:
            t_sent.append(s.lower())
        sent.append(" ".join(t_sent))
    sentence = " ".join(sent)
    return sentence
    


@app.route('/')
def home():
    return render_template('/fom_index.html')

@app.route('/predict',methods=['POST'])
def predict():
    text = request.form.getlist('text_input')
    clean_text = clean_sentence(text)
    x = vect.transform([str(text)])
    lang_pred = clf.predict(x)
    if lang_pred != 0:
        pred = 'This Submission Is Not English'
    else:
        pred = prediction_clf.predict(text)
        prob = prediction_clf.predict_proba(text)
        if pred[0] == 1:
            pred = 'This enslaved person is on the run, with %{:.2f} probability' .format(prob[0][1] * 100)
        else:
            pred = 'This enslaved person is incarcerated, with %{:.2f} probability' .format(prob[0][0] * 100)
    return render_template('/prediction.html', prediction_text='{}'.format(str(text[0])),
                           prediction = '{}'.format(pred))

if __name__ == "__main__":
    app.run(debug=True)