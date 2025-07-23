import joblib
import os
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

dirname = os.path.dirname(__file__)

# model = joblib.load(dirname+'/../output/spamassassin/logistic-regression/scoring_accuracy/20250623-0200/model.pkl')

# messages = [
#     "Congratulations! Thanks to a good friend U have WON the �2,000 Xmas prize. 2 claim is easy, just call 08718726978 NOW! Only 10p per minute. BT-national-rate",
#     "I will send them to your email. Do you mind  &lt;#&gt;  times per night?",
#     "44 7732584351, Do you want a New Nokia 3510i colour phone DeliveredTomorrow? With 300 free minutes to any mobile + 100 free texts + Free Camcorder reply or call 08000930705.",
#     "tap & spile at seven. * Is that pub on gas st off broad st by canal. Ok?",
#     "Ok then i come n pick u at engin?",
#     "You have 1 new voicemail. Please call 08719181513.",
#     "MOON has come to color your dreams, STARS to make them musical and my SMS to give you warm and Peaceful Sleep. Good Night",
#     "Just finished eating. Got u a plate. NOT leftovers this time.",
#     "Thanx a lot...",
#     "Hurry home u big butt. Hang up on your last caller if u have to. Food is done and I'm starving. Don't ask what I cooked.",
#     "Lol your right. What diet? Everyday I cheat anyway. I'm meant to be a fatty :(",
# ]
# # T, F, T, F, F, F, T, F, T, F, F (4/11 = 36%)


# predictions = model.predict(messages)

# print([predictions])

def booleanizerTaget(docs):
    final = []
    for doc in docs:
        if(doc=='spam' or doc=='1' or doc==1):
            final.append(1)
        else:
            final.append(0)
    return (final)

# with open(dirname+'/../output/spamassassin/logistic-regression/scoring_accuracy/20250707-1429/model.pkl', 'rb') as file:
#     model = pickle.load(file)

# model = joblib.load(dirname+'/../output/spam/logistic-regression/scoring_accuracy/20250706-1630/model.pkl')
# model = joblib.load(dirname+'/../output/spam_ham_dataset/logistic-regression/scoring_accuracy/20250706-1740/model.pkl')
# model = joblib.load(dirname+'/../output/spamassassin/logistic-regression/scoring_accuracy/20250707-2335/model.pkl')

# model = joblib.load(dirname+'/../output/spam/naive-bayes/scoring_accuracy/20250706-1625/model.pkl')
# model = joblib.load(dirname+'/../output/spam_ham_dataset/naive-bayes/scoring_accuracy/20250706-1735/model.pkl')
model = joblib.load(dirname+'/../output/spamassassin/naive-bayes/scoring_accuracy/20250707-2338/model.pkl')

# df = pd.read_csv(dirname+'/../input/spam.csv')
df = pd.read_csv(dirname+'/../input/spam_ham_dataset.csv')
# df = pd.read_csv(dirname+'/../input/spamassassin.csv')

X = df['text']
y = df['label']

y = booleanizerTaget(y)

y_pred = model.predict(X)

print("Acurácia:", accuracy_score(y, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y, y_pred))

# Matriz de confusão
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
print("\n")
print(f"MATRIZ DE CONFUSÃO:")
print("-"*40)
print(f"Verdadeiros Negativos: {tn}")
print(f"Falsos Positivos: {fp}")
print(f"Falsos Negativos: {fn}")
print(f"Verdadeiros Positivos: {tp}")