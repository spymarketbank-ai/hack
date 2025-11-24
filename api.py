from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import csv
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)


questions_qcm = [
{"id": 1, "question": "Quels types sont mutables en Python ?", "options": ["list", "tuple", "dict", "set", "str"], "answers": [0, 2, 3]},
{"id": 2, "question": "Quels algorithmes sont supervisés ?", "options": ["KMeans", "Régression logistique", "SVM", "Random Forest", "PCA"], "answers": [1, 2, 3]},
{"id": 3, "question": "Quels types de boucles existent en Python ?", "options": ["for", "while", "repeat", "loop", "do-while"], "answers": [0, 1]},
{"id": 4, "question": "Quels sont des frameworks Python pour le Machine Learning ?", "options": ["scikit-learn", "Django", "TensorFlow", "Flask", "PyTorch"], "answers": [0, 2, 4]},
{"id": 5, "question": "Quelles structures de données en Python ne supportent pas les doublons ?", "options": ["list", "set", "dict keys", "tuple", "str"], "answers": [1, 2]},
{"id": 6, "question": "Quels sont des modèles de classification ?", "options": ["Régression logistique", "SVM", "Random Forest", "ACP", "KMeans"], "answers": [0, 1, 2]},
{"id": 7, "question": "Quels sont des types immuables en Python ?", "options": ["tuple", "str", "list", "dict", "set"], "answers": [0, 1]},
{"id": 8, "question": "Quels sont des algorithmes non supervisés ?", "options": ["KMeans", "DBSCAN", "ACP", "Régression logistique", "Random Forest"], "answers": [0, 1, 2]},
{"id": 9, "question": "Quels sont des IDE utilisés pour Python ?", "options": ["PyCharm", "Jupyter Notebook", "RStudio", "VS Code", "Spyder"], "answers": [0, 1, 3, 4]},
{"id": 10, "question": "Quels sont des types de variables en statistiques ?", "options": ["Nominale", "Ordinale", "Continue", "Discrète", "SQL"], "answers": [0, 1, 2, 3]},
{"id": 11, "question": "Quels packages Python servent à la visualisation ?", "options": ["matplotlib", "pandas", "seaborn", "ggplot2", "plotly"], "answers": [0, 2, 4]},
{"id": 12, "question": "Quels sont des types de jointures SQL ?", "options": ["INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL OUTER JOIN", "CROSS JOIN"], "answers": [0, 1, 2, 3, 4]},
{"id": 13, "question": "Quels sont des algorithmes de réduction de dimension ?", "options": ["ACP", "t-SNE", "Régression linéaire", "UMAP", "Random Forest"], "answers": [0, 1, 3]},
{"id": 14, "question": "Quels sont des types de normalisation ?", "options": ["Min-Max", "Z-score", "One-hot encoding", "Robust Scaler", "StandardScaler"], "answers": [0, 1, 3, 4]},
{"id": 15, "question": "Quels sont des formats de fichiers de données ?", "options": ["CSV", "JSON", "XML", "Excel", "PNG"], "answers": [0, 1, 2, 3]},
{"id": 16, "question": "Quels sont des types d’apprentissage automatique ?", "options": ["Supervisé", "Non supervisé", "Semi-supervisé", "Apprentissage par renforcement", "Apprentissage profond"], "answers": [0, 1, 2, 3, 4]},
{"id": 17, "question": "Quels sont des systèmes de gestion de bases de données ?", "options": ["MySQL", "PostgreSQL", "MongoDB", "Oracle", "Hadoop"], "answers": [0, 1, 2, 3]},
{"id": 18, "question": "Quels sont des indicateurs de performance en Machine Learning ?", "options": ["Précision", "Recall", "F1-score", "RMSE", "AUC"], "answers": [0, 1, 2, 3, 4]},
{"id": 19, "question": "Quels sont des types de clés en SQL ?", "options": ["Clé primaire", "Clé étrangère", "Clé candidate", "Clé composée", "Clé de chiffrement"], "answers": [0, 1, 2, 3]},
{"id": 20, "question": "Quels sont des outils de gestion de versions ?", "options": ["Git", "Mercurial", "SVN", "Docker", "Perforce"], "answers": [0, 1, 2, 4]},
{"id": 21, "question": "Quels sont des types de réseaux de neurones ?", "options": ["Perceptron", "CNN", "RNN", "GAN", "SQLNet"], "answers": [0, 1, 2, 3]},
{"id": 22, "question": "Quels sont des types de données en Python ?", "options": ["int", "float", "str", "bool", "decimal"], "answers": [0, 1, 2, 3]},
{"id": 23, "question": "Quels sont des types de fichiers compressés ?", "options": ["zip", "tar", "gzip", "rar", "7z"], "answers": [0, 1, 2, 3, 4]},
{"id": 24, "question": "Quels sont des algorithmes d’ensemble ?", "options": ["Random Forest", "Bagging", "Boosting", "Stacking", "SVM"], "answers": [0, 1, 2, 3]},
{"id": 25, "question": "Quels sont des types de modèles en statistiques ?", "options": ["Linéaire", "Logistique", "ANOVA", "Arbres de décision", "CNN"], "answers": [0, 1, 2, 3]},
{"id": 26, "question": "Quels sont des systèmes d’exploitation ?", "options": ["Linux", "Windows", "macOS", "iOS", "Android"], "answers": [0, 1, 2, 3, 4]},
{"id": 27, "question": "Quels sont des langages de programmation ?", "options": ["Python", "R", "Java", "SQL", "HTML"], "answers": [0, 1, 2, 3]},
{"id": 28, "question": "Quels sont des types de variables Python ?", "options": ["globales", "locales", "libres", "environnementales", "statiques"], "answers": [0, 1, 2, 4]},
{"id": 29, "question": "Quels sont des systèmes de Big Data ?", "options": ["Hadoop", "Spark", "Flink", "Storm", "MySQL"], "answers": [0, 1, 2, 3]},
{"id": 30, "question": "Quels sont des algorithmes de clustering ?", "options": ["KMeans", "DBSCAN", "Agglomerative", "Mean Shift", "PCA"], "answers": [0, 1, 2, 3]},
{"id": 31, "question": "Quels sont des formats de bases de données NoSQL ?", "options": ["Clé-Valeur", "Colonnes", "Graphes", "Documents", "Réseaux"], "answers": [0, 1, 2, 3]},
{"id": 32, "question": "Quels sont des packages de manipulation de données en Python ?", "options": ["pandas", "numpy", "scikit-learn", "matplotlib", "seaborn"], "answers": [0, 1]},
{"id": 33, "question": "Quels sont des avantages de Git ?", "options": ["Suivi des versions", "Collaboration", "Branches", "Centralisation obligatoire", "Merge"], "answers": [0, 1, 2, 4]},
{"id": 34, "question": "Quels sont des algorithmes de tri ?", "options": ["Tri à bulles", "Tri fusion", "Tri rapide", "Tri par insertion", "Random Forest"], "answers": [0, 1, 2, 3]},
{"id": 35, "question": "Quels sont des types d’erreurs en Python ?", "options": ["SyntaxError", "IndentationError", "TypeError", "ValueError", "KeyError"], "answers": [0, 1, 2, 3, 4]},
{"id": 36, "question": "Quels sont des OS basés sur Linux ?", "options": ["Ubuntu", "Debian", "Fedora", "CentOS", "Windows"], "answers": [0, 1, 2, 3]},
{"id": 37, "question": "Quels sont des outils de visualisation de données ?", "options": ["Tableau", "Power BI", "Excel", "QlikView", "Stata"], "answers": [0, 1, 2, 3]},
{"id": 38, "question": "Quels sont des services cloud ?", "options": ["AWS", "Azure", "GCP", "Heroku", "GitHub"], "answers": [0, 1, 2, 3]},
{"id": 39, "question": "Quels sont des structures de contrôle en Python ?", "options": ["if", "for", "while", "switch", "try-except"], "answers": [0, 1, 2, 4]},
{"id": 40, "question": "Quels sont des algorithmes de boosting ?", "options": ["AdaBoost", "Gradient Boosting", "XGBoost", "LightGBM", "Random Forest"], "answers": [0, 1, 2, 3]},
{"id": 41, "question": "Quels sont des méthodes de régularisation ?", "options": ["L1", "L2", "ElasticNet", "Dropout", "BatchNorm"], "answers": [0, 1, 2]},
{"id": 42, "question": "Quels sont des types de mémoire en informatique ?", "options": ["RAM", "ROM", "Cache", "Disque dur", "CPU"], "answers": [0, 1, 2, 3]},
{"id": 43, "question": "Quels sont des types de visualisation ?", "options": ["Histogramme", "Nuage de points", "Courbe", "Camembert", "Heatmap"], "answers": [0, 1, 2, 3, 4]},
{"id": 44, "question": "Quels sont des types de réseaux en informatique ?", "options": ["LAN", "WAN", "MAN", "VPN", "HTTP"], "answers": [0, 1, 2, 3]},
{"id": 45, "question": "Quels sont des structures de données hiérarchiques ?", "options": ["Arbre", "Heap", "Trie", "Graphe", "Liste"], "answers": [0, 1, 2]},
{"id": 46, "question": "Quels sont des types de supervision en ML ?", "options": ["Supervisé", "Non supervisé", "Semi-supervisé", "Renforcement", "Auto-supervisé"], "answers": [0, 1, 2, 3, 4]},
{"id": 47, "question": "Quels sont des exemples de systèmes NoSQL ?", "options": ["MongoDB", "Cassandra", "Redis", "Neo4j", "PostgreSQL"], "answers": [0, 1, 2, 3]},
{"id": 48, "question": "Quels sont des outils de conteneurisation ?", "options": ["Docker", "Kubernetes", "Podman", "LXC", "VMware"], "answers": [0, 1, 2, 3]},
{"id": 49, "question": "Quels sont des protocoles de communication ?", "options": ["HTTP", "FTP", "SMTP", "SSH", "SQL"], "answers": [0, 1, 2, 3]},
{"id": 50, "question": "Quels sont des packages Python pour le deep learning ?", "options": ["TensorFlow", "PyTorch", "Keras", "MXNet", "FastAI"], "answers": [0, 1, 2, 3, 4]},
{"id": 51, "question": "Quels sont des outils de versionning collaboratif ?", "options": ["GitHub", "GitLab", "Bitbucket", "Azure DevOps", "Jira"], "answers": [0, 1, 2, 3]},
{"id": 52, "question": "Quels sont des tests en Machine Learning ?", "options": ["Train-test split", "Validation croisée", "Bootstrap", "A/B testing", "Dropout"], "answers": [0, 1, 2, 3]},
{"id": 53, "question": "Quels sont des métriques pour régression ?", "options": ["MSE", "RMSE", "MAE", "R2", "F1-score"], "answers": [0, 1, 2, 3]},
{"id": 54, "question": "Quels sont des algorithmes gourmands ?", "options": ["Prim", "Kruskal", "Dijkstra", "A*", "Random Forest"], "answers": [0, 1, 2, 3]},
{"id": 55, "question": "Quels sont des types d’apprentissage profond ?", "options": ["CNN", "RNN", "GAN", "Autoencodeurs", "SVM"], "answers": [0, 1, 2, 3]},
{"id": 56, "question": "Quels sont des modules standards Python ?", "options": ["os", "sys", "math", "random", "pandas"], "answers": [0, 1, 2, 3]},
{"id": 57, "question": "Quels sont des outils ETL ?", "options": ["Talend", "Informatica", "SSIS", "Airflow", "Excel"], "answers": [0, 1, 2, 3]},
{"id": 58, "question": "Quels sont des langages pour le Big Data ?", "options": ["Scala", "Java", "Python", "R", "HTML"], "answers": [0, 1, 2, 3]},
{"id": 59, "question": "Quels sont des étapes d’un projet Data Science ?", "options": ["Collecte de données", "Nettoyage", "Feature engineering", "Modélisation", "Déploiement"], "answers": [0, 1, 2, 3, 4]},
{"id": 60, "question": "Quels sont des avantages du cloud computing ?", "options": ["Scalabilité", "Flexibilité", "Paiement à l’usage", "Sécurité", "Propriétaire obligatoire"], "answers": [0, 1, 2, 3]}
]

# Partie 2 : Questions de Code


questions_code = [
    {"id": 101, "question": "Écris une fonction `est_pair(n)` qui retourne True si `n` est pair.", 
     "test_code": "assert est_pair(2) == True\nassert est_pair(3) == False"},
    
    {"id": 102, "question": "Écris une fonction `carre(x)` qui retourne le carré d’un nombre.", 
     "test_code": "assert carre(2) == 4\nassert carre(-3) == 9"},
    
    {"id": 103, "question": "Écris une fonction `inverse(chaine)` qui retourne la chaîne inversée.", 
     "test_code": "assert inverse('abc') == 'cba'\nassert inverse('Python') == 'nohtyP'"},
    
    {"id": 104, "question": "Écris une fonction `factorielle(n)` qui calcule la factorielle d’un nombre.", 
     "test_code": "assert factorielle(0) == 1\nassert factorielle(5) == 120"},
    
    {"id": 105, "question": "Écris une fonction `fibonacci(n)` qui retourne la suite de Fibonacci jusqu’à n termes.", 
     "test_code": "assert fibonacci(5) == [0,1,1,2,3]\nassert fibonacci(1) == [0]"},
    
    {"id": 106, "question": "Écris une fonction `pgcd(a, b)` qui calcule le plus grand commun diviseur.", 
     "test_code": "assert pgcd(12, 18) == 6\nassert pgcd(7, 5) == 1"},
    
    {"id": 107, "question": "Écris une fonction `palindrome(mot)` qui retourne True si un mot est un palindrome.", 
     "test_code": "assert palindrome('radar') == True\nassert palindrome('python') == False"},
    
    {"id": 108, "question": "Écris une fonction `min_max(liste)` qui retourne le minimum et le maximum d’une liste.", 
     "test_code": "assert min_max([3,1,7]) == (1,7)\nassert min_max([10]) == (10,10)"},
    
    {"id": 109, "question": "Écris une fonction `voyelles(chaine)` qui compte le nombre de voyelles.", 
     "test_code": "assert voyelles('hello') == 2\nassert voyelles('xyz') == 0"},
    
    {"id": 110, "question": "Écris une fonction `moyenne(liste)` qui retourne la moyenne d’une liste de nombres.", 
     "test_code": "assert moyenne([1,2,3]) == 2\nassert moyenne([10,20]) == 15"},
    
    # --- ML & Data ---
    {"id": 111, "question": "Écris un code qui sépare un dataset sklearn en train/test avec train_test_split.", 
     "test_code": "from sklearn.model_selection import train_test_split\nX=[1,2,3,4];y=[0,1,0,1]\nX_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)"},
    
    {"id": 112, "question": "Écris un code qui entraîne une régression linéaire avec scikit-learn.", 
     "test_code": "from sklearn.linear_model import LinearRegression\nmodel=LinearRegression().fit([[1],[2],[3]],[2,4,6])"},
    
    {"id": 113, "question": "Écris un code qui entraîne une régression logistique avec scikit-learn.", 
     "test_code": "from sklearn.linear_model import LogisticRegression\nclf=LogisticRegression().fit([[0],[1],[2],[3]],[0,0,1,1])"},
    
    {"id": 114, "question": "Écris un code qui applique KMeans avec sklearn et trouve 2 clusters.", 
     "test_code": "from sklearn.cluster import KMeans\nkmeans=KMeans(n_clusters=2,n_init=10).fit([[1],[2],[10],[11]])"},
    
    {"id": 115, "question": "Écris un code qui applique PCA et affiche la variance expliquée.", 
     "test_code": "from sklearn.decomposition import PCA\nimport numpy as np\nX=np.random.rand(10,5)\npca=PCA(n_components=2).fit(X)\nassert hasattr(pca,'explained_variance_ratio_')"},
    
    {"id": 116, "question": "Écris un code qui entraîne un arbre de décision.", 
     "test_code": "from sklearn.tree import DecisionTreeClassifier\nclf=DecisionTreeClassifier().fit([[0],[1]],[0,1])"},
    
    {"id": 117, "question": "Écris un code qui calcule une matrice de confusion.", 
     "test_code": "from sklearn.metrics import confusion_matrix\ncm=confusion_matrix([0,1,1],[0,0,1])"},
    
    {"id": 118, "question": "Écris un code qui entraîne un SVM pour classer 2 points.", 
     "test_code": "from sklearn.svm import SVC\nclf=SVC().fit([[0],[1]],[0,1])"},
    
    {"id": 119, "question": "Écris un code qui entraîne un RandomForestClassifier.", 
     "test_code": "from sklearn.ensemble import RandomForestClassifier\nclf=RandomForestClassifier().fit([[0],[1]],[0,1])"},
    
    {"id": 120, "question": "Écris un code qui applique StandardScaler à un dataset.", 
     "test_code": "from sklearn.preprocessing import StandardScaler\nscaler=StandardScaler().fit([[1],[2],[3]])"},
    
    # --- Fonctions Python (suite jusqu’à 40) ---
    {"id": 121, "question": "Écris une fonction `compter_mots(texte)` qui compte le nombre de mots.", 
     "test_code": "assert compter_mots('bonjour le monde') == 3"},
    
    {"id": 122, "question": "Écris une fonction `est_anagramme(m1,m2)` qui vérifie si deux mots sont des anagrammes.", 
     "test_code": "assert est_anagramme('chien','niche') == True\nassert est_anagramme('python','java') == False"},
    
    {"id": 123, "question": "Écris une fonction `tri_bulle(liste)` qui trie une liste avec l’algorithme du tri à bulle.", 
     "test_code": "assert tri_bulle([3,2,1]) == [1,2,3]"},
    
    {"id": 124, "question": "Écris une fonction `recherche_binaire(liste,x)` qui retourne l’indice de x ou -1.", 
     "test_code": "assert recherche_binaire([1,2,3,4],3) == 2\nassert recherche_binaire([1,2,3],5) == -1"},
    
    {"id": 125, "question": "Écris une fonction `compter_occurrences(liste,x)` qui compte combien de fois x apparaît.", 
     "test_code": "assert compter_occurrences([1,2,2,3],2) == 2"},
    
    {"id": 126, "question": "Écris une fonction `produit_liste(liste)` qui retourne le produit des éléments.", 
     "test_code": "assert produit_liste([1,2,3]) == 6"},
    
    {"id": 127, "question": "Écris une fonction `fusion(l1,l2)` qui fusionne deux listes triées.", 
     "test_code": "assert fusion([1,3,5],[2,4]) == [1,2,3,4,5]"},
    
    {"id": 128, "question": "Écris une fonction `flatten(liste)` qui aplatit une liste de listes.", 
     "test_code": "assert flatten([[1,2],[3,4]]) == [1,2,3,4]"},
    
    {"id": 129, "question": "Écris une fonction `est_armstrong(n)` qui vérifie si n est un nombre d’Armstrong.", 
     "test_code": "assert est_armstrong(153) == True\nassert est_armstrong(123) == False"},
    
    {"id": 130, "question": "Écris une fonction `premiers(n)` qui retourne les n premiers nombres premiers.", 
     "test_code": "assert premiers(5) == [2,3,5,7,11]"},
    
    {"id": 131, "question": "Écris une fonction `mot_le_plus_long(phrase)` qui retourne le mot le plus long.", 
     "test_code": "assert mot_le_plus_long('le python avance') == 'avance'"},
    
    {"id": 132, "question": "Écris une fonction `majuscule(texte)` qui met en majuscules toutes les lettres.", 
     "test_code": "assert majuscule('abc') == 'ABC'"},
    
    {"id": 133, "question": "Écris une fonction `intersection(l1,l2)` qui retourne les éléments communs.", 
     "test_code": "assert intersection([1,2,3],[2,3,4]) == [2,3]"},
    
    {"id": 134, "question": "Écris une fonction `unique(liste)` qui retourne la liste sans doublons.", 
     "test_code": "assert unique([1,2,2,3]) == [1,2,3]"},
    
    {"id": 135, "question": "Écris une fonction `nb_pairs(liste)` qui compte les nombres pairs.", 
     "test_code": "assert nb_pairs([1,2,3,4]) == 2"},
    
    {"id": 136, "question": "Écris une fonction `somme_carres(n)` qui calcule la somme des carrés jusqu’à n.", 
     "test_code": "assert somme_carres(3) == 14"},
    
    {"id": 137, "question": "Écris une fonction `max_2d(matrice)` qui retourne le max d’une matrice 2D.", 
     "test_code": "assert max_2d([[1,2],[3,4]]) == 4"},
    
    {"id": 138, "question": "Écris une fonction `transpose(matrice)` qui retourne la transposée.", 
     "test_code": "assert transpose([[1,2],[3,4]]) == [[1,3],[2,4]]"},
    
    {"id": 139, "question": "Écris une fonction `occurences_caracteres(texte)` qui compte chaque caractère.", 
     "test_code": "assert occurences_caracteres('aba')['a'] == 2"},
    
    {"id": 140, "question": "Écris une fonction `somme_diagonale(matrice)` qui calcule la somme diagonale.", 
     "test_code": "assert somme_diagonale([[1,2],[3,4]]) == 5"},
]




# --------------------------
# Endpoints Partie QCM
RESULTS_FILE = "results.csv"

# --------------------------
# Utils : Sauvegarde résultats
# --------------------------
def save_result(user, quiz_type, score, total, answers):
    file_exists = os.path.isfile(RESULTS_FILE)
    with open(RESULTS_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "user", "quiz_type", "score", "total", "answers"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user,
            quiz_type,
            score,
            total,
            answers
        ])


# --------------------------
# Endpoint : Questions QCM (20 aléatoires, sans réponses)
# --------------------------
@app.route("/questions_qcm", methods=["GET"])
def get_questions_qcm():
    selected = random.sample(questions_qcm, 20)
    safe_questions = []
    for q in selected:
        shuffled = q["options"][:]
        random.shuffle(shuffled)
        safe_questions.append({
            "id": q["id"],
            "question": q["question"],
            "options": shuffled
        })
    return jsonify(safe_questions)


# --------------------------
# Endpoint : Soumettre réponses QCM
# --------------------------
@app.route("/submit_quiz", methods=["POST"])
def submit_quiz():
    data = request.json
    user = data.get("user", "anonymous")
    user_answers = data.get("answers", [])  # [{ "id": 1, "selected": [0,2] }]

    if not user_answers:
        return jsonify({"error": "Pas de réponses fournies"}), 400

    total_score = 0
    total_questions = len(user_answers)

    for ua in user_answers:
        q = next((q for q in questions_qcm if q["id"] == ua["id"]), None)
        if q:
            if set(ua.get("selected", [])) == set(q["answers"]):
                total_score += 1

    # Sauvegarder score + réponses
    save_result(user, "QCM", total_score, total_questions, user_answers)

    return jsonify({
        "score": total_score,
        "total": total_questions,
        "success_rate": f"{(total_score/total_questions)*100:.2f}%"
    })


# --------------------------
# Endpoint : Questions Code (20 aléatoires, sans test_code)
# --------------------------
@app.route("/questions_code", methods=["GET"])
def get_questions_code():
    selected = random.sample(questions_code, 20)
    safe_questions = [{"id": q["id"], "question": q["question"]} for q in selected]
    return jsonify(safe_questions)


# --------------------------
# Endpoint : Soumettre réponses Code
# --------------------------
@app.route("/submit_code", methods=["POST"])
def submit_code():
    data = request.json
    user = data.get("user", "anonymous")
    user_answers = data.get("answers", [])  # [{ "id": 101, "code": "def est_pair..." }]

    if not user_answers:
        return jsonify({"error": "Pas de code fourni"}), 400

    total_score = 0
    total_questions = len(user_answers)

    for ua in user_answers:
        q = next((q for q in questions_code if q["id"] == ua["id"]), None)
        if q:
            try:
                # Exécuter le code utilisateur + tests
                exec(ua["code"], globals())
                exec(q["test_code"], globals())
                total_score += 1
            except Exception:
                pass  # échec => 0 point

    # Sauvegarder score + réponses
    save_result(user, "CODE", total_score, total_questions, user_answers)

    return jsonify({
        "score": total_score,
        "total": total_questions,
        "success_rate": f"{(total_score/total_questions)*100:.2f}%"
    })


if __name__ == "__main__":
    app.run(debug=True)