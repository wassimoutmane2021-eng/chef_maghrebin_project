import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import re

# Télécharger les ressources NLTK
nltk.download('punkt')
nltk.download('stopwords')

# --- Prétraitement du texte ---
def preprocess(text):
    """Nettoie le texte : ponctuation, stopwords, minuscules."""
    # Supprimer ponctuation et nombres
    text = re.sub(r'[^\w\s]', ' ', text)  # Remplace la ponctuation par un espace
    text = re.sub(r'\d+', '', text)       # Supprime les nombres

    # Minuscules + tokenisation
    words = word_tokenize(text.lower())

    # Supprimer les stopwords (français + mots arabes translittérés)
    stop_words = set(stopwords.words('french'))
    stop_words.update(['le', 'la', 'les', 'de', 'des', 'et', 'à', 'au', 'aux', 'du', 'un', 'une', 'pour', 'avec'])
    words = [word for word in words if word not in stop_words and len(word) > 2]  # Ignore les mots trop courts

    return ' '.join(words)

# --- Charger les recettes ---
def load_recipes(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# --- Extraire le titre d'une recette ---
def extract_title(recipe):
    """Extrait le titre d'une recette (ex: **1. Couscous aux légumes**)."""
    match = re.search(r'\*\*(.*?)\*\*', recipe)
    return match.group(1).strip() if match else "Recette sans titre"

# --- Trouver les recettes pertinentes ---
def get_recipe(query, text, top_n=1):
    """
    Retourne les recettes complètes les plus pertinentes pour la requête.
    Les recettes sont séparées par '---' dans le fichier texte.
    """
    # Séparer le texte en recettes individuelles
    recipes = [r.strip() for r in text.split('---') if r.strip()]

    # Prétraitement
    processed_query = preprocess(query)
    processed_recipes = [preprocess(r) for r in recipes]

    # Vectorisation TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_recipes)
    query_vec = vectorizer.transform([processed_query])

    # Calcul de similitude
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Trier les recettes par pertinence
    ranked_recipes = sorted(zip(recipes, similarities), key=lambda x: x[1], reverse=True)

    # Retourner les top_n recettes
    return [recipe for recipe, score in ranked_recipes[:top_n]]

# --- Fonction principale du chatbot ---
def chatbot(query, text):
    """
    Retourne la recette la plus pertinente pour la requête.
    """
    recipes = get_recipe(query, text, top_n=1)
    if recipes:
        return recipes[0]  # Retourne la recette complète
    else:
        return "Désolé, je ne connais pas cette recette. Essayez une autre formulation (ex: *recette de la chorba*, *comment faire des makrouts*)."

# --- Interface Streamlit ---
def main():
    st.title("🍲 Chatbot des Recettes Maghrébines")
    st.write("Demandez une recette (ex: *Comment faire un tajine d’agneau ?*, *Recette de la brick à l’œuf*).")

    file_path = "recettes_maghrebines.txt"
    try:
        text = load_recipes(file_path)
    except FileNotFoundError:
        st.error("Fichier 'recettes_maghrebines.txt' introuvable. Vérifiez le chemin.")
        return

    user_query = st.text_input("Votre question :", key="query")

    if user_query:
        response = chatbot(user_query, text)
        if not response.startswith("Désolé"):
            st.subheader("✨ Réponse :")
            st.markdown(response)  # Affiche la recette avec mise en forme

            # Option pour voir d'autres recettes similaires
            if st.checkbox("Voir d'autres recettes similaires"):
                similar_recipes = get_recipe(user_query, text, top_n=3)
                for i, recipe in enumerate(similar_recipes, 1):
                    title = extract_title(recipe)
                    st.markdown(f"**{i}. {title}**")
                    if st.button(f"Voir la recette {i}", key=f"btn_{i}"):
                        st.markdown(recipe)
        else:
            st.warning(response)  # Affiche le message d'erreur en orange

if __name__ == "__main__":
    main()
