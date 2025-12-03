
import os
import sys
import torch
from transformers import logging as hf_logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from rag_formulaire.pipeline import RAGPipeline
from rag_formulaire.notebook_utils import force_cleanup, display_result

# Bloque les warnings "generation flags"
hf_logging.set_verbosity_error()

# On garde un contexte court pour la mémoire
os.environ["RAG_FORM_FINAL_EVIDENCE_K"] = "3"

def main():
    print("Initializing Pipeline...")
    pipeline = RAGPipeline()
    
    test_questions = [
        # --- ✅ PARTIE 1 : Questions sur des documents INDEXÉS (Doivent réussir) ---
        "Qui doit signer le formulaire IMM 5476 pour désigner un représentant ?",
        "Que doit-on déclarer à propos des maladies mentales dans le questionnaire médical IMM 5955 ?",
        "Quels documents peuvent servir de preuve d'expérience de travail au Canada selon le formulaire IMM 0134 ?",
        "À qui s'adresse l'offre d'emploi pour les ressortissants étrangers dispensés d'EIMT (IMM 0116) ?",
        "Quel est le rôle de l'interprète décrit dans le formulaire IMM 5744 ?",
        "Quelles informations l'employeur doit-il fournir sur l'adresse commerciale dans le formulaire IMM 0267 ?",
        "Quelles sont les responsabilités de l'employeur concernant l'offre d'emploi dans le formulaire IMM 0273 ?",
        "Dans quel cas les frais relatifs au droit de résidence permanente sont-ils remboursés (IMM 5741) ?",

        # --- ❌ PARTIE 2 : Questions "Test de Sécurité" (Documents ABSENTS) ---
        "Quels sont les documents requis dans la liste de contrôle IMM 5488 ?",
        "Qui doit être listé dans le formulaire de renseignements sur la famille IMM 5707 ?"
    ]

    print(f"🚀 Lancement du test hybride ({len(test_questions)} questions)...\n")
    force_cleanup()

    for i, question in enumerate(test_questions, 1):
        print(f"▶️ Question {i}/{len(test_questions)}: {question}")
        try:
            # Appel du pipeline
            result = pipeline.ask_question(question)

            # Affichage simplifié pour le log console
            answer = result.get('answer', '')
            evidence = result.get('evidence', [])
            if evidence:
                sources = list(set([c.base_chunk.form_code for c in evidence]))
                print(f"   ✅ Sources : {sources}")
            else:
                print("   ⚠️ Aucun extrait trouvé.")

            # Affichage riche (Markdown) - might just print in terminal
            display_result(result)

        except Exception as e:
            print(f"   ⚠️ Erreur : {e}")

        print("-" * 50)
        force_cleanup()

if __name__ == "__main__":
    main()
