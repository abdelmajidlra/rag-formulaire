
import os
import sys
import time
import torch
import gc

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from rag_formulaire.pipeline import RAGPipeline

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    else:
        print("CUDA not available.")

def main():
    print("Initializing Pipeline...")
    pipeline = RAGPipeline()
    print_gpu_memory()

    questions = [
        "Qui doit signer le formulaire IMM 5476 ?",
        "Que doit-on déclarer à propos des maladies mentales dans le questionnaire médical IMM 5955 ?",
        "Quels documents peuvent servir de preuve d'expérience de travail au Canada selon le formulaire IMM 0134 ?",
        "À qui s'adresse l'offre d'emploi pour les ressortissants étrangers dispensés d'EIMT (IMM 0116) ?",
        "Quel est le rôle de l'interprète décrit dans le formulaire IMM 5744 ?",
        "Quelles informations l'employeur doit-il fournir sur l'adresse commerciale dans le formulaire IMM 0267 ?",
        "Comment remplir la section des antécédents dans le formulaire IMM 5669 ?",
        "Quels sont les frais pour une demande de parrainage (IMM 1344) ?",
        "Où envoyer la demande de citoyenneté (IMM 0002) ?",
        "Qui peut agir comme répondant pour un réfugié (IMM 5373) ?"
    ]

    print(f"Starting stress test with {len(questions)} questions...")
    
    for i, q in enumerate(questions):
        print(f"\n--- Question {i+1}/{len(questions)}: {q} ---")
        start_time = time.time()
        try:
            result = pipeline.ask_question(q)
            elapsed = time.time() - start_time
            print(f"Answer generated in {elapsed:.2f}s")
            # print(f"Answer: {result['answer'][:100]}...")
        except Exception as e:
            print(f"ERROR on question {i+1}: {e}")
        
        print_gpu_memory()

    print("\nTest completed.")

if __name__ == "__main__":
    main()
