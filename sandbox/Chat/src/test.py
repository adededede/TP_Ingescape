import re

def analyse(texte):
    # on regarde si l'utilisateur veut creer un musee
    pattern_creation = re.compile(r'(creer)|(crée)|(ouvre)|(ouvrir)')
    # ou bien ajouter un tableau au musee
    pattern_ajout = re.compile(r'(ajout)')
    # ou bien supprimer un tableau au musee
    pattern_suppression = re.compile(r'(enleve)|(enlève)|(supprime)')
    # ou bien fermer le musee
    pattern_fermeture = re.compile(r'(ferme)') 
    if pattern_creation.search(texte):
        creation_musee(texte)
    elif pattern_ajout.search(texte):
        pass
    elif pattern_suppression.search(texte):
        pass
    elif pattern_fermeture.search(texte):
        pass
    else:
        # on ne peut pas interpréter ce message
        print("Votre message n'est pas interprétable, veuillez reformuler svp...")

def creation_musee(texte):
    # on nettoie le message
    pattern = re.compile(r'(musée)|(musee)')
    fin = (pattern.search(texte)).end()
    texte = texte[fin:]
    # on recupere le numero de tableaux a mettre
    pattern = re.compile(r'[0123456789]{1,}')
    nb_tableaux = pattern.search(texte)
    nb_tableaux = nb_tableaux.group(0)
    # on recupere le theme du musée
    list_couleur = re.finditer(r'(tableau)|(couleur)',texte)
    for match in list_couleur:
        fin = match.end()
    couleur = texte[fin+1:]
    couleur = couleur.replace(" ", "")
    nb_tableaux = nb_tableaux.replace(" ", "")
    print("CREATION: nb_tableaux:", nb_tableaux, ", couleur:", couleur,".")

def ajout_musee(texte):
    # le message doit être de type: "(formulation) + (verbe d'ajout) + (formulation) + X +- tableaux +- de couleur + (couleur)"]
    # on recupere le numero de tableaux a mettre
    pattern = re.compile(r'[0123456789]{1,}')
    nb_tableaux = pattern.search(texte)
    nb_tableaux = nb_tableaux.group(0)
    # on recupere le theme du musée
    list_couleur = re.finditer(r'(tableau)|(couleur)',texte)
    for match in list_couleur:
        fin = match.end()
    couleur = texte[fin+1:]
    couleur = couleur.replace(" ", "")
    return "ajout-"+couleur+"-"+nb_tableaux

if __name__ == "__main__":
    # analyse("ouvre un musée de 10 tableaux rouge")
    # analyse("ouvre un musée de 10 tableaux")
    # analyse("je veux ouvrir un musée de 10 tableaux rouge")
    # analyse("je veux ouvrir un musée de 10 tableaux de couleur rouge")
    print(ajout_musee("Ajoute 1 tableau rouge"))

