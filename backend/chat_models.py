import json
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_ollama import ChatOllama


MODEL_NAME = "llama3.2:3b-instruct-fp16"


def format_documents(documents: List[Document]):
    return "\n\n".join(doc.page_content for doc in documents)


class RetrievalAugmentedGenerator:

    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0)
        self.rag_prompt = """Vous êtes un assistant pour des tâches de question-réponse. 

Voici le contexte à utiliser pour répondre à la question :

{context} 

Réfléchissez attentivement au contexte ci-dessus. 

Maintenant, examinez la question de l'utilisateur :

{question}

Fournissez une réponse à cette question en utilisant uniquement le contexte ci-dessus. 

Utilisez un maximum de trois phrases et gardez la réponse concise.

Réponse :"""

    def invoke(self, documents: List[Document], question: str) -> str:
        documents_txt = format_documents(documents)
        rag_prompt_formatted = self.rag_prompt.format(context=documents_txt, question=question)
        result = self.llm.invoke([HumanMessage(content=rag_prompt_formatted)])
        print(result.content)
        return result.content


class QuestionRewriter:

    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0)
        self.prompt = """Regardez l'entrée et essayez de raisonner sur l'intention / signification sous-jacente. \n 
Voici la question initiale :

{question} 

Formulez une question améliorée, soyez concis, seule la nouvelle question doit etre repondu :"""

    def invoke(self, question: str):
        prompt_formatted = self.prompt.format(
            question=question
        )
        result = self.llm.invoke(
            [HumanMessage(content=prompt_formatted)]
        )
        print(result.content)
        return result.content


class RetrievalGrader:

    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0, format="json")
        self.instructions = """Vous êtes un correcteur évaluant la pertinence d'un document récupéré par rapport à une question posée.

Si le document contient des mots-clés ou des informations sémantiques liées à la question, évaluez-le comme pertinent."""
        self.prompt = """Voici le document récupéré : \n\n {document} \n\n Voici la question de l'utilisateur : \n\n {question}. 

Évaluez soigneusement et objectivement si le document contient au moins une information pertinente pour la question.

Retournez un JSON avec une seule clé, `binary_score`, qui est 'oui' ou 'non' pour indiquer si le document contient des informations pertinentes pour la question."""

    def invoke(self, document: Document, question: str):
        prompt_formatted = self.prompt.format(
            document=document, question=question
        )
        result = self.llm.invoke(
            [SystemMessage(content=self.instructions)]
            + [HumanMessage(content=prompt_formatted)]
        )
        print(json.loads(result.content))
        return json.loads(result.content)

class HallucinationGrader:

    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0, format="json")
        self.instructions = """
Vous êtes un enseignant notant un quiz. 

Vous recevrez des FAITS et une RÉPONSE D'ÉTUDIANT. 

Voici les critères de notation à suivre :

(1) Assurez-vous que la RÉPONSE D'ÉTUDIANT est fondée sur les FAITS. 

(2) Assurez-vous que la RÉPONSE D'ÉTUDIANT ne contient pas d'informations "hallucinées" en dehors des FAITS.

Note :

Une note de "oui" signifie que la réponse de l'étudiant respecte tous les critères. C'est la meilleure note. 

Une note de "non" signifie que la réponse de l'étudiant ne respecte pas tous les critères. C'est la note la plus basse.

Expliquez votre raisonnement étape par étape pour justifier votre évaluation. 

Évitez de simplement énoncer la réponse correcte dès le début."""
        self.prompt = """FAITS : \n\n {documents} \n\n RÉPONSE D'ÉTUDIANT : {generation}. 

Retournez un JSON avec deux clés : `binary_score` qui est 'oui' ou 'non' pour indiquer si la RÉPONSE D'ÉTUDIANT est fondée sur les FAITS, et une clé `explanation` qui contient l'explication de la note."""

    def invoke(self, documents: List[Document], generation: str):
        prompt_formatted = self.prompt.format(
            documents=format_documents(documents), generation=generation
        )
        result = self.llm.invoke(
            [SystemMessage(content=self.instructions)]
            + [HumanMessage(content=prompt_formatted)]
        )
        return json.loads(result.content)

class AnswerGrader:

    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0, format="json")
        self.instructions = """Vous êtes un enseignant notant un quiz. 

Vous recevrez une QUESTION et une RÉPONSE D'ÉTUDIANT. 

Voici les critères de notation à suivre :

(1) La RÉPONSE D'ÉTUDIANT aide à répondre à la QUESTION.

Note :

Une note de "oui" signifie que la réponse de l'étudiant respecte tous les critères. C'est la meilleure note. 

L'étudiant peut recevoir une note de "oui" même si la réponse contient des informations supplémentaires qui ne sont pas explicitement demandées dans la question.

Une note de "non" signifie que la réponse de l'étudiant ne respecte pas tous les critères. C'est la note la plus basse.

Expliquez votre raisonnement étape par étape pour justifier votre évaluation. 

Évitez de simplement énoncer la réponse correcte dès le début."""
        self.prompt = """QUESTION : \n\n {question} \n\n RÉPONSE D'ÉTUDIANT : {generation}. 

Retournez un JSON avec deux clés : `binary_score` qui est 'oui' ou 'non' pour indiquer si la RÉPONSE D'ÉTUDIANT respecte les critères, et une clé `explanation` qui contient l'explication de la note."""

    def invoke(self, question: str, generation: str):
        prompt_formatted = self.prompt.format(
            question=question, generation=generation
        )
        result = self.llm.invoke(
            [SystemMessage(content=self.instructions)]
            + [HumanMessage(content=prompt_formatted)]
        )
        return json.loads(result.content)

class Router:

    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0, format="json")
        self.instructions = """Vous êtes un expert dans l'acheminement d'une question utilisateur vers une base de données vectorielle ou une recherche web.

La base de données vectorielle contient des documents relatifs aux résultats de TotalEnergies en 2023.

Utilisez la base de données vectorielle pour les questions sur ces sujets. Pour toutes les autres questions, et surtout celles concernant l'actualité, utilisez la recherche web.

Retournez un JSON avec une seule clé `datasource`, qui est 'websearch' ou 'vectorstore' en fonction de la question."""

    def invoke(self, question: str):
        result = self.llm.invoke(
            [SystemMessage(content=self.instructions)]
            + [HumanMessage(content=question)]
        )
        return json.loads(result.content)
