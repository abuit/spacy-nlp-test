import logging
import pickle
import random
import spacy
from spacy.util import minibatch, compounding
from spacy.lang.en import English

#https://spacy.io/usage/training
NEW_LABEL = ['source-port', 'target-port', 'cargo']

with open("spacy.dat", 'rb') as fp:
    TRAIN_DATA = pickle.load(fp)

def main():
    try:
        # Create empty model based on English
        nlp = spacy.blank('en')
        
        # Get the entity recognition pipe
        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe(ner)
        # otherwise, get it, so we can add labels to it
        else:
            ner = nlp.get_pipe("ner")

        # Add new entity labels to entity recognizer
        for i in NEW_LABEL:
            ner.add_label(i)  

        optimizer = nlp.begin_training()

        # Get names of other pipes to disable them during training to train only NER
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            for itn in range(100):
                random.shuffle(TRAIN_DATA)
                losses = {}
                batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(
                        texts, 
                        annotations, 
                        sgd=optimizer, 
                        drop=0.35,
                        losses=losses)
                print('Losses', losses)

        # Test the trained model
        test_text = 'Hi sir, from AMS to LON I\'d like to move 12k of salt.'
        doc = nlp(test_text)
        print("Entities in '%s'" % test_text)
        print([(ent.text, ent.label_) for ent in doc.ents])

        test_text = 'Hey there. From Amsterdam to London we will to move 2000 kilograms of grain.'
        doc = nlp(test_text)
        print("Entities in '%s'" % test_text)
        print([(ent.text, ent.label_) for ent in doc.ents])

    except Exception as e:
        logging.exception("Error: " + str(e))
        return None

main()