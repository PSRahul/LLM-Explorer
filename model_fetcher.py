import transformers
from transformers import pipeline


def main():

    # Get an Text2Text Generator from Hugging Face Model
    model_generator = pipeline("text2text-generation", model="t5-base")

    # Let's Try Sentence Classification
    print(model_generator("sst2 sentence: Dune 2 is extraordinary!", max_length=20))


if __name__ == "__main__":
    main()
