import transformers
from transformers import pipeline


def main():
    model_generator = pipeline("text2text-generation", model="t5-base", device="cuda:1")
    model_generator("question: How do I push models back to CPU in PyTorch?")


if __name__ == "__main__":
    main()
