# Start a local model inference by downloading the llmafile from
# https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#other-example-llamafiles
# For this experiment, I am using TinyLlama-1.1B as the inference engine model

from openai import OpenAI


def main():
    # Let's coonnect to the local language model

    client = OpenAI(
        base_url="http://127.0.0.1:8080/v1",
        api_key="no-key-required",
    )

    content = """
                    I have the following CSV data:

                    name,region
                    Rahul,India
                    Mark,America
                    Schneider,Germany
                    OKenny,Ireland
                    Mieke,Germany """

    completion = client.chat.completions.create(
        model="LLaMA_CPP",
        messages=[
            {
                "role": "system",
                "content": "Generate examples of data similar to the provided input data",
            },
            {"role": "user", "content": content},
        ],
    )
    print(f"Model Used - {completion.model} ")
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
